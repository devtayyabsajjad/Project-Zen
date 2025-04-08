import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, CONTAINER_NAME

# Page config
st.set_page_config(
    page_title="Gold Timeframe Analysis",
    page_icon="🪙",
    layout="wide"
)

# Title with clear explanation
st.title("🪙 Gold Price Analysis - All Timeframes")

# Check Azure connection
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    # Try to list blobs to verify connection
    next(container_client.list_blobs(), None)
    st.success("✅ Successfully connected to Azure Storage")
except Exception as e:
    st.error(f"❌ Failed to connect to Azure Storage: {str(e)}")

st.markdown("""
This tool lets you analyze gold prices across different timeframes.
First, select which timeframe file you want to use from the dropdown menu.
""")

# Function to calculate ATR (Average True Range)
def calculate_atr(df, period=14):
    """Calculate Average True Range for the dataframe."""
    # Make sure we have high, low, close columns
    if 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
        st.error("Cannot calculate ATR: Missing required price columns (High, Low, Close)")
        return df
    
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Calculate True Range
    df_copy['previous_close'] = df_copy['Close'].shift(1)
    df_copy['high-low'] = df_copy['High'] - df_copy['Low']
    df_copy['high-prev_close'] = abs(df_copy['High'] - df_copy['previous_close'])
    df_copy['low-prev_close'] = abs(df_copy['Low'] - df_copy['previous_close'])
    
    # True Range is the maximum of these three values
    df_copy['tr'] = df_copy[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
    
    # Calculate ATR using a simple moving average of the TR
    df_copy['atr'] = df_copy['tr'].rolling(window=period).mean()
    
    # Calculate ATR as a percentage of price
    df_copy['atr_pct'] = (df_copy['atr'] / df_copy['Close']) * 100
    
    # Calculate ATR relative to historical average (for Points System)
    # We use a longer lookback to establish the baseline
    long_period = period * 5  # 5x longer for baseline
    if len(df_copy) >= long_period:
        df_copy['atr_baseline'] = df_copy['atr'].rolling(window=long_period).mean()
        df_copy['atr_ratio'] = df_copy['atr'] / df_copy['atr_baseline']
        
        # Apply the 1-2-3 points system
        # 1: Below average ATR (Low volatility)
        # 2: Average ATR (Normal volatility)
        # 3: Above average ATR (High volatility)
        conditions = [
            df_copy['atr_ratio'] < 0.8,              # Low volatility
            (df_copy['atr_ratio'] >= 0.8) & (df_copy['atr_ratio'] <= 1.2),  # Normal volatility
            df_copy['atr_ratio'] > 1.2               # High volatility
        ]
        choices = [1, 2, 3]
        df_copy['atr_points'] = np.select(conditions, choices, default=2)
    else:
        # Even if we don't have enough data for a baseline, still provide ATR points
        # Using fixed thresholds based on ATR percentage
        df_copy['atr_points'] = 2  # Default to normal volatility
        if 'atr_pct' in df_copy.columns:
            # Define thresholds based on ATR percentage
            low_threshold = df_copy['atr_pct'].quantile(0.25) if len(df_copy) > 4 else 0.5
            high_threshold = df_copy['atr_pct'].quantile(0.75) if len(df_copy) > 4 else 1.5
            
            # Apply the 1-2-3 points system based on percentiles
            conditions = [
                df_copy['atr_pct'] < low_threshold,   # Low volatility
                (df_copy['atr_pct'] >= low_threshold) & (df_copy['atr_pct'] <= high_threshold),  # Normal volatility
                df_copy['atr_pct'] > high_threshold    # High volatility
            ]
            choices = [1, 2, 3]
            df_copy['atr_points'] = np.select(conditions, choices, default=2)
    
    # Drop temporary columns
    df_copy.drop(['previous_close', 'high-low', 'high-prev_close', 'low-prev_close'], axis=1, inplace=True)
    
    return df_copy

# New function to add the 7-level BARCODE classification system
def add_barcode_classification(df):
    """
    Adds the 7-level BARCODE classification system to the dataframe:
    1. Decennial (0-9): Which year in the decade
    2. Presidential (1-4): Year in the presidential term
    3. Quarter (1-4): Quarter of the year
    4. Month (1-12): Month of the year
    5. Week (1-4/5): Week of the month
    6. Day (1-5): Day of the week
    7. Session (1-3): Trading session of the day
    """
    # 1. Decennial classification (0-9)
    df['year'] = df['time'].dt.year
    df['decade'] = (df['year'] // 10) * 10
    df['decennial'] = df['year'] % 10
    
    # 2. Presidential classification (1-4)
    # Starting with 1789 as the first presidential year
    # Presidential cycles start in year after election (traditionally)
    df['presidential'] = ((df['year'] - 1789) % 4) + 1
    
    # 3. Quarter classification (1-4)
    df['quarter'] = df['time'].dt.quarter
    
    # 4. Month classification (1-12)
    df['month'] = df['time'].dt.month
    df['month_name'] = df['time'].dt.strftime('%b')
    
    # 5. Week classification (1-5)
    # Week of month (1-5)
    df['day_of_month'] = df['time'].dt.day
    df['week_of_month'] = ((df['day_of_month'] - 1) // 7) + 1
    
    # 6. Day classification (1-5 for Monday-Friday)
    # 0=Monday, 6=Sunday in pandas, so we adjust to 1-7
    df['day_of_week'] = df['time'].dt.dayofweek + 1
    df['day_name'] = df['time'].dt.strftime('%a')
    
    # 7. Session classification (1-3)
    # For intraday data only
    if 'time' in df.columns and df['time'].dt.hour.nunique() > 1:
        # Define trading sessions based on hour of day (UTC)
        df['hour'] = df['time'].dt.hour
        # Session 1: Asian (00:00-08:00 UTC)
        # Session 2: European (08:00-16:00 UTC)
        # Session 3: American (16:00-24:00 UTC)
        conditions = [
            (df['hour'] >= 0) & (df['hour'] < 8),
            (df['hour'] >= 8) & (df['hour'] < 16),
            (df['hour'] >= 16) & (df['hour'] <= 23)
        ]
        choices = [1, 2, 3]
        df['session'] = np.select(conditions, choices, default=0)
        
        # Add session names for readability
        session_map = {1: 'Asian', 2: 'European', 3: 'American', 0: 'Unknown'}
        df['session_name'] = df['session'].map(session_map)
    
    return df

# Function to load data from any timeframe file
@st.cache_data
def load_data(file_path):
    """Load data from Azure Blob Storage."""
    try:
        # Create a blob service client
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Get a container client
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        # Get a blob client
        blob_client = container_client.get_blob_client(file_path)
        
        # Download the blob content
        download_stream = blob_client.download_blob()
        
        # Read the CSV data directly from the stream
        df = pd.read_csv(download_stream)
        
        # Convert date column
        date_col = 'Date' if 'Date' in df.columns else 'time'
        
        # Convert to datetime
        df['time'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Convert price columns from strings with commas to float
        price_columns = ['Open', 'High', 'Low', 'Close', 'open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Standardize column names (ensure we have uppercase column names)
        if 'open' in df.columns and 'Open' not in df.columns:
            df['Open'] = df['open']
        if 'high' in df.columns and 'High' not in df.columns:
            df['High'] = df['high']
        if 'low' in df.columns and 'Low' not in df.columns:
            df['Low'] = df['low']
        if 'close' in df.columns and 'Close' not in df.columns:
            df['Close'] = df['close']
        
        # Sort by time to ensure data is in chronological order
        df = df.sort_values('time')
        
        # Add BARCODE classification levels
        df = add_barcode_classification(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Available timeframe files
timeframe_files = {
    "Monthly Data (1974+)": "Gold_M_25_74.csv",
    "Weekly Data (1974+)": "Gold_W_25_74.csv",
    "Daily Data (1974+)": "Gold_D_25_74.csv",
    "4-Hour Data (2008+)": "Gold_4h_25_08.csv",
    "1-Hour Data (2008+)": "Gold_1h_25_08.csv",
    "30-Minute Data (2008+)": "Gold_30min_25_08.csv",
    "15-Minute Data (2008+)": "Gold_15min_25_08.csv",
    "10-Minute Data (2008+)": "Gold_10min_25_08.csv",
    "5-Minute Data (2010+)": "Gold_5min_25_10.csv",
    "1-Minute Data (2015+)": "Gold_1min_25_15.csv"
}

# Step 1: Select timeframe file
selected_timeframe = st.selectbox(
    "Step 1: Select which timeframe to analyze:",
    options=list(timeframe_files.keys())
)

file_path = timeframe_files[selected_timeframe]
timeframe_label = selected_timeframe.split()[0]  # Get the timeframe part (1-Minute, Daily, etc.)

# Sidebar for ATR settings
st.sidebar.header("Analysis Settings")
atr_period = st.sidebar.slider("ATR Period", min_value=1, max_value=50, value=14, 
                              help="Number of periods used to calculate ATR")
show_atr_pct = st.sidebar.checkbox("Show ATR as % of Price", value=True, 
                                  help="Display ATR as a percentage of price instead of absolute value")

# Add time window selection for seasonal analysis
st.sidebar.header("Seasonal Analysis Settings")
analysis_years = st.sidebar.slider("Years to Include in Seasonal Analysis", 
                                  min_value=1, max_value=50, value=15,
                                  help="Number of years of data to include in seasonal calculations")

# Add new section for BARCODE profile settings
st.sidebar.header("BARCODE Profile Settings")
show_barcode = st.sidebar.checkbox("Enable BARCODE Profile Analysis", value=True,
                                 help="Enable the 7-level sequential profile analysis")

barcode_level = st.sidebar.selectbox(
    "Select BARCODE Detail Level",
    options=["All Levels", "Decennial", "Presidential", "Quarter", "Month", "Week", "Day", "Session"],
    help="Select which level of the BARCODE profile to focus on"
)

# ATR points system settings
st.sidebar.header("ATR Analysis Settings")
atr_ratio_enabled = st.sidebar.checkbox("Enable ATR Points System (1-2-3 Scale)", value=True,
                                      help="Categorize ATR values on a 1-2-3 scale for easy pattern recognition")

# Load the data
df = load_data(file_path)

if df is not None:
    # Display file information
    st.success(f"✅ Loaded {len(df):,} rows from {file_path}")
    st.info(f"📅 Data ranges from {df['time'].min().date()} to {df['time'].max().date()}")
    
    # Display the raw data sample
    with st.expander(f"👀 See raw {timeframe_label} data sample (first 5 rows)"):
        st.dataframe(df.head())
        
        # Show column descriptions
        st.markdown("""
        **Column Descriptions:**
        - **Date/time**: The timestamp for this data point
        - **Open**: Gold price at the beginning of this time period
        - **High**: Highest gold price during this time period
        - **Low**: Lowest gold price during this time period
        - **Close**: Gold price at the end of this time period
        - **Volume**: Number of contracts traded (if available)
        """)
    
    # Step 2: Select time period to analyze
    st.subheader(f"Step 2: Select time period to analyze")
    
    available_years = sorted(df['time'].dt.year.unique(), reverse=True)
    
    # Allow different selection methods
    period_selection = st.radio(
        "Select by:",
        options=["Recent Data", "Specific Year", "Custom Date Range"]
    )
    
    if period_selection == "Recent Data":
        recent_options = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 180 days": 180,
            "Last year": 365,
            "Last 3 years": 1095,
            "Last 5 years": 1825,
            "All data": None
        }
        recent_period = st.selectbox("Select period:", options=list(recent_options.keys()))
        days_to_include = recent_options[recent_period]
        
        if days_to_include is not None:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_include)
            filtered_df = df[df['time'] >= cutoff_date].copy()
        else:
            filtered_df = df.copy()  # All data
            
    elif period_selection == "Specific Year":
        selected_year = st.selectbox("Select year:", options=available_years)
        filtered_df = df[df['time'].dt.year == selected_year].copy()
        
    else:  # Custom Date Range
        min_date = df['time'].min().date()
        max_date = df['time'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=max_date - pd.Timedelta(days=30),
                                      min_value=min_date, 
                                      max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", 
                                    value=max_date,
                                    min_value=min_date, 
                                    max_value=max_date)
            
        filtered_df = df[
            (df['time'].dt.date >= start_date) & 
            (df['time'].dt.date <= end_date)
        ].copy()
    
    # Check if we have data for the selected period
    if filtered_df.empty:
        st.warning(f"⚠️ No data available for the selected time period. Try a different selection.")
    else:
        st.success(f"✅ Found {len(filtered_df):,} {timeframe_label} data points from {filtered_df['time'].min().date()} to {filtered_df['time'].max().date()}")
        
        # Calculate ATR
        filtered_df = calculate_atr(filtered_df, period=atr_period)
        
        # Step 3: Calculate and display key metrics
        st.subheader("Step 3: Key Performance Metrics")
        
        # Calculate performance metrics
        first_close = filtered_df['Close'].iloc[0]   # First entry in filtered data
        last_close = filtered_df['Close'].iloc[-1]   # Last entry in filtered data
        total_return = ((last_close / first_close) - 1) * 100
        
        # Calculate green vs red candles
        filtered_df['candle_color'] = np.where(filtered_df['Close'] >= filtered_df['Open'], 'green', 'red')
        green_candles = (filtered_df['candle_color'] == 'green').sum()
        red_candles = (filtered_df['candle_color'] == 'red').sum()
        total_candles = len(filtered_df)
        green_pct = (green_candles / total_candles) * 100 if total_candles > 0 else 0
        
        # Calculate ATR stats
        current_atr = filtered_df['atr'].iloc[-1] if 'atr' in filtered_df.columns and not filtered_df['atr'].isna().all() else 0
        current_atr_pct = filtered_df['atr_pct'].iloc[-1] if 'atr_pct' in filtered_df.columns and not filtered_df['atr_pct'].isna().all() else 0
        
        # Display key metrics clearly
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Start Price", f"${first_close:.2f}", 
                     f"{filtered_df['time'].iloc[0].date()}")
            st.metric("Total Return", f"{total_return:.2f}%")
        with col2:
            st.metric("End Price", f"${last_close:.2f}", 
                     f"{filtered_df['time'].iloc[-1].date()}")
            st.metric("Green Candles", f"{green_pct:.1f}%", 
                     f"{green_candles} of {total_candles} candles")
        with col3:
            if show_atr_pct:
                st.metric(f"Current ATR ({atr_period})", f"{current_atr_pct:.2f}%", 
                        "% of Price")
                st.info(f"This means price typically moves ±{current_atr_pct:.2f}% over {atr_period} periods")
            else:
                st.metric(f"Current ATR ({atr_period})", f"${current_atr:.2f}", 
                        "Absolute Value")
                st.info(f"This means price typically moves ±${current_atr:.2f} over {atr_period} periods")
        
        # Step 4: Display price chart with ATR
        st.subheader(f"Step 4: {timeframe_label} Price Chart with ATR")
        
        # Create tabs for different chart views
        chart_tab1, chart_tab2 = st.tabs(["Price Chart", "ATR Chart"])
        
        with chart_tab1:
            # Create a price chart with TradingView style
            if 'Volume' in filtered_df.columns and not filtered_df['Volume'].isna().all():
                # Create subplot with volume
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.05,
                                   row_heights=[0.8, 0.2])
                
                # Add candlestick chart to top subplot
                fig.add_trace(go.Candlestick(
                    x=filtered_df['time'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name='Price',
                    increasing_line_color='#26A69A',  # TradingView green
                    decreasing_line_color='#EF5350',  # TradingView red
                ), row=1, col=1)
                
                # Add volume as bar chart in bottom subplot
                colors = ['#26A69A' if row['Close'] >= row['Open'] else '#EF5350' for _, row in filtered_df.iterrows()]
                fig.add_trace(go.Bar(
                    x=filtered_df['time'],
                    y=filtered_df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['Close'].rolling(window=20).mean(),
                    name='20 MA',
                    line=dict(color='#2962FF', width=1)  # TradingView blue
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['Close'].rolling(window=50).mean(),
                    name='50 MA',
                    line=dict(color='#FF6D00', width=1)  # TradingView orange
                ), row=1, col=1)
                
                # Update layout for both subplots
                fig.update_layout(
                    title=f"Gold Price - {timeframe_label} Chart (TradingView Style)",
                    xaxis_title="",
                    yaxis_title="Price (USD)",
                    height=700,
                    template="plotly_dark",  # Dark theme like TradingView
                    xaxis_rangeslider_visible=False,  # Hide default rangeslider
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=10, r=10, b=10, t=50)
                )
                
                # Update Y-axis formatting for price
                fig.update_yaxes(
                    title_text="Price (USD)",
                    tickprefix="$",
                    gridcolor="#1e222d",  # TradingView grid color
                    row=1, col=1
                )
                
                # Update Y-axis formatting for volume
                fig.update_yaxes(
                    title_text="Volume",
                    gridcolor="#1e222d",
                    row=2, col=1
                )
                
                # Add custom range selector
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ]),
                            bgcolor="#1e222d",
                            activecolor="#2962FF"
                        ),
                        type="date"
                    )
                )
                
                # Add grid lines with TradingView styling
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#1e222d",
                    zeroline=False
                )
                
            else:
                # If no volume data, create single chart
                fig = go.Figure()
                
                # Add candlestick chart
                fig.add_trace(go.Candlestick(
                    x=filtered_df['time'],
                    open=filtered_df['Open'],
                    high=filtered_df['High'],
                    low=filtered_df['Low'],
                    close=filtered_df['Close'],
                    name='Price',
                    increasing_line_color='#26A69A',  # TradingView green
                    decreasing_line_color='#EF5350',  # TradingView red
                ))
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['Close'].rolling(window=20).mean(),
                    name='20 MA',
                    line=dict(color='#2962FF', width=1)  # TradingView blue
                ))
                
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['Close'].rolling(window=50).mean(),
                    name='50 MA',
                    line=dict(color='#FF6D00', width=1)  # TradingView orange
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Gold Price - {timeframe_label} Chart (TradingView Style)",
                    xaxis_title="",
                    yaxis_title="Price (USD)",
                    height=600,
                    template="plotly_dark",  # Dark theme like TradingView
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=10, r=10, b=10, t=50)
                )
                
                # Add custom range selector
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                dict(count=7, label="1w", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ]),
                            bgcolor="#1e222d",
                            activecolor="#2962FF"
                        ),
                        rangeslider=dict(visible=True, thickness=0.05),
                        type="date"
                    )
                )
                
                # Update Y-axis formatting
                fig.update_yaxes(
                    tickprefix="$",
                    gridcolor="#1e222d"  # TradingView grid color
                )
                
                # Add grid lines with TradingView styling
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="#1e222d",
                    zeroline=False
                )
            
            # Display chart with advanced configuration
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
            })
            
            # Add TradingView-style controls instruction
            st.markdown("""
            <style>
            .tradingview-controls {
                background-color: #1e222d;
                padding: 10px;
                border-radius: 5px;
                color: white;
                margin-bottom: 20px;
            }
            </style>
            <div class="tradingview-controls">
                <strong>Chart Controls:</strong> Use mouse wheel to zoom, drag to pan, double-click to reset, right-click for more options.
                Try drawing tools from the modebar at top-right for technical analysis.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **How to read this chart:**
            - Each candle represents one {timeframe_label.lower()} period
            - Green candles: Price closed HIGHER than it opened (bullish)
            - Red candles: Price closed LOWER than it opened (bearish)
            - The top and bottom "wicks" show the high and low prices during that period
            - Blue line: 20-period moving average
            - Orange line: 50-period moving average
            - Use the time buttons at top to quickly change the time range
            - Draw trendlines and shapes using the toolbar at the top-right
            """)
        
        with chart_tab2:
            # Create ATR chart with TradingView styling
            fig = go.Figure()
            
            # Add ATR line
            if show_atr_pct:
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['atr_pct'],
                    mode='lines',
                    name=f'ATR ({atr_period})',
                    line=dict(color='#B388FF', width=2)  # Purple with TradingView style
                ))
                
                # Add average line
                avg_atr_pct = filtered_df['atr_pct'].mean()
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=[avg_atr_pct] * len(filtered_df),
                    mode='lines',
                    line=dict(color='#FF6D00', width=1, dash='dash'),
                    name=f'Avg ({avg_atr_pct:.2f}%)'
                ))
                
                # Add smoothed ATR line
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['atr_pct'].rolling(window=atr_period*5).mean(),
                    mode='lines',
                    line=dict(color='#26A69A', width=1),
                    name=f'Smoothed ATR'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"ATR (%) - {timeframe_label} Chart - {atr_period} Period",
                    xaxis_title="",
                    yaxis_title="ATR (% of Price)",
                    height=500,
                    template="plotly_dark"  # Dark theme like TradingView
                )
            else:
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['atr'],
                    mode='lines',
                    name=f'ATR ({atr_period})',
                    line=dict(color='#B388FF', width=2)  # Purple with TradingView style
                ))
                
                # Add average line
                avg_atr = filtered_df['atr'].mean()
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=[avg_atr] * len(filtered_df),
                    mode='lines',
                    line=dict(color='#FF6D00', width=1, dash='dash'),
                    name=f'Avg (${avg_atr:.2f})'
                ))
                
                # Add smoothed ATR line
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df['atr'].rolling(window=atr_period*5).mean(),
                    mode='lines',
                    line=dict(color='#26A69A', width=1),
                    name=f'Smoothed ATR'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"ATR ($) - {timeframe_label} Chart - {atr_period} Period",
                    xaxis_title="",
                    yaxis_title="ATR (USD)",
                    height=500,
                    template="plotly_dark"  # Dark theme like TradingView
                )
            
            # Add custom range selector and styling
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ]),
                        bgcolor="#1e222d",
                        activecolor="#2962FF"
                    ),
                    rangeslider=dict(visible=True, thickness=0.05),
                    type="date"
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=10, r=10, b=10, t=50)
            )
            
            # Add grid lines with TradingView styling
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#1e222d",
                zeroline=False
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="#1e222d",
                zeroline=False
            )
            
            # Display chart with advanced configuration
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
            })
            
            # Add TradingView-style controls instruction
            st.markdown("""
            <style>
            .tradingview-controls {
                background-color: #1e222d;
                padding: 10px;
                border-radius: 5px;
                color: white;
                margin-bottom: 20px;
            }
            </style>
            <div class="tradingview-controls">
                <strong>Chart Controls:</strong> Use mouse wheel to zoom, drag to pan, double-click to reset, right-click for more options.
                Try drawing tools from the modebar at top-right for technical analysis.
            </div>
            """, unsafe_allow_html=True)
            
            # Add ATR explanation
            st.markdown("""
            **What is ATR (Average True Range)?**
            
            ATR measures market volatility. It shows how much an asset typically moves over a specific time period.
            
            **How to read this chart:**
            - Higher ATR = More volatility (larger price movements)
            - Lower ATR = Less volatility (smaller price movements)
            - Orange line = Average ATR over the entire period
            - Green line = Smoothed ATR trend
            - Use the time buttons at top to quickly change the time range
            - Draw support/resistance on ATR using the toolbar at the top-right
            """)
            
            # Calculate ATR statistics
            avg_atr = filtered_df['atr'].mean() if 'atr' in filtered_df.columns and not filtered_df['atr'].isna().all() else 0
            avg_atr_pct = filtered_df['atr_pct'].mean() if 'atr_pct' in filtered_df.columns and not filtered_df['atr_pct'].isna().all() else 0
            max_atr = filtered_df['atr'].max() if 'atr' in filtered_df.columns and not filtered_df['atr'].isna().all() else 0
            max_atr_pct = filtered_df['atr_pct'].max() if 'atr_pct' in filtered_df.columns and not filtered_df['atr_pct'].isna().all() else 0
            
            # Display ATR metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if show_atr_pct:
                    st.metric("Average ATR", f"{avg_atr_pct:.2f}%")
                else:
                    st.metric("Average ATR", f"${avg_atr:.2f}")
            
            with col2:
                if show_atr_pct:
                    st.metric("Current ATR", f"{current_atr_pct:.2f}%")
                else:
                    st.metric("Current ATR", f"${current_atr:.2f}")
            
            with col3:
                if show_atr_pct:
                    st.metric("Maximum ATR", f"{max_atr_pct:.2f}%")
                else:
                    st.metric("Maximum ATR", f"${max_atr:.2f}")
            
            # ATR Trading Applications
            st.subheader("ATR Trading Applications")
            
            # Check if ATR columns exist in the dataframe
            has_atr = 'atr' in filtered_df.columns and len(filtered_df) > 0
            has_atr_pct = 'atr_pct' in filtered_df.columns and len(filtered_df) > 0
            
            # Calculate ATR values safely
            current_atr = filtered_df['atr'].iloc[-1] if has_atr else 0
            current_atr_pct = filtered_df['atr_pct'].iloc[-1] if has_atr_pct else 0
            avg_atr = filtered_df['atr'].mean() if has_atr else 0 
            avg_atr_pct = filtered_df['atr_pct'].mean() if has_atr_pct else 0
            
            # Calculate volatility comparison safely
            if avg_atr_pct > 0:
                volatility_comparison = f"{(current_atr_pct/avg_atr_pct*100):.1f}% of normal volatility"
            else:
                volatility_comparison = "N/A (insufficient data)"
            
            st.markdown(f"""
            **1. Stop Loss Placement**
            - Tight Stop: Current Price ± {(current_atr_pct/3):.2f}% (⅓ ATR)
            - Normal Stop: Current Price ± {(current_atr_pct):.2f}% (1 ATR)
            - Wide Stop: Current Price ± {(current_atr_pct*2):.2f}% (2 ATR)
            
            **2. Profit Targets**
            - Conservative: Current Price ± {(current_atr_pct):.2f}% (1 ATR)
            - Moderate: Current Price ± {(current_atr_pct*2):.2f}% (2 ATR)
            - Aggressive: Current Price ± {(current_atr_pct*3):.2f}% (3 ATR)
            
            **3. Volatility Comparison**
            - Current vs. Average: {volatility_comparison}
            """)
        
        # Step 5: Performance Analysis
        st.subheader("Step 5: Performance Analysis")
        
        # Calculate period returns (depends on the timeframe)
        if timeframe_label in ["Daily", "Weekly", "Monthly"]:
            # For these timeframes, we can do month/day of week/week of year analysis
            
            # Add seasonal analysis section based on timeframe
            st.markdown("### Seasonal Patterns Analysis")
            st.info(f"Analyzing patterns over the last {analysis_years} years of data")
            
            # Get data for the seasonal analysis based on the selected number of years
            years_ago = pd.Timestamp.now() - pd.Timedelta(days=365 * analysis_years)
            seasonal_df = df[df['time'] >= years_ago].copy()
            
            if len(seasonal_df) > 0:
                # Add period return calculation based on Open to Close for each period
                seasonal_df['return_pct'] = ((seasonal_df['Close'] - seasonal_df['Open']) / seasonal_df['Open']) * 100
                
                # Add week and month columns for analysis
                seasonal_df['week'] = seasonal_df['time'].dt.isocalendar().week
                seasonal_df['month'] = seasonal_df['time'].dt.month
                seasonal_df['month_name'] = seasonal_df['time'].dt.strftime('%b')
                
                # Create tabs for seasonal analysis
                seasonal_tab1, seasonal_tab2 = st.tabs(["Return by Week of Year", "Return by Month"])
                
                with seasonal_tab1:
                    # Calculate average return by week
                    week_returns = seasonal_df.groupby('week')['return_pct'].mean().reset_index()
                    
                    # Create chart similar to corn futures example
                    fig = go.Figure()
                    
                    # Add bars with conditional formatting (green for positive, red for negative)
                    fig.add_trace(go.Bar(
                        x=week_returns['week'],
                        y=week_returns['return_pct'],
                        marker_color=['green' if x >= 0 else 'red' for x in week_returns['return_pct']],
                        text=week_returns['return_pct'].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto'
                    ))
                    
                    # Calculate average return
                    avg_weekly_return = week_returns['return_pct'].mean()
                    
                    # Add average line
                    fig.add_trace(go.Scatter(
                        x=week_returns['week'],
                        y=[avg_weekly_return] * len(week_returns),
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        name=f'Average ({avg_weekly_return:.2f}%)'
                    ))
                    
                    # Update layout to match corn futures example
                    fig.update_layout(
                        title=f"AVERAGE RETURN BY WEEK<br><sup>Gold / OPEN-CLOSE over {analysis_years} years</sup>",
                        xaxis_title="Week of Year",
                        yaxis_title="%",
                        height=500,
                        template="plotly_white",
                        xaxis=dict(
                            tickmode='linear',
                            tick0=1,
                            dtick=2,
                            range=[0, 54]
                        ),
                        yaxis=dict(
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        ),
                        margin=dict(t=80)
                    )
                    
                    # Add grid lines
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display statistics
                    positive_weeks = sum(1 for x in week_returns['return_pct'] if x > 0)
                    total_weeks = len(week_returns)
                    pct_positive = (positive_weeks / total_weeks) * 100 if total_weeks > 0 else 0
                    
                    best_week = week_returns.loc[week_returns['return_pct'].idxmax()]
                    worst_week = week_returns.loc[week_returns['return_pct'].idxmin()]
                    
                    st.markdown(f"""
                    **Weekly Return Statistics:**
                    - **Average Weekly Return:** {avg_weekly_return:.2f}%
                    - **Positive Weeks:** {positive_weeks} out of {total_weeks} ({pct_positive:.1f}%)
                    - **Best Week:** Week {int(best_week['week'])} ({best_week['return_pct']:.2f}%)
                    - **Worst Week:** Week {int(worst_week['week'])} ({worst_week['return_pct']:.2f}%)
                    """)
                
                with seasonal_tab2:
                    # Calculate average return by month
                    month_returns = seasonal_df.groupby(['month', 'month_name'])['return_pct'].mean().reset_index()
                    month_returns = month_returns.sort_values('month')
                    
                    # Create chart similar to corn futures example
                    fig = go.Figure()
                    
                    # Add bars with conditional formatting
                    fig.add_trace(go.Bar(
                        x=month_returns['month_name'],
                        y=month_returns['return_pct'],
                        marker_color=['green' if x >= 0 else 'red' for x in month_returns['return_pct']],
                        text=month_returns['return_pct'].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto'
                    ))
                    
                    # Calculate average return
                    avg_monthly_return = month_returns['return_pct'].mean()
                    
                    # Add average line
                    fig.add_trace(go.Scatter(
                        x=month_returns['month_name'],
                        y=[avg_monthly_return] * len(month_returns),
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        name=f'Average ({avg_monthly_return:.2f}%)'
                    ))
                    
                    # Update layout to match corn futures example
                    fig.update_layout(
                        title=f"AVERAGE RETURN BY MONTH<br><sup>Gold / OPEN-CLOSE over {analysis_years} years</sup>",
                        xaxis_title="Month",
                        yaxis_title="%",
                        height=500,
                        template="plotly_white",
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        ),
                        yaxis=dict(
                            zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='black'
                        ),
                        margin=dict(t=80)
                    )
                    
                    # Add grid lines
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display statistics
                    positive_months = sum(1 for x in month_returns['return_pct'] if x > 0)
                    total_months = len(month_returns)
                    pct_positive = (positive_months / total_months) * 100 if total_months > 0 else 0
                    
                    best_month = month_returns.loc[month_returns['return_pct'].idxmax()]
                    worst_month = month_returns.loc[month_returns['return_pct'].idxmin()]
                    
                    st.markdown(f"""
                    **Monthly Return Statistics:**
                    - **Average Monthly Return:** {avg_monthly_return:.2f}%
                    - **Positive Months:** {positive_months} out of {total_months} ({pct_positive:.1f}%)
                    - **Best Month:** {best_month['month_name']} ({best_month['return_pct']:.2f}%)
                    - **Worst Month:** {worst_month['month_name']} ({worst_month['return_pct']:.2f}%)
                    """)
                
                # Add hierarchical timeframe analysis explanation
                st.markdown("""
                ### Hierarchical Timeframe Analysis
                
                When analyzing market patterns, it's essential to follow a top-down approach:
                
                1. **Monthly patterns** provide the long-term directional bias
                2. **Weekly patterns** indicate medium-term trends
                3. **Daily patterns** show short-term opportunities
                4. Lower timeframes (4H, 1H, etc.) are for timing entries/exits
                
                Higher timeframes (HTF) take precedence over lower timeframes. If the monthly trend is bullish but the daily is bearish, the monthly trend has more weight in your analysis.
                """)
            else:
                st.warning(f"Not enough historical data for seasonal analysis. Need at least {analysis_years} years of data.")
            
            # Add time period columns for original analysis
            if len(filtered_df) >= 7:  # Need at least a week of data
                filtered_df['day_of_week'] = filtered_df['time'].dt.dayofweek
                filtered_df['day_name'] = filtered_df['time'].dt.day_name()
                
                # Create day of week performance chart
                day_perf = filtered_df.groupby('day_of_week').agg({
                    'day_name': 'first',
                    'candle_color': lambda x: (x == 'green').mean() * 100,
                    'atr_pct': 'mean'
                }).reset_index()
                
                day_perf = day_perf.sort_values('day_of_week')
                
                # Create tabs for different analyses
                analysis_tab1, analysis_tab2 = st.tabs(["Green Day Analysis", "ATR by Day of Week"])
                
                with analysis_tab1:
                    # Create chart
                    fig = go.Figure()
                    
                    # Add bars for day of week performance
                    fig.add_trace(go.Bar(
                        x=day_perf['day_name'],
                        y=day_perf['candle_color'],
                        marker_color=['green' if x >= 50 else 'red' for x in day_perf['candle_color']],
                        text=day_perf['candle_color'].apply(lambda x: f"{x:.1f}%"),
                        textposition='auto'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Percentage of Green Candles by Day of Week",
                        xaxis_title="Day of Week",
                        yaxis_title="Green Candles (%)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **What this chart shows:**
                    - For each day of the week, what percentage of candles were green
                    - Green bars = More than 50% of candles were green on this day
                    - Red bars = Less than 50% of candles were green on this day
                    - Higher bars indicate more consistently green days
                    """)
                
                with analysis_tab2:
                    # Create chart for ATR by day of week
                    fig = go.Figure()
                    
                    # Add bars for day of week ATR
                    fig.add_trace(go.Bar(
                        x=day_perf['day_name'],
                        y=day_perf['atr_pct'],
                        marker_color='purple',
                        text=day_perf['atr_pct'].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Average ATR by Day of Week",
                        xaxis_title="Day of Week",
                        yaxis_title="ATR (% of Price)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **What this chart shows:**
                    - For each day of the week, what's the average ATR (volatility)
                    - Higher bars = More volatile days (larger price movements)
                    - Lower bars = Less volatile days (smaller price movements)
                    - This can help you identify which days typically have the most price movement
                    """)
            
            if timeframe_label in ["Daily", "Weekly"] and len(filtered_df) >= 30:
                # Month analysis
                filtered_df['month'] = filtered_df['time'].dt.month
                filtered_df['month_name'] = filtered_df['time'].dt.strftime('%b')
                
                month_perf = filtered_df.groupby('month').agg({
                    'month_name': 'first',
                    'candle_color': lambda x: (x == 'green').mean() * 100,
                    'atr_pct': 'mean'
                }).reset_index()
                
                month_perf = month_perf.sort_values('month')
                
                # Create tabs for different analyses
                month_tab1, month_tab2 = st.tabs(["Green Month Analysis", "ATR by Month"])
                
                with month_tab1:
                    # Create chart
                    fig = go.Figure()
                    
                    # Add bars for monthly performance
                    fig.add_trace(go.Bar(
                        x=month_perf['month_name'],
                        y=month_perf['candle_color'],
                        marker_color=['green' if x >= 50 else 'red' for x in month_perf['candle_color']],
                        text=month_perf['candle_color'].apply(lambda x: f"{x:.1f}%"),
                        textposition='auto'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Percentage of Green Candles by Month",
                        xaxis_title="Month",
                        yaxis_title="Green Candles (%)",
                        height=400,
                        template="plotly_white",
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **What this chart shows:**
                    - For each month, what percentage of candles were green
                    - Green bars = More than 50% of candles were green in this month
                    - Red bars = Less than 50% of candles were green in this month
                    - Higher bars indicate more consistently green months
                    """)
                
                with month_tab2:
                    # Create chart for ATR by month
                    fig = go.Figure()
                    
                    # Add bars for monthly ATR
                    fig.add_trace(go.Bar(
                        x=month_perf['month_name'],
                        y=month_perf['atr_pct'],
                        marker_color='purple',
                        text=month_perf['atr_pct'].apply(lambda x: f"{x:.2f}%"),
                        textposition='auto'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Average ATR by Month",
                        xaxis_title="Month",
                        yaxis_title="ATR (% of Price)",
                        height=400,
                        template="plotly_white",
                        xaxis=dict(
                            categoryorder='array',
                            categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **What this chart shows:**
                    - For each month, what's the average ATR (volatility)
                    - Higher bars = More volatile months (larger price movements)
                    - Lower bars = Less volatile months (smaller price movements)
                    - This can help you identify seasonal volatility patterns
                    """)
        
        # Step 6: Winning streak analysis
        st.subheader("Step 6: Streak Analysis")
        
        # Calculate streaks
        filtered_df['streak_change'] = filtered_df['candle_color'].ne(filtered_df['candle_color'].shift()).cumsum()
        
        # Group streaks
        streaks = filtered_df.groupby(['candle_color', 'streak_change']).size().reset_index(name='length')
        
        # Get max streaks
        max_green_streak = streaks[streaks['candle_color'] == 'green']['length'].max() if not streaks[streaks['candle_color'] == 'green'].empty else 0
        max_red_streak = streaks[streaks['candle_color'] == 'red']['length'].max() if not streaks[streaks['candle_color'] == 'red'].empty else 0
        
        # Get current streak
        last_color = filtered_df.iloc[-1]['candle_color']
        last_streak = filtered_df[filtered_df['streak_change'] == filtered_df.iloc[-1]['streak_change']].shape[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Longest Green Streak", f"{max_green_streak} candles")
        with col2:
            st.metric("Longest Red Streak", f"{max_red_streak} candles")
        with col3:
            st.metric("Current Streak", f"{last_streak} {last_color} candles")
        
        # Display streak distribution
        green_streaks = streaks[streaks['candle_color'] == 'green']['length'].value_counts().sort_index()
        red_streaks = streaks[streaks['candle_color'] == 'red']['length'].value_counts().sort_index()
        
        # Convert to DataFrames for display
        green_df = pd.DataFrame({
            'Streak Length': green_streaks.index,
            'Frequency': green_streaks.values
        })
        
        red_df = pd.DataFrame({
            'Streak Length': red_streaks.index,
            'Frequency': red_streaks.values
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Green Streak Distribution")
            st.dataframe(green_df.sort_values('Streak Length'))
        with col2:
            st.subheader("Red Streak Distribution")
            st.dataframe(red_df.sort_values('Streak Length'))
        
        st.markdown("""
        **What this means:**
        - **Streak Length**: How many consecutive green or red candles
        - **Frequency**: How many times this streak length occurred
        - This shows you the pattern of consecutive winning and losing periods
        """)
            
else:
    st.error(f"Failed to load data from {file_path}. Please check if the file exists.")

# Add sidebar help
st.sidebar.title("How to Use This Tool")
st.sidebar.markdown("""
### Steps:
1. **Select a timeframe file** - Choose which data file to analyze (1min, 5min, daily, etc.)
2. **Choose a time period** - Select what date range to focus on
3. **Review the metrics** - See the key performance data
4. **Explore the charts** - Analyze price movement and patterns
5. **Check the streak analysis** - See patterns of consecutive green/red candles

### ATR (Average True Range):
- **What it is**: A measure of market volatility
- **Higher ATR**: More volatility, larger price movements
- **Lower ATR**: Less volatility, smaller price movements
- **Uses**: Setting stop losses, profit targets, identifying potential breakouts

### Timeframe Descriptions:
- **1-Minute**: Each candle = 1 minute of price movement
- **5-Minute**: Each candle = 5 minutes of price movement
- **Daily**: Each candle = 1 trading day
- **Weekly**: Each candle = 1 trading week
- **Monthly**: Each candle = 1 trading month
""")

st.sidebar.info("The calculations are based on the open and close prices from each specific timeframe file.") 

# Display BARCODE classification if enabled
if show_barcode:
    with st.expander("🔍 BARCODE Classification System"):
        st.markdown("""
        ## BARCODE Classification System
        
        The BARCODE system classifies each data point in 7 levels:
        1. **Decennial (0-9)**: Position in decade (0=year ending in 0, 9=year ending in 9)
        2. **Presidential (1-4)**: Year in presidential term (1=first year, 4=fourth year)
        3. **Quarter (1-4)**: Quarter of year (1=Q1, 4=Q4)
        4. **Month (1-12)**: Month of year (1=Jan, 12=Dec)
        5. **Week (1-5)**: Week of month (1=first week, 5=fifth week if exists)
        6. **Day (1-7)**: Day of week (1=Monday, 7=Sunday)
        7. **Session (1-3)**: Trading session (1=Asian, 2=European, 3=American)
        
        This classification helps identify patterns across different time cycles.
        """)
        
        # Show a sample of the classification for current data
        barcode_sample = df[['time', 'decennial', 'presidential', 'quarter', 
                            'month', 'month_name', 'week_of_month', 'day_of_week', 'day_name']]
        if 'session' in df.columns:
            barcode_sample = pd.concat([barcode_sample, df[['session', 'session_name']]], axis=1)
        
        st.dataframe(barcode_sample.head(10))

# Add new BARCODE profile visualization section after the Step 5: Performance Analysis section
if show_barcode and len(filtered_df) > 0:
    st.header("BARCODE Sequential Profile Analysis")
    st.markdown("""
    This analysis examines market patterns across different time cycles using the BARCODE classification system.
    Each level provides insights into how gold performs during specific time periods.
    """)
    
    # Create tabs for different BARCODE levels
    barcode_tabs = st.tabs([
        "Decennial", "Presidential", "Quarter", "Month", "Week", "Day", 
        "Session" if 'session' in filtered_df.columns else "Hour"
    ])
    
    # Define a premium color palette
    colors = {
        'green': '#00BD9D',        # Teal green for positive returns
        'red': '#FF5C75',          # Coral red for negative returns
        'blue': '#4A6FE3',         # Royal blue for normal periods
        'gold': '#FFD700',         # Gold for highlighting
        'purple': '#9D6EFD',       # Purple for special insights
        'background': '#131722',   # Dark background like TradingView
        'grid': '#363A45',         # Grid lines
        'text': '#D1D4DC'          # Light text
    }
    
    # Function to create premium-styled barcode charts
    def create_barcode_chart(df, group_col, name_col=None, title=None, show_average=True):
        """Create a premium-styled bar chart for BARCODE analysis"""
        if name_col is None:
            name_col = group_col
        
        # Function to safely get a column or provide a default
        def safe_get_column(df, col_name, default_value=0):
            if col_name in df.columns:
                return df[col_name]
            else:
                return pd.Series([default_value] * len(df))
        
        # Calculate metrics for each group
        metrics = df.groupby(group_col).agg({
            'candle_color': [
                ('green_pct', lambda x: (x == 'green').mean() * 100),
                ('count', 'count')
            ],
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Low': 'min'
        })
        
        # Conditionally add ATR metrics if available
        if 'atr' in df.columns:
            atr_metrics = df.groupby(group_col).agg({
                'atr': 'mean'
            })
            metrics = pd.concat([metrics, atr_metrics], axis=1)
        
        if 'atr_pct' in df.columns:
            atr_pct_metrics = df.groupby(group_col).agg({
                'atr_pct': 'mean'
            })
            metrics = pd.concat([metrics, atr_pct_metrics], axis=1)
        
        # Flatten the multi-index columns
        metrics.columns = ['_'.join(col).strip('_') for col in metrics.columns.values]
        
        # Calculate returns - safely handle column names
        if 'Close' in metrics.columns and 'Open' in metrics.columns and not metrics['Open'].isnull().all():
            metrics['return_pct'] = ((metrics['Close'] - metrics['Open']) / metrics['Open']) * 100
        elif 'Close_last' in metrics.columns and 'Open_first' in metrics.columns and not metrics['Open_first'].isnull().all():
            metrics['return_pct'] = ((metrics['Close_last'] - metrics['Open_first']) / metrics['Open_first']) * 100
        else:
            # If we can't calculate returns, create empty column
            metrics['return_pct'] = 0
        
        # Add names if available
        if name_col in df.columns and name_col != group_col:
            # Get the most common name for each group
            names = df.groupby(group_col)[name_col].first()
            metrics = metrics.join(names.to_frame())
        
        # Sort by the group column
        metrics = metrics.reset_index().sort_values(group_col)
        
        # Create a figure with two subplots
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Red/Green Probability (%)", "Avg. Return (%)"),
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )
        
        # Add green/red probability bars
        fig.add_trace(
            go.Bar(
                x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                y=metrics['candle_color_green_pct'],
                name="Green Probability (%)",
                marker_color=[colors['green'] if val > 50 else colors['red'] for val in metrics['candle_color_green_pct']],
                text=metrics['candle_color_green_pct'].round(1).astype(str) + '%',
                textposition='auto',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Add a 50% reference line
        fig.add_trace(
            go.Scatter(
                x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                y=[50] * len(metrics),
                mode='lines',
                line=dict(color=colors['grid'], width=1, dash='dash'),
                name="50% Line",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add ATR visualization
        # Always attempt to show the ATR information, with fallbacks
        atr_data = None
        
        # First try to use atr_pct if available
        if 'atr_pct' in metrics.columns:
            atr_data = metrics['atr_pct']
            atr_name = "ATR (%)"
        # If not, try to use regular atr with a scaled value
        elif 'atr' in metrics.columns:
            # Scale ATR to make it visible on the same chart
            avg_price = metrics['Close'].mean() if 'Close' in metrics.columns else (metrics['Close_last'].mean() if 'Close_last' in metrics.columns else 100)
            atr_data = metrics['atr'] / avg_price * 100
            atr_name = "ATR (scaled %)"
            
        if atr_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                    y=atr_data,
                    mode='lines+markers',
                    line=dict(color=colors['purple'], width=2),
                    name=atr_name,
                    marker=dict(
                        size=8,
                        symbol='diamond'
                    )
                ),
                row=1, col=1,
                secondary_y=True
            )
            
            # If atr_points is available, add markers with different colors based on points
            if 'atr_points' in metrics.columns:
                # Separate points into different traces for better styling
                for point_val in [1, 2, 3]:
                    mask = metrics['atr_points'] == point_val
                    if mask.any():
                        point_names = {1: "Low Vol (1)", 2: "Normal Vol (2)", 3: "High Vol (3)"}
                        point_colors = {1: colors['blue'], 2: colors['gold'], 3: colors['red']}
                        point_symbols = {1: "circle", 2: "square", 3: "triangle-up"}
                        
                        # Add marker for this point value
                        fig.add_trace(
                            go.Scatter(
                                x=metrics.loc[mask, name_col] if name_col in metrics.columns else metrics.loc[mask, group_col],
                                y=atr_data[mask],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color=point_colors[point_val],
                                    symbol=point_symbols[point_val],
                                    line=dict(width=2, color='white')
                                ),
                                name=point_names[point_val],
                                showlegend=True
                            ),
                            row=1, col=1,
                            secondary_y=True
                        )
        
        # Add return bars
        fig.add_trace(
            go.Bar(
                x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                y=metrics['return_pct'],
                name="Avg. Return (%)",
                marker_color=[colors['green'] if val > 0 else colors['red'] for val in metrics['return_pct']],
                text=metrics['return_pct'].round(2).astype(str) + '%',
                textposition='auto',
                opacity=0.8
            ),
            row=2, col=1
        )
        
        # Add a zero line for returns
        fig.add_trace(
            go.Scatter(
                x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                y=[0] * len(metrics),
                mode='lines',
                line=dict(color=colors['grid'], width=1, dash='dash'),
                name="Zero Line",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add sample count as text
        fig.add_trace(
            go.Scatter(
                x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                y=metrics['candle_color_count'],
                mode='text',
                text="n=" + metrics['candle_color_count'].astype(str),
                textposition='top center',
                textfont=dict(size=8, color=colors['text']),
                name="Sample Size",
                showlegend=False
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        # Add overall average line if requested
        if show_average:
            avg_green = metrics['candle_color_green_pct'].mean()
            avg_return = metrics['return_pct'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                    y=[avg_green] * len(metrics),
                    mode='lines',
                    line=dict(color=colors['gold'], width=2, dash='dot'),
                    name=f"Avg Green %: {avg_green:.1f}%",
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
                    y=[avg_return] * len(metrics),
                    mode='lines',
                    line=dict(color=colors['gold'], width=2, dash='dot'),
                    name=f"Avg Return: {avg_return:.2f}%",
                ),
                row=2, col=1
            )
        
        # Update layout for premium look
        fig.update_layout(
            title=title if title else f"{group_col.title()} Profile Analysis",
            template="plotly_dark",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color=colors['text']
            ),
            paper_bgcolor=colors['background'],
            plot_bgcolor=colors['background'],
            margin=dict(l=40, r=40, b=40, t=80)
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Green Candle %", secondary_y=False, row=1, col=1, 
                        gridcolor=colors['grid'], tickformat='.1f')
        if atr_data is not None:
            fig.update_yaxes(title_text=atr_name, secondary_y=True, row=1, col=1, 
                            gridcolor=colors['grid'], tickformat='.2f')
        fig.update_yaxes(title_text="Return %", secondary_y=False, row=2, col=1, 
                        gridcolor=colors['grid'], tickformat='.2f')
        fig.update_yaxes(title_text="Sample Count", secondary_y=True, row=2, col=1, 
                        gridcolor=colors['grid'], visible=False)
        
        # Update x-axis
        fig.update_xaxes(
            categoryorder='array',
            categoryarray=metrics[name_col] if name_col in metrics.columns else metrics[group_col],
            gridcolor=colors['grid']
        )
        
        return fig, metrics
    
    # Generate charts for each BARCODE level
    # 1. Decennial Analysis
    with barcode_tabs[0]:
        st.subheader("Decennial Pattern Analysis (Years 0-9)")
        st.markdown("""
        This analysis shows how gold typically performs in different years of a decade.
        For example, years ending in 0 (like 2020), years ending in 1 (like 2021), etc.
        """)
        
        fig, decennial_metrics = create_barcode_chart(
            filtered_df, 
            'decennial', 
            title="Gold Performance by Decennial Pattern (Years 0-9)"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': 'decennial_pattern'}
        })
        
        # Show data table
        with st.expander("See Decennial Pattern Data"):
            st.dataframe(decennial_metrics)
    
    # 2. Presidential Analysis
    with barcode_tabs[1]:
        st.subheader("Presidential Cycle Analysis (Years 1-4)")
        st.markdown("""
        This analysis shows how gold typically performs in different years of the 4-year presidential cycle.
        Year 1 represents the first year after an election, and Year 4 represents an election year.
        """)
        
        # Map presidential years to more descriptive names
        presidential_map = {
            1: "Post-Election (Year 1)",
            2: "Midterm (Year 2)",
            3: "Pre-Election (Year 3)",
            4: "Election (Year 4)"
        }
        filtered_df['presidential_name'] = filtered_df['presidential'].map(presidential_map)
        
        fig, presidential_metrics = create_barcode_chart(
            filtered_df, 
            'presidential', 
            'presidential_name',
            title="Gold Performance by Presidential Cycle (Years 1-4)"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': 'presidential_cycle'}
        })
        
        # Show data table
        with st.expander("See Presidential Cycle Data"):
            st.dataframe(presidential_metrics)
    
    # 3. Quarter Analysis
    with barcode_tabs[2]:
        st.subheader("Quarterly Pattern Analysis (Q1-Q4)")
        st.markdown("""
        This analysis shows how gold typically performs in different quarters of the year.
        Patterns may reveal seasonal tendencies that repeat annually.
        """)
        
        # Map quarters to descriptive names
        quarter_map = {
            1: "Q1 (Jan-Mar)",
            2: "Q2 (Apr-Jun)",
            3: "Q3 (Jul-Sep)",
            4: "Q4 (Oct-Dec)"
        }
        filtered_df['quarter_name'] = filtered_df['quarter'].map(quarter_map)
        
        fig, quarter_metrics = create_barcode_chart(
            filtered_df, 
            'quarter', 
            'quarter_name',
            title="Gold Performance by Quarter (Q1-Q4)"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': 'quarter_pattern'}
        })
        
        # Show data table
        with st.expander("See Quarterly Pattern Data"):
            st.dataframe(quarter_metrics)
    
    # 4. Month Analysis
    with barcode_tabs[3]:
        st.subheader("Monthly Pattern Analysis (Jan-Dec)")
        st.markdown("""
        This analysis shows how gold typically performs in different months of the year.
        Patterns may reveal seasonal tendencies that repeat annually.
        """)
        
        # Create month order for proper display
        month_order = list(range(1, 13))
        
        fig, month_metrics = create_barcode_chart(
            filtered_df, 
            'month', 
            'month_name',
            title="Gold Performance by Month of Year"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': 'month_pattern'}
        })
        
        # Show heat map of monthly performance over years
        st.subheader("Monthly Performance Heat Map")
        
        if len(filtered_df['year'].unique()) > 1:
            # Create a pivot table of returns by year and month
            monthly_pivot = pd.pivot_table(
                filtered_df,
                values='Close',
                index='year',
                columns='month_name',
                aggfunc=lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0]) * 100  # Return calculation
            )
            
            # Reorder columns to Jan-Dec
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly_pivot = monthly_pivot.reindex(columns=month_names)
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=monthly_pivot.values,
                x=monthly_pivot.columns,
                y=monthly_pivot.index,
                colorscale=[
                    [0, colors['red']],
                    [0.5, colors['background']],
                    [1, colors['green']]
                ],
                text=monthly_pivot.round(2).values,
                texttemplate="%{text:.2f}%",
                colorbar=dict(title="Return %"),
                hoverongaps=False,
                hoverinfo="x+y+z+text"
            ))
            
            fig_heatmap.update_layout(
                title="Monthly Returns by Year (%)",
                xaxis_title="Month",
                yaxis_title="Year",
                height=400,
                template="plotly_dark",
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color=colors['text']
                ),
                paper_bgcolor=colors['background'],
                plot_bgcolor=colors['background']
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Show data table
        with st.expander("See Monthly Pattern Data"):
            st.dataframe(month_metrics)
    
    # 5. Week of Month Analysis
    with barcode_tabs[4]:
        st.subheader("Weekly Pattern Analysis (Weeks 1-5)")
        st.markdown("""
        This analysis shows how gold typically performs in different weeks of a month.
        Week 1 is days 1-7, Week 2 is days 8-14, and so on.
        """)
        
        # Map weeks to more descriptive names
        week_map = {
            1: "Week 1 (days 1-7)",
            2: "Week 2 (days 8-14)",
            3: "Week 3 (days 15-21)",
            4: "Week 4 (days 22-28)",
            5: "Week 5 (days 29-31)"
        }
        filtered_df['week_name'] = filtered_df['week_of_month'].map(week_map)
        
        fig, week_metrics = create_barcode_chart(
            filtered_df, 
            'week_of_month', 
            'week_name',
            title="Gold Performance by Week of Month"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': 'week_pattern'}
        })
        
        # Show data table
        with st.expander("See Weekly Pattern Data"):
            st.dataframe(week_metrics)
    
    # 6. Day of Week Analysis
    with barcode_tabs[5]:
        st.subheader("Daily Pattern Analysis (Mon-Sun)")
        st.markdown("""
        This analysis shows how gold typically performs on different days of the week.
        Patterns may reveal which days tend to be more bullish or bearish.
        """)
        
        # Create day order for proper display (1=Monday to 7=Sunday)
        day_order = list(range(1, 8))
        
        fig, day_metrics = create_barcode_chart(
            filtered_df, 
            'day_of_week', 
            'day_name',
            title="Gold Performance by Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': 'day_pattern'}
        })
        
        # Show data table
        with st.expander("See Daily Pattern Data"):
            st.dataframe(day_metrics)
    
    # 7. Session Analysis (for intraday data)
    with barcode_tabs[6]:
        if 'session' in filtered_df.columns:
            st.subheader("Trading Session Analysis (Asian, European, American)")
            st.markdown("""
            This analysis shows how gold typically performs during different trading sessions:
            - Session 1: Asian session (00:00-08:00 UTC)
            - Session 2: European session (08:00-16:00 UTC)
            - Session 3: American session (16:00-24:00 UTC)
            """)
            
            fig, session_metrics = create_barcode_chart(
                filtered_df, 
                'session', 
                'session_name',
                title="Gold Performance by Trading Session"
            )
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {'format': 'png', 'filename': 'session_pattern'}
            })
            
            # Add session transition analysis
            st.subheader("Session Transition Analysis")
            st.markdown("""
            This analysis examines how gold performs when transitioning between trading sessions.
            It can reveal patterns that occur at session handovers.
            """)
            
            # Create session transition column (previous session to current session)
            filtered_df['prev_session'] = filtered_df['session'].shift(1)
            filtered_df['session_transition'] = filtered_df['prev_session'].astype(str) + "->" + filtered_df['session'].astype(str)
            
            # Filter out rows with missing previous session (e.g., first row)
            transition_df = filtered_df.dropna(subset=['prev_session'])
            
            # Filter out invalid transitions (e.g., day changes)
            valid_transitions = ["1->2", "2->3", "3->1"]  # Asian->European, European->American, American->Asian
            transition_df = transition_df[transition_df['session_transition'].isin(valid_transitions)]
            
            # Map transition codes to descriptive names
            transition_map = {
                "1->2": "Asian → European",
                "2->3": "European → American",
                "3->1": "American → Asian"
            }
            transition_df['transition_name'] = transition_df['session_transition'].map(transition_map)
            
            if len(transition_df) > 0:
                fig, transition_metrics = create_barcode_chart(
                    transition_df, 
                    'session_transition', 
                    'transition_name',
                    title="Gold Performance During Session Transitions"
                )
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {'format': 'png', 'filename': 'session_transition'}
                })
                
                # Show data table
                with st.expander("See Session Transition Data"):
                    st.dataframe(transition_metrics)
            else:
                st.info("Not enough data for session transition analysis with current filter settings.")
            
            # Show data table for sessions
            with st.expander("See Trading Session Data"):
                st.dataframe(session_metrics)
        else:
            # If no session data available, show hourly analysis instead
            st.subheader("Hourly Pattern Analysis (Hours 0-23)")
            st.markdown("""
            For timeframes that have hour data but no session classification,
            this analysis shows how gold performs during different hours of the day (UTC time).
            """)
            
            if 'hour' in filtered_df.columns:
                # Format hours for better display
                filtered_df['hour_formatted'] = filtered_df['hour'].apply(lambda x: f"{x:02d}:00 UTC")
                
                fig, hour_metrics = create_barcode_chart(
                    filtered_df, 
                    'hour', 
                    'hour_formatted',
                    title="Gold Performance by Hour of Day (UTC)"
                )
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {'format': 'png', 'filename': 'hour_pattern'}
                })
                
                # Show data table
                with st.expander("See Hourly Pattern Data"):
                    st.dataframe(hour_metrics)
            else:
                st.info("No intraday time data available for this timeframe selection.")

# Add combined BARCODE profile table
st.header("🔍 BARCODE Profile Table")
st.markdown("""
This table shows the complete BARCODE classification for the selected time period.
You can use it to identify specific patterns or filter data for further analysis.
""")

# Create a sample BARCODE table from filtered data
if len(filtered_df) > 0:
    # Select relevant columns for the BARCODE table - base columns first
    barcode_cols = ['time', 'decennial', 'presidential', 'quarter', 'month', 'week_of_month', 'day_of_week']
    if 'session' in filtered_df.columns:
        barcode_cols.append('session')
    
    # Add descriptive name columns if they exist, avoiding duplicates
    name_cols = {
        'presidential_name': 'presidential',
        'quarter_name': 'quarter',
        'month_name': 'month',
        'week_name': 'week_of_month',
        'day_name': 'day_of_week', 
        'session_name': 'session'
    }
    
    # Only add name columns if they exist and aren't already included
    for name_col, base_col in name_cols.items():
        if name_col in filtered_df.columns:
            barcode_cols.append(name_col)
    
    # Add price and ATR data
    data_cols = ['Open', 'High', 'Low', 'Close', 'candle_color', 'atr', 'atr_pct']
    if 'atr_points' in filtered_df.columns:
        data_cols.append('atr_points')
    
    # Combine all columns
    table_cols = barcode_cols + data_cols
    
    # Create the BARCODE table with a sample of the data
    barcode_table = filtered_df[table_cols].copy()
    
    # Format the time column for better display
    barcode_table['time'] = barcode_table['time'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Add ATR points color coding if available
    if 'atr_points' in barcode_table.columns:
        def color_atr_points(val):
            if val == 1:
                return f'background-color: {colors["blue"]}; color: white'
            elif val == 2:
                return f'background-color: {colors["gold"]}; color: black'
            elif val == 3:
                return f'background-color: {colors["red"]}; color: white'
            return ''
        
        # Apply color coding
        styled_table = barcode_table.style.map(
            color_atr_points, 
            subset=['atr_points']
        )
        
        # Apply color coding to candle colors
        def color_candles(val):
            if val == 'green':
                return f'background-color: {colors["green"]}; color: white'
            elif val == 'red':
                return f'background-color: {colors["red"]}; color: white'
            return ''
        
        styled_table = styled_table.map(
            color_candles,
            subset=['candle_color']
        )
        
        # Display the styled table
        st.dataframe(styled_table)
    else:
        # Display the regular table if no ATR points
        st.dataframe(barcode_table)
        
    # Add export option for the BARCODE table
    if st.button("Export BARCODE Table to CSV"):
        csv = barcode_table.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"gold_barcode_profile_{timeframe_label.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
else:
    st.warning("No data available to display BARCODE profile table.")

# Add ATR Points System explanation
if 'atr_points' in filtered_df.columns and atr_ratio_enabled:
    st.header("ATR Points System (1-2-3 Scale)")
    st.markdown("""
    The ATR Points System classifies volatility into three categories:
    
    1. **Low Volatility (Blue)**: ATR is below 80% of its historical baseline
    2. **Normal Volatility (Gold)**: ATR is between 80-120% of its historical baseline
    3. **High Volatility (Red)**: ATR is above 120% of its historical baseline
    
    This system helps identify unusual market conditions and potential trading opportunities.
    """)
    
    # Create ATR Points distribution chart
    atr_points_count = filtered_df['atr_points'].value_counts().sort_index()
    
    # Map point values to labels
    point_labels = {
        1: "Low (1)",
        2: "Normal (2)",
        3: "High (3)"
    }
    
    # Create colors for each point value
    point_colors = {
        1: colors['blue'],
        2: colors['gold'],
        3: colors['red']
    }
    
    # Create the chart
    fig_points = go.Figure()
    
    # Add bars for each point value
    for point in sorted(atr_points_count.index):
        fig_points.add_trace(go.Bar(
            x=[point_labels[point]],
            y=[atr_points_count[point]],
            name=point_labels[point],
            marker_color=point_colors[point],
            text=str(atr_points_count[point]),  # Convert numpy.int64 to string
            textposition='auto'
        ))
    
    # Update layout
    fig_points.update_layout(
        title="Distribution of ATR Points",
        xaxis_title="Volatility Category",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=colors['text']
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background']
    )
    
    st.plotly_chart(fig_points, use_container_width=True)
    
    # Calculate and display transitions between ATR points
    st.subheader("ATR Points Transitions")
    st.markdown("""
    This analysis shows how volatility transitions from one state to another.
    For example, how often low volatility (1) is followed by high volatility (3).
    """)
    
    # Create ATR points transition column
    filtered_df['prev_atr_points'] = filtered_df['atr_points'].shift(1)
    filtered_df['atr_transition'] = filtered_df['prev_atr_points'].astype(str) + "->" + filtered_df['atr_points'].astype(str)
    
    # Filter out rows with missing previous points
    transition_df = filtered_df.dropna(subset=['prev_atr_points'])
    
    # Count transitions
    transitions = transition_df['atr_transition'].value_counts().reset_index()
    transitions.columns = ['Transition', 'Count']
    
    # Create a descriptive transition column
    transition_map = {
        "1->1": "Low → Low",
        "1->2": "Low → Normal",
        "1->3": "Low → High",
        "2->1": "Normal → Low",
        "2->2": "Normal → Normal",
        "2->3": "Normal → High",
        "3->1": "High → Low",
        "3->2": "High → Normal",
        "3->3": "High → High"
    }
    transitions['Description'] = transitions['Transition'].map(transition_map)
    
    # Sort by the first number then the second
    transitions['From'] = transitions['Transition'].str[0].astype(int)
    transitions['To'] = transitions['Transition'].str[-1].astype(int)
    transitions = transitions.sort_values(['From', 'To'])
    
    # Create chord diagram or sankey diagram
    from_values = []
    to_values = []
    values = []
    
    # Prepare data for the diagram
    for _, row in transitions.iterrows():
        from_values.append(row['From'] - 1)  # 0-indexed
        to_values.append(row['To'] - 1)      # 0-indexed
        values.append(row['Count'])
    
    # Create chord diagram
    fig_transitions = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Low (1)", "Normal (2)", "High (3)"],
            color=[colors['blue'], colors['gold'], colors['red']]
        ),
        link=dict(
            source=from_values,
            target=to_values,
            value=values,
            color="rgba(200, 200, 200, 0.3)"
        )
    )])
    
    fig_transitions.update_layout(
        title="ATR Volatility State Transitions",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color=colors['text']
        ),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        height=400
    )
    
    st.plotly_chart(fig_transitions, use_container_width=True)
    
    # Show transition data table
    with st.expander("See ATR Transition Data"):
        transitions_display = transitions[['Description', 'Count']]
        transitions_display['Percentage'] = (transitions_display['Count'] / transitions_display['Count'].sum() * 100).round(2)
        transitions_display['Percentage'] = transitions_display['Percentage'].astype(str) + '%'
        st.dataframe(transitions_display)

# Add ATR explanation
with st.expander("💹 Understand ATR (Average True Range) Analysis"):
    # Check if ATR columns exist in the dataframe
    has_atr = 'atr' in filtered_df.columns and len(filtered_df) > 0
    has_atr_pct = 'atr_pct' in filtered_df.columns and len(filtered_df) > 0
    
    # Calculate current ATR and average ATR for comparison
    current_atr = filtered_df['atr'].iloc[-1] if has_atr else 0
    current_atr_pct = filtered_df['atr_pct'].iloc[-1] if has_atr_pct else 0
    avg_atr = filtered_df['atr'].mean() if has_atr else 0 
    avg_atr_pct = filtered_df['atr_pct'].mean() if has_atr_pct else 0
    
    # Calculate volatility comparison safely
    if avg_atr_pct > 0:
        volatility_comparison = f"{(current_atr_pct/avg_atr_pct*100):.1f}% of normal volatility"
    else:
        volatility_comparison = "N/A (insufficient data)"
    
    st.markdown(f"""
    ### ATR (Average True Range) Explained
    
    ATR measures market volatility by calculating the average range between high and low prices.
    Higher ATR = Higher volatility = Wider price swings
    
    **Current ATR Analysis for {timeframe_label} Data:**
    
    **1. Raw Values**
    - Current ATR: {current_atr:.2f} price units
    - Current ATR%: {current_atr_pct:.2f}% of price
    - Average ATR%: {avg_atr_pct:.2f}% of price
    
    **2. Potential Price Movement**
    - Conservative: Current Price ± {(current_atr_pct):.2f}% (1 ATR)
    - Moderate: Current Price ± {(current_atr_pct*2):.2f}% (2 ATR)
    - Aggressive: Current Price ± {(current_atr_pct*3):.2f}% (3 ATR)
    
    **3. Volatility Comparison**
    - Current vs. Average: {volatility_comparison if avg_atr_pct > 0 else "N/A (insufficient data)"}
    """)
