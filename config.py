# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get sensitive information from environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = os.getenv('CONTAINER_NAME')

# Define dataset file paths in Azure Storage
timeframe_files = [
    "Gold_1min_25_15.csv",
    "Gold_5min_25_10.csv",
    "Gold_10min_25_08.csv",
    "Gold_15min_25_08.csv",
    "Gold_30min_25_08.csv",
    "Gold_1h_25_08.csv",
    "Gold_4h_25_08.csv",
    "Gold_D_25_74.csv",
    "Gold_W_25_74.csv",
    "Gold_M_25_74.csv"
]
