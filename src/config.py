
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
IMAGES_DIR = REPORT_DIR / "images"
STATS_DIR = REPORT_DIR / "stats"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

# Files
FRAUD_DATA_PATH = RAW_DATA_DIR / "Fraud_Data.csv"
IP_DATA_PATH = RAW_DATA_DIR / "IpAddress_to_Country.csv"
CREDIT_CARD_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

# Model Config
RANDOM_SEED = 42

# Features
NUMERICAL_FEATURES_FRAUD = [
    'purchase_value', 
    'age', 
    'hour_of_day', 
    'time_since_signup', 
    'device_id_count', 
    'ip_address_count'
]

CATEGORICAL_FEATURES_FRAUD = [
    'source', 
    'browser', 
    'sex'
]

COLS_TO_DROP_FRAUD = [
    'user_id', 
    'signup_time', 
    'purchase_time', 
    'device_id', 
    'ip_address', 
    'country', 
    'ip_int', 
    'lower_bound_ip_address', 
    'upper_bound_ip_address', 
    'day_of_week'
]
