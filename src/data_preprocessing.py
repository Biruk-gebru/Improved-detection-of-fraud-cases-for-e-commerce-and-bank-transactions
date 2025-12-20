import pandas as pd
import numpy as np
from src.utils import map_ip_to_country

def load_data(fraud_path, ip_path, credit_path):
    fraud_df = pd.read_csv(fraud_path)
    ip_df = pd.read_csv(ip_path)
    credit_df = pd.read_csv(credit_path)
    return fraud_df, ip_df, credit_df

def clean_data(df):
    """Basic cleaning: drop duplicates and handle missing values."""
    df = df.drop_duplicates()
    # Placeholder for more specific cleaning
    return df

def feature_engineer_fraud(df, ip_df):
    """
    Apply feature engineering to Fraud_Data.csv as per Task 1 specifications.
    """
    # 1. Geolocation Mapping
    df = map_ip_to_country(df, ip_df)
    
    # 2. Time-based features
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # 3. Time since signup
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # 4. Transaction frequency/velocity (simplified for now)
    # Number of transactions per device
    df['device_trans_count'] = df.groupby('device_id')['user_id'].transform('count')
    # Number of transactions per IP
    df['ip_trans_count'] = df.groupby('ip_address')['user_id'].transform('count')
    
    return df

if __name__ == "__main__":
    # Test loading and basic processing
    fraud_path = 'data/raw/Fraud_Data.csv'
    ip_path = 'data/raw/IpAddress_to_Country.csv'
    credit_path = 'data/raw/creditcard.csv'
    
    fraud_df, ip_df, credit_df = load_data(fraud_path, ip_path, credit_path)
    print(f"Loaded Fraud Data: {fraud_df.shape}")
    
    fraud_df = clean_data(fraud_df)
    fraud_df = feature_engineer_fraud(fraud_df, ip_df)
    print(f"Processed Fraud Data: {fraud_df.shape}")
    print(fraud_df[['user_id', 'country', 'hour_of_day', 'time_since_signup']].head())
