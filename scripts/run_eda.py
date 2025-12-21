import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_preprocessing import load_data, clean_data, feature_engineer_fraud
from src.utils import save_stats, save_plot

# Ensure directories exist
os.makedirs('report/images', exist_ok=True)
os.makedirs('report/stats', exist_ok=True)

def run_eda_fraud():
    print("Running EDA for Fraud Data...")
    fraud_path = 'data/raw/Fraud_Data.csv'
    ip_path = 'data/raw/IpAddress_to_Country.csv'
    credit_path = 'data/raw/creditcard.csv'
    
    fraud_df, ip_df, credit_df = load_data(fraud_path, ip_path, credit_path)
    
    # 1. Basic Stats
    stats = fraud_df.describe()
    save_stats(stats, 'fraud_data_raw_stats.csv')
    
    # 2. Class Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=fraud_df, palette='viridis')
    plt.title('Fraud Class Distribution')
    save_plot(plt, 'fraud_class_distribution.png')
    plt.close()
    
    class_counts = fraud_df['class'].value_counts(normalize=True).to_frame()
    save_stats(class_counts, 'fraud_class_counts.csv')
    
    # 3. Cleaning & Feature Engineering
    fraud_df = clean_data(fraud_df)
    fraud_df = feature_engineer_fraud(fraud_df, ip_df)
    
    # 4. Univariate Analysis - Age
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_df['age'], bins=30, kde=True)
    plt.title('Distribution of User Age')
    save_plot(plt, 'fraud_age_distribution.png')
    plt.close()
    
    # 5. Geolocation Analysis
    country_fraud = fraud_df.groupby('country')['class'].mean().sort_values(ascending=False).head(10).to_frame()
    save_stats(country_fraud, 'top_10_fraud_countries_rate.csv')
    
    plt.figure(figsize=(12, 8))
    country_fraud['class'].plot(kind='bar')
    plt.title('Top 10 Countries by Fraud Rate')
    plt.ylabel('Fraud Rate')
    save_plot(plt, 'top_fraud_countries.png')
    plt.close()
    
    # 6. Bivariate Analysis - Purchase Value vs Class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='purchase_value', data=fraud_df)
    plt.title('Purchase Value by Class')
    save_plot(plt, 'fraud_purchase_value_box.png')
    plt.close()

    print("EDA for Fraud Data completed.")

def run_eda_credit():
    print("Running EDA for Credit Card Data...")
    credit_df = pd.read_csv('data/raw/creditcard.csv')
    
    # stats
    stats = credit_df.describe()
    save_stats(stats, 'creditcard_stats.csv')
    
    # class dist
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=credit_df)
    plt.title('Credit Card Fraud Distribution')
    save_plot(plt, 'creditcard_class_distribution.png')
    plt.close()
    
    # Correlation matrix of anonymized features
    plt.figure(figsize=(12, 10))
    corr = credit_df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Credit Card Feature Correlation')
    save_plot(plt, 'creditcard_correlation.png')
    plt.close()
    
    print("EDA for Credit Card Data completed.")

if __name__ == "__main__":
    run_eda_fraud()
    run_eda_credit()
