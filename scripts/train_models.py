import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, clean_data, feature_engineer_fraud, transform_data
from src.utils import save_stats, save_plot
from src.config import (
    FRAUD_DATA_PATH, IP_DATA_PATH, CREDIT_CARD_DATA_PATH,
    NUMERICAL_FEATURES_FRAUD, CATEGORICAL_FEATURES_FRAUD, COLS_TO_DROP_FRAUD,
    MODELS_DIR
)

def train_and_evaluate(X, y, dataset_name):
    """
    Generic training pipeline for a given dataset.
    """
    print(f"\n--- Training Pipeline for {dataset_name} ---")
    
    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)
    y = y.astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    results.append({
        'Dataset': dataset_name,
        'Model': 'Logistic Regression',
        'Test ROC AUC': roc_auc_score(y_test, y_prob_lr),
        'CV Mean AUC': cv_scores_lr.mean()
    })
    
    # 2. Random Forest (with Tuning for Fraud, simplified for Credit to save time/compute)
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # For speed in this demonstration, we'll use a smaller search space or defaults for Credit
    if dataset_name == "Fraud_Data":
        param_dist = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
        rf_search = RandomizedSearchCV(rf, param_dist, n_iter=5, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42)
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        cv_score_rf = rf_search.best_score_
    else:
        # Credit data is large, running full CV tuning might be slow. Using a strong default.
        best_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        cv_score_rf = cross_val_score(best_rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        best_rf.fit(X_train, y_train)

    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    results.append({
        'Dataset': dataset_name,
        'Model': 'Random Forest',
        'Test ROC AUC': roc_auc_score(y_test, y_prob_rf),
        'CV Mean AUC': cv_score_rf
    })
    
    # Save model
    joblib.dump(best_rf, os.path.join(MODELS_DIR, f'{dataset_name.lower()}_rf_model.joblib'))
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    results.append({
        'Dataset': dataset_name,
        'Model': 'XGBoost',
        'Test ROC AUC': roc_auc_score(y_test, y_prob_xgb),
        'CV Mean AUC': cv_scores_xgb.mean()
    })
    
    # Plot ROC
    plt.figure(figsize=(10, 6))
    for name, prob in zip(['LR', 'RF', 'XGB'], [y_prob_lr, y_prob_rf, y_prob_xgb]):
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    save_plot(plt, f'{dataset_name.lower()}_roc_curve.png')
    plt.close()
    
    return results

def run_training():
    fraud_df, ip_df, credit_df = load_data(FRAUD_DATA_PATH, IP_DATA_PATH, CREDIT_CARD_DATA_PATH)
    
    all_results = []
    
    # --- Process Fraud Data ---
    print("Processing Fraud Data...")
    fraud_df = clean_data(fraud_df)
    fraud_df = feature_engineer_fraud(fraud_df, ip_df)
    fraud_transformed, _ = transform_data(fraud_df, CATEGORICAL_FEATURES_FRAUD, NUMERICAL_FEATURES_FRAUD)
    
    X_fraud = fraud_transformed.drop(columns=[col for col in COLS_TO_DROP_FRAUD if col in fraud_transformed.columns] + ['class'])
    y_fraud = fraud_transformed['class']
    
    all_results.extend(train_and_evaluate(X_fraud, y_fraud, "Fraud_Data"))
    
    # --- Process Credit Card Data ---
    print("\nProcessing Credit Card Data...")
    # Credit Card data is already PCA transformed and clean, mostly needs scaling of 'Amount'
    # and dropping Time (or scaling it). 'Class' is target.
    # No categorical features to encode.
    
    # Basic cleaning
    credit_df = clean_data(credit_df)
    
    # Scaling Amount and Time
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    credit_df['Amount'] = scaler.fit_transform(credit_df[['Amount']])
    credit_df['Time'] = scaler.fit_transform(credit_df[['Time']])
    
    X_credit = credit_df.drop(columns=['Class'])
    y_credit = credit_df['Class']
    
    all_results.extend(train_and_evaluate(X_credit, y_credit, "CreditCard_Data"))
    
    # Save Combined Results
    results_df = pd.DataFrame(all_results)
    save_stats(results_df, 'combined_model_comparison.csv')
    print("\nCombined Results:")
    print(results_df)

if __name__ == "__main__":
    run_training()
