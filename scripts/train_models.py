import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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

def run_training():
    print("Loading data...")
    fraud_df, ip_df, credit_df = load_data(FRAUD_DATA_PATH, IP_DATA_PATH, CREDIT_CARD_DATA_PATH)
    
    print("Preprocessing Fraud Data...")
    fraud_df = clean_data(fraud_df)
    fraud_df = feature_engineer_fraud(fraud_df, ip_df)
    
    # Transform data and get the preprocessor object
    fraud_transformed, preprocessor = transform_data(fraud_df, CATEGORICAL_FEATURES_FRAUD, NUMERICAL_FEATURES_FRAUD)
    
    # Save the preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    X = fraud_transformed.drop(columns=[col for col in COLS_TO_DROP_FRAUD if col in fraud_transformed.columns] + ['class'])
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').astype(float)
    y = fraud_transformed['class'].astype(int)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    results = []
    
    # Define Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression (Baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    
    # Cross-validation
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"LR CV ROC-AUC: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")
    
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    results.append({
        'Model': 'Logistic Regression', 
        'Test ROC AUC': roc_auc_score(y_test, y_prob_lr),
        'CV Mean AUC': cv_scores_lr.mean()
    })
    
    # 2. Random Forest (Tuning)
    print("Tuning Random Forest...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    from sklearn.model_selection import RandomizedSearchCV
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, 
                                   n_iter=10, cv=cv, verbose=1, random_state=42, scoring='roc_auc', n_jobs=-1)
    rf_random.fit(X_train, y_train)
    
    best_rf = rf_random.best_estimator_
    print(f"Best RF Params: {rf_random.best_params_}")
    print(f"Best RF CV Score: {rf_random.best_score_:.4f}")
    
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    results.append({
        'Model': 'Random Forest (Tuned)', 
        'Test ROC AUC': roc_auc_score(y_test, y_prob_rf),
        'CV Mean AUC': rf_random.best_score_
    })
    
    # Save the best model
    rf_model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    joblib.dump(best_rf, rf_model_path)
    print(f"Tuned Random Forest model saved to {rf_model_path}")
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Cross-validation for XGB
    cv_scores_xgb = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"XGB CV ROC-AUC: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")
    
    xgb.fit(X_train, y_train)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    results.append({
        'Model': 'XGBoost', 
        'Test ROC AUC': roc_auc_score(y_test, y_prob_xgb),
        'CV Mean AUC': cv_scores_xgb.mean()
    })
    
    # Save statistics
    results_df = pd.DataFrame(results)
    save_stats(results_df, 'model_comparison_results.csv')
    print("Results saved to report/stats/model_comparison_results.csv")
    print(results_df)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    for name, prob in zip(['LR', 'RF (Tuned)', 'XGB'], [y_prob_lr, y_prob_rf, y_prob_xgb]):
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Tuned)')
    plt.legend()
    save_plot(plt, 'roc_curve_comparison.png')
    print("Plot saved to report/images/roc_curve_comparison.png")

if __name__ == "__main__":
    run_training()
