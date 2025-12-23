import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, clean_data, feature_engineer_fraud, transform_data
from src.utils import save_stats, save_plot

def run_explainability():
    # 1. Load Data
    print("Loading Data...")
    base_dir = os.path.dirname(__file__)
    fraud_path = os.path.join(base_dir, '../data/raw/Fraud_Data.csv')
    ip_path = os.path.join(base_dir, '../data/raw/IpAddress_to_Country.csv')
    credit_path = os.path.join(base_dir, '../data/raw/creditcard.csv')

    fraud_df, ip_df, credit_df = load_data(fraud_path, ip_path, credit_path)

    # 2. Preprocess
    print("Preprocessing...")
    fraud_df = clean_data(fraud_df)
    fraud_df = feature_engineer_fraud(fraud_df, ip_df)
    
    num_cols_fraud = ['purchase_value', 'age', 'hour_of_day', 'time_since_signup', 'device_id_count', 'ip_address_count']
    cat_cols_fraud = ['source', 'browser', 'sex']
    
    fraud_transformed, _ = transform_data(fraud_df, cat_cols_fraud, num_cols_fraud)
    
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'country', 'ip_int', 'lower_bound_ip_address', 'upper_bound_ip_address', 'day_of_week']
    # Drop columns if they exist
    cols_to_drop_actual = [col for col in cols_to_drop if col in fraud_transformed.columns]
    X = fraud_transformed.drop(columns=cols_to_drop_actual + ['class'])
    y = fraud_transformed['class'].astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Model Training (Random Forest)
    print("Training Random Forest Model for Explainability...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 4. Feature Importance (Built-in)
    print("Calculating Built-in Feature Importance...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print(feature_importance.head())
    save_stats(feature_importance, 'rf_feature_importance.csv')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    save_plot(plt, 'rf_feature_importance.png')
    plt.close()

    # 5. SHAP Analysis
    print("Calculating SHAP Values...")
    explainer = shap.TreeExplainer(rf_model)
    
    # Sample for performance (SHAP can be slow on large datasets)
    sample_size = 2000
    if len(X_test) > sample_size:
        X_shap = X_test.sample(sample_size, random_state=42)
    else:
        X_shap = X_test
    
    # Ensure all data is numeric for plotting (SHAP needs this for color coding High/Low)
    X_shap = X_shap.apply(pd.to_numeric, errors='coerce')
    print("X_shap dtypes:")
    print(X_shap.dtypes)
        
    shap_values = explainer.shap_values(X_shap)
    
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print("SHAP values is a list")
        shap_values_target = shap_values[1]
    elif isinstance(shap_values, np.ndarray):
        print(f"SHAP values is an array of shape {shap_values.shape}")
        if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
            shap_values_target = shap_values[:, :, 1]
        else:
            shap_values_target = shap_values
    else:
        shap_values_target = shap_values

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    # Explicitly using dot plot and passing features
    shap.summary_plot(shap_values_target, X_shap, show=False, plot_type="dot")
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    save_plot(plt, 'shap_summary_plot.png')
    plt.close()

    # SHAP Feature Importance (Mean |SHAP|)
    print("Calculating SHAP Feature Importance...")
    shap_importance = pd.DataFrame(list(zip(X_shap.columns, np.abs(shap_values_target).mean(0))), columns=['feature', 'mean_shap_value'])
    shap_importance.sort_values(by='mean_shap_value', ascending=False, inplace=True)
    save_stats(shap_importance, 'shap_feature_importance.csv')
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='mean_shap_value', y='feature', data=shap_importance)
    plt.title('SHAP Feature Importance (Mean |SHAP|)')
    plt.tight_layout()
    save_plot(plt, 'shap_feature_importance_plot.png')
    plt.close()

    # SHAP Dependence Plots (Top 3 features)
    top_features = shap_importance['feature'].head(3).tolist()
    for feature in top_features:
        print(f"Generating Dependence Plot for {feature}...")
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values_target, X_shap, show=False)
        plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        save_plot(plt, f'shap_dependence_{feature}.png')
        plt.close()

    print("Explainability Tasks Completed.")
    
    # 6. Force Plots for specific cases (TP, FP, FN)
    print("Generating Force Plots for specific cases...")
    y_pred = rf_model.predict(X_test)
    
    # Reset index to make sure we can index into them positionally
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)
    
    tp_indices = y_test_reset[(y_test_reset == 1) & (y_pred_series == 1)].index
    fp_indices = y_test_reset[(y_test_reset == 0) & (y_pred_series == 1)].index
    fn_indices = y_test_reset[(y_test_reset == 1) & (y_pred_series == 0)].index
    
    cases = {
        'True_Positive': tp_indices,
        'False_Positive': fp_indices,
        'False_Negative': fn_indices
    }
    
    for name, indices in cases.items():
        if len(indices) > 0:
            idx = indices[0] # Pick the first available
            print(f"Generating plot for {name} at index {idx}")
            instance = X_test_reset.iloc[[idx]]
            
            # Calculate SHAP for this single instance
            shap_values_single = explainer.shap_values(instance)
            
            if isinstance(shap_values_single, list):
                shap_val = shap_values_single[1]
                base_value = explainer.expected_value[1]
            elif len(np.array(shap_values_single).shape) == 3:
                shap_val = shap_values_single[:,:,1]
                base_value = explainer.expected_value[1]
            else:
                shap_val = shap_values_single
                base_value = explainer.expected_value
                
            # Create Force Plot (Static PNG)
            print(f"Generating static plot for {name}...")
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                base_value,
                shap_val[0],
                instance.iloc[0],
                matplotlib=True,
                show=False
            )
            
            # Save as PNG
            save_plot(plt, f'force_plot_{name}.png')
            plt.close()
        else:
            print(f"No {name} found in test set.")

if __name__ == "__main__":
    run_explainability()
