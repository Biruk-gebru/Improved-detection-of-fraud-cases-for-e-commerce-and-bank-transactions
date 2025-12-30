import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sys
import os
import joblib
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, clean_data, feature_engineer_fraud, apply_transformation
from src.utils import save_stats, save_plot
from src.config import (
    FRAUD_DATA_PATH, IP_DATA_PATH, CREDIT_CARD_DATA_PATH,
    NUMERICAL_FEATURES_FRAUD, CATEGORICAL_FEATURES_FRAUD, COLS_TO_DROP_FRAUD,
    MODELS_DIR
)

def run_explainability():
    # 1. Check for saved models
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print("Model or preprocessor not found. Please run scripts/train_models.py first.")
        # Optional: could trigger training here, but cleaner to separate concerns
        return

    # 2. Load Data
    print("Loading Data...")
    fraud_df, ip_df, credit_df = load_data(FRAUD_DATA_PATH, IP_DATA_PATH, CREDIT_CARD_DATA_PATH)

    # 3. Preprocess
    print("Preprocessing...")
    fraud_df = clean_data(fraud_df)
    fraud_df = feature_engineer_fraud(fraud_df, ip_df)
    
    # 4. Load Artifacts
    print("Loading saved model and preprocessor...")
    preprocessor = joblib.load(preprocessor_path)
    rf_model = joblib.load(model_path)
    
    # 5. Transform Data
    print("Transforming data using saved preprocessor...")
    fraud_transformed = apply_transformation(fraud_df, preprocessor, CATEGORICAL_FEATURES_FRAUD, NUMERICAL_FEATURES_FRAUD)
    
    # Drop columns
    # Note: apply_transformation might have already handled columns present in preprocessor
    # But we need to ensure X doesn't have extra columns that weren't in training if strict
    # However, RF usually only cares about feature order/names if dataframe.
    
    cols_to_drop_actual = [col for col in COLS_TO_DROP_FRAUD if col in fraud_transformed.columns]
    X = fraud_transformed.drop(columns=cols_to_drop_actual + ['class'])
    y = fraud_transformed['class'].astype(int)
    
    # Ensure all data is numeric for SHAP
    X = X.apply(pd.to_numeric, errors='coerce')

    # 6. Feature Importance (Built-in)
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

    # 7. SHAP Analysis
    print("Calculating SHAP Values...")
    try:
        explainer = shap.TreeExplainer(rf_model)
    except Exception as e:
        print(f"Error creating TreeExplainer: {e}. Trying generic Explainer.")
        explainer = shap.Explainer(rf_model, X)
    
    # Sample for performance
    sample_size = 2000
    if len(X) > sample_size:
        X_shap = X.sample(sample_size, random_state=42)
    else:
        X_shap = X
        
    shap_values = explainer.shap_values(X_shap)
    
    # Handle SHAP return type (mostly for RF binary class)
    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]
        base_value = explainer.expected_value[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_target = shap_values[:, :, 1]
        base_value = explainer.expected_value[1]
    else:
        shap_values_target = shap_values
        if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)):
             base_value = explainer.expected_value[1]
        else:
             base_value = explainer.expected_value

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_target, X_shap, show=False, plot_type="dot")
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    save_plot(plt, 'shap_summary_plot.png')
    plt.close()

    # SHAP Feature Importance (Mean |SHAP|)
    print("Calculating SHAP Feature Importance...")
    # Fix for when shap_values_target is array
    if isinstance(shap_values_target, np.ndarray):
         mean_shap = np.abs(shap_values_target).mean(0)
    else:
         mean_shap = np.abs(shap_values_target.values).mean(0)
         
    shap_importance = pd.DataFrame(list(zip(X_shap.columns, mean_shap)), columns=['feature', 'mean_shap_value'])
    shap_importance.sort_values(by='mean_shap_value', ascending=False, inplace=True)
    save_stats(shap_importance, 'shap_feature_importance.csv')
    
    # SHAP Dependence Plots (Top 3)
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
    
    # 8. Force Plots for specific cases
    print("Generating Force Plots for specific cases...")
    # Use the model to predict on X_shap to find TP/FP/FN within the sample
    y_pred_sample = rf_model.predict(X_shap)
    # Get ground truth for X_shap indices
    y_true_sample = y.loc[X_shap.index]
    
    X_shap_reset = X_shap.reset_index(drop=True)
    y_true_reset = y_true_sample.reset_index(drop=True)
    y_pred_reset = pd.Series(y_pred_sample)
    
    tp_indices = y_true_reset[(y_true_reset == 1) & (y_pred_reset == 1)].index
    fp_indices = y_true_reset[(y_true_reset == 0) & (y_pred_reset == 1)].index
    fn_indices = y_true_reset[(y_true_reset == 1) & (y_pred_reset == 0)].index
    
    cases = {
        'True_Positive': tp_indices,
        'False_Positive': fp_indices,
        'False_Negative': fn_indices
    }
    
    for name, indices in cases.items():
        if len(indices) > 0:
            idx = indices[0]
            print(f"Generating plot for {name} at sample index {idx}")
            instance = X_shap_reset.iloc[[idx]]
            shap_val_single = shap_values_target[idx]
            
            plt.figure(figsize=(20, 3))
            shap.force_plot(
                base_value,
                shap_val_single,
                instance.iloc[0],
                matplotlib=True,
                show=False
            )
            save_plot(plt, f'force_plot_{name}.png')
            plt.close()
        else:
            print(f"No {name} found in sample.")

if __name__ == "__main__":
    run_explainability()
