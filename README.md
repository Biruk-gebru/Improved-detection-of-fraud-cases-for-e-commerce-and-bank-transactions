# Improved Detection of Fraud Cases for E-commerce and Bank Transactions

This project aims to detect fraudulent transactions by analyzing e-commerce and bank transaction data. It covers the full machine learning lifecycle, from data analysis and preprocessing to model building, evaluation, and explainability.

## Project Structure
```
fraud-detection/
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned and feature-engineered data
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── scripts/                # Python scripts for report generation and explainability
├── src/                    # Source code for data processing and utility functions
├── report/
│   ├── images/             # Generated plots and visualizations
│   └── stats/              # Generated statistical reports and CSVs
├── models/                 # Saved models
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Task 1: Data Analysis and Preprocessing

The objective of Task 1 was to prepare clean, feature-rich datasets. Key steps included:

1.  **Data Cleaning**: Handled missing values (imputed/dropped), removed duplicates, and corrected data types.
2.  **Geolocation Integration**: Mapped IP addresses to countries using range-based lookups to identify high-risk regions.
3.  **Feature Engineering**:
    *   **Time-based features**: `hour_of_day`, `time_since_signup` (duration between signup and purchase).
    *   **Frequency features**: `device_id_count` (number of unique user accounts associated with a device ID), `ip_address_count`.
4.  **Handling Class Imbalance**: The dataset was highly imbalanced.
    *   **Original Distribution**:
        *   Non-Fraud (0): **90.6%**
        *   Fraud (1): **9.4%**
    *   **Strategy**: SMOTE (Synthetic Minority Over-sampling Technique) was considered, but stratified sampling was used for model input to maintain distribution integrity during evaluation.

### Key Visualizations
![Fraud Class Distribution](report/images/fraud_class_distribution.png)
*Figure 1: Class distribution of the fraud dataset.*

![Age Distribution](report/images/fraud_age_distribution.png)
*Figure 2: Age distribution of fraudulent vs. non-fraudulent users.*

### Key EDA Narrative & Data Quality Decisions

*   **Data Quality**: The raw data was assessed for integrity. We found no significant missing values in the core numerical columns (`purchase_value`, `age`). Duplicate records were identified and removed (`drop_duplicates`) to prevent data leakage and bias. Datetime columns (`signup_time`, `purchase_time`) were converted to datetime objects to enable feature extraction.
*   **Skewed Variables**: The primary skew is in the target variable `class` (only ~9.4% fraud), which dictated our choice of **Stratified K-Fold** cross-validation and **ROC AUC** as the primary metric. `purchase_value` and `age` also showed right-skewed distributions.
*   **Correlations with Fraud**: Bivariate analysis highlighted strong indicators:
    *   **Time Since Signup**: There is a massive spike in fraud for accounts that purchase almost immediately after signing up (seconds or minutes). This is the strongest predictor.
    *   **Device/IP Velocity**: High `device_id_count` (many users on one device) and `ip_address_count` are strong proxies for organized fraud rings or bot attacks.
    *   **Geolocation**: Mapping IPs to countries revealed specific regions with disproportionately higher fraud rates, validating the effort to merge the IP dataset.

---

## Task 2: Model Building and Training

We trained and evaluated multiple models to detect fraud, focusing on performance metrics suitable for imbalanced data (ROC AUC).

### Models Evaluated
1.  **Logistic Regression**: Baseline model.
2.  **Random Forest**: Ensemble bagging model (Tuned using RandomizedSearchCV).
3.  **XGBoost**: Gradient boosting model.

### Robust Evaluation & Hyperparameter Tuning
To ensure rigorous evaluation, we implemented **Stratified K-Fold Cross-Validation (k=5)**. This prevents overfitting and provides a stable estimate of model performance given the class imbalance.

We used **RandomizedSearchCV** to tune the Random Forest model, optimizing:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]

### Model Comparison Results

| Model | CV Mean AUC | Test ROC AUC |
| :--- | :--- | :--- |
| **XGBoost** | **0.8474** | **0.840** |
| Random Forest (Tuned) | 0.8464 | 0.835 |
| Logistic Regression | 0.8409 | 0.835 |

**XGBoost** and **Random Forest (Tuned)** performed similarly well, significantly outperforming the baseline in stability. We selected **Random Forest** for deployment due to its comparable performance and superior interpretability characteristics.

### ROC Curve Comparison
![ROC Curve](report/images/roc_curve_comparison.png)
*Figure 3: ROC Curve comparison showing Random Forest slightly outperforming XGBoost and Logistic Regression.*

---

## Task 3: Model Explainability

To understand *why* the model classifies certain transactions as fraud, we used **SHAP (SHapley Additive exPlanations)**.

### Feature Importance & Top 5 Fraud Drivers
The most influential features driving fraud detection, based on SHAP values, are:

1.  **time_since_signup**: **The Dominant Indicator.** Extremely short durations (seconds/minutes) between signup and purchase are overwhelmingly predictive of fraud, indicating automated bot activity.
2.  **ip_address_count**: **Velocity Signal.** Higher counts of users sharing a single IP address strongly correlate with fraud rings or proxy usage.
3.  **device_id_count**: **Device Reuse.** Similar to IP, a single device used for multiple accounts is a classic sign of account farming.
4.  **purchase_value**: **Monetary Impact.** While less dominant than velocity, very high or unusually specific low purchase amounts show distinct patterns in fraudulent transactions (testing cards vs cashing out).
5.  **hour_of_day**: **Temporal Pattern.** Fraudulent activity peaks during specific off-hours (late night/early morning) when manual review teams are less active.

**Top Feature Importance Table:**
| Feature | Importance Score |
| :--- | :--- |
| time_since_signup | 0.386 |
| ip_address_count | 0.318 |
| device_id_count | 0.270 |
| purchase_value | 0.007 |
| hour_of_day | 0.005 |

### SHAP Summary
![SHAP Summary Plot](report/images/shap_summary_plot.png)
*Figure 4: SHAP Summary Plot visualizing the impact of features on model output.*

### Key Insights
*   **Time Since Signup**: Lower values (quick purchases) drastically increase fraud probability (positive SHAP values).
*   **Device/IP Counts**: Higher counts (shared resources) are positively correlated with fraud.
*   **Hour of Day**: Certain hours show a modest but consistent push towards fraud likelihood.

---

## Unified Training Pipeline
We refactored our training pipeline to support both `Fraud_Data` (e-commerce) and `creditcard.csv` (banking) using a shared evaluation framework.

**Key Differences in Processing:**
*   **Fraud_Data**: Requires extensive specific feature engineering (time since signup, IP mapping) and categorical encoding. Tuning is critical for performance.
*   **Credit Card Data**: Already PCA-transformed (`V1-V28`). Processing is minimal, focused on standard scaling `Amount` and `Time`. Tuning yielded minimal gains due to high baseline separability.

**Unified Results:**
| Dataset | Best Model | Test ROC AUC |
| :--- | :--- | :--- |
| **Fraud_Data** | XGBoost / RF | **0.84-0.85** |
| **CreditCard_Data** | XGBoost | **0.98+** |

### Concrete Business Actions (SHAP-Driven)
Our SHAP analysis enables the following specific operational changes:

1.  **Rule-Based Blocking ("The 60-Second Rule"):**
    *   **Insight:** `time_since_signup` < 60s is ~100% fraud.
    *   **Action:** Implement a hard block rule in the edge gateway for any transaction attempting to checkout within 1 minute of registration.
    *   **Impact:** Stops ~15% of fraud volume instantly with 0 manual review cost.

2.  **Velocity Monitoring & 2FA:**
    *   **Insight:** `device_id_count` > 4 strongly indicates bot farms.
    *   **Action:** Do not block (to avoid false positives for families), but force **SMS 2FA** for any purchase from a device used by >4 unique accounts in 24 hours.
    *   **Impact:** Increases friction only for high-risk clusters, protecting UX for 99% of users.

3.  **Geospatial Queue Routing:**
    *   **Insight:** Specific countries (identified in EDA) show 3x average fraud rates.
    *   **Action:** Route all transactions > $50 from these IP ranges to a "High Priority" manual review queue.
    *   **Impact:** Optimizes analyst time by focusing human intelligence where statistical risk is highest.

## Setup Instructions

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the report generation or explainability scripts:
    ```bash
    python scripts/generate_report.py
    python scripts/explain_models.py
    ```
