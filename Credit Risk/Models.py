# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import pickle

# Load the dataset
try:
    df = pd.read_csv('credit_risk_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'credit_risk_dataset.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# --- 1. Exploratory Data Analysis (EDA) ---
# Understanding the data is crucial for building a fair and effective model.
# This step helps in identifying potential biases as per ISO/IEC 42001 guidelines.

print("\n--- Exploratory Data Analysis ---")
print("\nDataset Info:")
df.info()

print("\nTarget Variable Distribution (loan_status):")
# This is important to check for class imbalance.
print(df['loan_status'].value_counts(normalize=True))

print("\nDescriptive Statistics for Numerical Features:")
print(df.describe())

# --- 2. Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Handle Missing Values
print("\nMissing values before handling:")
print(df.isnull().sum())

# For simplicity, we'll fill numerical NaNs with the median and categorical NaNs with the mode.
for col in df.select_dtypes(include=['number']).columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled NaNs in '{col}' with median value: {median_val}")

print("\nMissing values after handling:")
print(df.isnull().sum())

# Encode Categorical Features using One-Hot Encoding
print("\nEncoding categorical features...")
# `get_dummies` converts categorical variables into dummy/indicator variables.
# `drop_first=True` is used to avoid multicollinearity (dummy variable trap).
df_encoded = pd.get_dummies(df, drop_first=True)
print("Data shape after encoding:", df_encoded.shape)
print("Columns after encoding:", df_encoded.columns.tolist())

# --- 3. Data Splitting ---
print("\n--- Splitting Data into Training and Testing Sets ---")

# Separate features (X) and target (y)
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

# Split the data into 80% training and 20% testing
# `stratify=y` ensures that the proportion of the target variable is the same in both train and test sets.
# This is crucial for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")

# --- 4. Model Training and Evaluation ---

def evaluate_model(model_name, y_true, y_pred, y_prob):
    """
    Calculates and prints evaluation metrics for a model.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("--------------------------------" + "-" * len(model_name))

# 3. XGBoost Classifier
print("\nTraining XGBoost Classifier model...")
# For XGBoost, `scale_pos_weight` is a common way to handle class imbalance.
# It's the ratio of the number of negative class to the positive class.
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count

xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
evaluate_model("XGBoost Classifier", y_test, y_pred_xgb, y_prob_xgb)

# 4. LightGBM Classifier
print("\nTraining LightGBM Classifier model...")
lgb_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
evaluate_model("LightGBM Classifier", y_test, y_pred_lgb, y_prob_lgb)

# --- 5. Save Models, Encoder and Training Data ---
print("\n--- Saving models and objects to disk ---")

# Define a dictionary of models to save for cleaner code
models_to_save = {
    'xgboost_model.pkl': xgb_model,
    'lightgbm_model.pkl': lgb_model
}

# Loop through the dictionary and save each model using pickle
for filename, model in models_to_save.items():
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Saved model to {filename}")

# Save the column list from the one-hot encoding process.
# This acts as the 'encoder' to ensure new data in the app has the same feature structure.
with open('encoder.pkl', 'wb') as file:
    pickle.dump(X.columns, file)
print("Saved encoder columns to encoder.pkl")

# Save the training data, which is needed for model explainability (e.g., with SHAP)
with open('X_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)
print("Saved training data to X_train.pkl")