import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import joblib
import os
from tabulate import tabulate

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# 1. Load Data
raw_dataset = load_breast_cancer()
breast_cancer_df = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
diagnostic_labels = pd.Series(raw_dataset.target, name='target')

print(f"Dataset Shape: {breast_cancer_df.shape}")
print(f"Target Distribution:\n{diagnostic_labels.value_counts()}")

# 2. Split Data
# Using a unique random state for reproducibility
UNIQUE_SEED = 123 
train_features, test_features, train_labels, test_labels = train_test_split(
    breast_cancer_df, diagnostic_labels, test_size=0.2, random_state=UNIQUE_SEED, stratify=diagnostic_labels
)

# 3. Create a sample test file for upload demo
sample_upload_data = test_features.copy()
sample_upload_data['target'] = test_labels
sample_upload_data.to_csv("sample_test_data.csv", index=False)
print("Saved sample_test_data.csv for upload demo.")

# 4. Scale Data
feature_scaler = StandardScaler()
train_features_scaled = feature_scaler.fit_transform(train_features)
test_features_scaled = feature_scaler.transform(test_features)

# Save scaler
joblib.dump(feature_scaler, "models/scaler.joblib")

# 5. Define Models
classification_pipeline = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=UNIQUE_SEED),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=UNIQUE_SEED),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=UNIQUE_SEED)
}

# 6. Train & Evaluate
performance_metrics = []

print("\nTraining and Evaluating Models...")
for model_name, classifier in classification_pipeline.items():
    # Train
    classifier.fit(train_features_scaled, train_labels)
    
    # Predict
    predicted_labels = classifier.predict(test_features_scaled)
    predicted_probs = classifier.predict_proba(test_features_scaled)[:, 1] if hasattr(classifier, "predict_proba") else None
    
    # Metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    auc_val = roc_auc_score(test_labels, predicted_probs) if predicted_probs is not None else roc_auc_score(test_labels, predicted_labels)
    precision_val = precision_score(test_labels, predicted_labels)
    recall_val = recall_score(test_labels, predicted_labels)
    f1_val = f1_score(test_labels, predicted_labels)
    mcc_val = matthews_corrcoef(test_labels, predicted_labels)
    
    performance_metrics.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc_val,
        "Precision": precision_val,
        "Recall": recall_val,
        "F1": f1_val,
        "MCC": mcc_val
    })
    
    # Save Model
    saved_filename = model_name.replace(" ", "_").lower() + ".joblib"
    joblib.dump(classifier, f"models/{saved_filename}")
    print(f"Saved {model_name} to models/{saved_filename}")

# 7. Create Comparison Table
metrics_df = pd.DataFrame(performance_metrics)
print("\nModel Comparison Table:")
try:
    print(tabulate(metrics_df, headers='keys', tablefmt='pipe', showindex=False))
except:
    print(metrics_df.to_string())

# Save results for README
metrics_df.to_csv("model_comparison.csv", index=False)
print("\nSaved model_comparison.csv for README reference.")
