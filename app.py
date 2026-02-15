import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Fallback Training Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Page Config
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        font-family: 'Inter', sans-serif;
        color: #333333;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Headings force color */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 700;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        color: #000000 !important;
    }

    div[data-testid="metric-container"] label {
        color: #000000 !important; /* Force Pitch Black Label */
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Force Pitch Black Value */
    }

    div[data-testid="metric-container"] p {
        color: #000000 !important; /* Force Pitch Black Paragraphs */
    }
    
    /* Specific text elements */
    .stMarkdown p {
        color: #444444 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    
    /* Custom Container */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🧬 Breast Cancer AI Diagnostic Tool</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="custom-card">
    <p style="font-size:18px; color:#555;">
    This advanced diagnostic tool utilizes six state-of-the-art Machine Learning algorithms to classify breast mass samples as 
    <b>Malignant</b> or <b>Benign</b>. Upload your diagnostic data to receive instant, high-precision predictions.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Asklepios_stick.svg/1200px-Asklepios_stick.svg.png", width=50) # Placeholder medical icon
st.sidebar.header("Configuration")
st.sidebar.markdown("---")
# Download Button for Sample Data
if os.path.exists("sample_test_data.csv"):
    with open("sample_test_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Example CSV",
            data=file,
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download this sample file to test the app."
        )

# 1. Model Selection
st.sidebar.subheader("🤖 Select Model")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Choose Classifier", model_options)

# 2. File Upload
st.sidebar.subheader("📂 Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"], help="Ensure your CSV matches the Breast Cancer Wisconsin dataset format.")

# Load Scaler and Model
@st.cache_resource
def load_artifacts(model_name):
    # Use absolute paths to ensure robustness
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    # helper to find file in either models/ or root/
    def find_file(filename):
        # Check models/ folder
        path_1 = os.path.join(models_dir, filename)
        if os.path.exists(path_1):
            return path_1
            
        # Check root folder (fallback)
        path_2 = os.path.join(base_dir, filename)
        if os.path.exists(path_2):
            return path_2
            
        return None

    # Load Scaler
    scaler_path = find_file("scaler.joblib")
    scaler = None
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load scaler (Version Mismatch): {e}. Retraining...")
            
    # Load Model
    filename = model_name.replace(" ", "_").lower() + ".joblib"
    model_path = find_file(filename)
    model = None
    
    if model_path:
        try:
            model = joblib.load(model_path)
            
            # VALIDATION STEP: Try to predict on a dummy input to ensure compatibility
            # This catches 'monotonic_cst' errors that happen at inference time, not load time
            if scaler is not None:
                dummy_input = np.zeros((1, 30)) # Breast cancer dataset has 30 features
                model.predict(dummy_input)
                
        except Exception as e:
            st.warning(f"⚠️ Model compatibility issue detected: {e}. Retraining on the fly...")
            model = None # Force retraining

    # Fallback: Retrain if artifacts are missing or failed to load/validate
    if scaler is None or model is None:
        scaler, model = retrain_model_on_fly(model_name)
        
    return scaler, model

@st.cache_resource
def retrain_model_on_fly(model_name):
    """
    Retrains the specific model and scaler in-memory if loading fails.
    This handles version mismatches between local env and Streamlit Cloud.
    """
    # 1. Load Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # 2. Split (Same seed as original training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. Define Model
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=123),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=123),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=123)
    }
    
    if model_name not in models:
        st.error(f"Unknown model architecture: {model_name}")
        return None, None
        
    clf = models[model_name]
    clf.fit(X_train_scaled, y_train)
    
    return scaler, clf

# Main Logic
if uploaded_file is not None:
    try:
        # Load Data
        input_df = pd.read_csv(uploaded_file)
        
        st.markdown(f"### 🔍 Analysis using {selected_model_name}")
        
        # Display Data Preview in Expander
        with st.expander("Preview Uploaded Data", expanded=False):
            st.dataframe(input_df.head(), use_container_width=True)
        
        # Check for Target Column (Ground Truth)
        if 'target' in input_df.columns:
            y_true = input_df['target']
            X = input_df.drop('target', axis=1)
            has_ground_truth = True
        else:
            X = input_df
            has_ground_truth = False
            
        # Load Artifacts
        scaler, model = load_artifacts(selected_model_name)
        
        # STOP if artifacts failed to load
        if scaler is None or model is None:
            st.warning("⚠️ Please check your repository file structure. Models should be in a 'models/' folder or at the root.")
            st.stop()
        
        # Preprocess
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)
        prediction_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Create Results DataFrame
        results_df = X.copy()
        results_df['Prediction'] = prediction
        results_df['Diagnosis'] = results_df['Prediction'].map({0: 'Malignant', 1: 'Benign'})
        if prediction_proba is not None:
            results_df['Probability'] = prediction_proba

        # Display Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Prediction Results")
            st.dataframe(results_df[['Diagnosis'] + (['Probability'] if prediction_proba is not None else [])].head(10), use_container_width=True)
        
        with col2:
            st.subheader("Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            results_df['Diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax_pie)
            plt.ylabel('')
            st.pyplot(fig_pie)

        # Evaluation Metrics (Only if Ground Truth is available)
        if has_ground_truth:
            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")
            
            # Metrics
            acc = accuracy_score(y_true, prediction)
            prec = precision_score(y_true, prediction)
            rec = recall_score(y_true, prediction)
            f1 = f1_score(y_true, prediction)
            mcc = matthews_corrcoef(y_true, prediction)
            
            # AUC requires probabilities if available
            if prediction_proba is not None:
                auc = roc_auc_score(y_true, prediction_proba)
            else:
                auc = roc_auc_score(y_true, prediction)

            # Display Metrics in Columns with Styling
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("AUC", f"{auc:.3f}")
            m3.metric("Precision", f"{prec:.3f}")
            m4.metric("Recall", f"{rec:.3f}")
            m5.metric("F1 Score", f"{f1:.3f}")
            m6.metric("MCC", f"{mcc:.3f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            col_cm, col_empty = st.columns([1, 1])
            with col_cm:
                cm = confusion_matrix(y_true, prediction)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.title("Confusion Matrix")
                st.pyplot(fig)
            
        else:
            st.info("ℹ️ Upload a CSV with a 'target' column to see evaluation metrics and confusion matrix.")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.error("Please ensure your CSV matches the training data format.")

else:
    # Landing Page State
    st.info("👈 Please select a model and upload a CSV file from the sidebar to begin.")
    st.markdown("""
    ### Quick Start Guide
    1. **Download Sample Data**: If you don't have a dataset, check the project folder for `sample_test_data.csv`.
    2. **Upload**: Drag and drop the CSV file into the sidebar uploader.
    3. **Select Model**: Choose from 6 different algorithms to compare results.
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>Built for BITS Pilani ML Assignment 2</div>", unsafe_allow_html=True)
