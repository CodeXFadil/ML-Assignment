# Machine Learning Assignment 2 Submission

## 1. GitHub Repository Link
This repository contains the complete source code, `requirements.txt`, and a clear `README.md`.

*   **Repository URL:** [https://github.com/CodeXFadil/ML-Assignment](https://github.com/CodeXFadil/ML-Assignment)

## 2. Live Streamlit App Link
The application has been successfully deployed using Streamlit Community Cloud.

*   **Live App URL:** [https://ml-assignment-fadil.streamlit.app](https://ml-assignment-fadil.streamlit.app)

## 3. Screenshot (BITS Virtual Lab)
*(Please insert the screenshot of the assignment execution on BITS Virtual Lab here before converting to PDF)*

<br>
<br>
<br>
<br>
<br>
<br>
[PLACEHOLDER FOR SCREENSHOT]
<br>
<br>
<br>
<br>
<br>
<br>

---

## 4. GitHub README Content

# ML Assignment 2: Breast Cancer Classification

This repository contains the solution for Machine Learning Assignment 2 (BITS Pilani - WILP). It includes multiple classification models trained on the Breast Cancer Wisconsin (Diagnostic) dataset and an interactive Streamlit web application.

## 1. Problem Statement
The goal is to develop predictive models to classify breast mass samples as either **Malignant (0)** or **Benign (1)** based on 30 real-valued features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## 2. Dataset Description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: UCI Machine Learning Repository (accessed via `sklearn.datasets`)
- **Instances**: 569
- **Features**: 30 numeric attributes (radius, texture, perimeter, area, smoothness, etc.) plus 1 target variable.
- **Target**: 
    - 0: Malignant
    - 1: Benign

## 3. Models Used & Comparison

The following 6 classification models were implemented and evaluated on a held-out test set (20% of data).

| ML Model Name       | Accuracy |      AUC | Precision | Recall | F1 Score | MCC Score |
|:--------------------|---------:|---------:|----------:|-------:|---------:|----------:|
| Logistic Regression |   0.9737 |   0.9940 |    0.9726 | 0.9861 |   0.9793 |    0.9433 |
| Decision Tree       |   0.9649 |   0.9673 |    0.9857 | 0.9583 |   0.9718 |    0.9260 |
| kNN                 |   0.9737 |   0.9974 |    0.9726 | 0.9861 |   0.9793 |    0.9433 |
| Naive Bayes         |   0.9561 |   0.9934 |    0.9718 | 0.9583 |   0.9650 |    0.9064 |
| Random Forest       |   0.9649 |   0.9977 |    0.9722 | 0.9722 |   0.9722 |    0.9246 |
| XGBoost             |   **0.9912** |   **1.0000** |    **1.0000** | 0.9861 |   **0.9930** |    **0.9814** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong baseline performance (Accuracy 97.37%) and high MCC (0.94), confirming the dataset's linear separability characteristics. |
| **Decision Tree** | Performed well (96.49%) but slightly lower AUC than ensemble methods, as expected for single trees. |
| **kNN** | Tied with Logistic Regression (97.37%), showing that neighbor-based classification works effectively when features are properly scaled. |
| **Naive Bayes** | While robust (AUC 0.99), it had slightly lower accuracy (95.61%) compared to other models, likely due to feature independence assumptions. |
| **Random Forest (Ensemble)** | Excellent AUC (0.9977) and balanced metrics, demonstrating the power of bagging to reduce variance. |
| **XGBoost (Ensemble)** | **Top Performer**. Achieved near-perfect metrics (Accuracy 99.12%, AUC 1.0), showcasing its superior gradient boosting capability on tabular data. |

## 4. How to Run Locally

1. **Clone the Repository**
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```
4. **Upload Data**
   - Use the `sample_test_data.csv` generated in the project folder to test the app.

## 5. Deployment
The app is designed to be deployed on Streamlit Community Cloud.
1. Push this code to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select this repository and the `app.py` file.
4. Click **Deploy**.

## 6. Project Structure

- `app.py`: The main Streamlit application file.
- `train_models.py`: Script to train models and save them.
- `models/`: Directory containing saved `.joblib` models and the scaler.
- `requirements.txt`: List of python dependencies.
- `sample_test_data.csv`: Sample data for testing the app.
