# Diabetes Prediction Using Machine Learning Models

This project focuses on building and evaluating various machine learning models to predict diabetes using a medical dataset. The goal is to identify which model performs best in terms of precision, recall, F1-score, and overall diagnostic effectiveness.

## Models Tested

The following classification models were tested:

* **Logistic Regression**
* **Decision Tree (DT)**
* **Random Forest (RF)**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Classifier (SVC)**

## Dataset

The dataset used includes various health indicators such as BMI, blood pressure, cholesterol levels, mental and physical health metrics, and a binary diabetes diagnosis label (0 = no diabetes, 1 = diabetes).

## Preprocessing Steps

* Removed or imputed missing values
* Feature scaling using MinMaxScaler
* Applied SMOTE to balance the classes

## Evaluation Metrics

Each model was evaluated using the following metrics:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1 Score
* Hamming Loss

## Summary of Findings

* **Logistic Regression**: Served as a strong baseline, easy to interpret, fast to train, and showed good performance.
* **Decision Tree**: Performed well with tunable depth; benefited from balancing precision and recall. Slightly prone to overfitting.
* **Random Forest**: Achieved high accuracy and stability, especially after hyperparameter tuning. Performed best in terms of balanced performance.
* **K-Nearest Neighbors**: Moderate performance, sensitive to data scaling and number of neighbors.
* **Support Vector Classifier (SVC)**: High precision but suffered in recall. Best used when false positives are more costly.

## How to Run

```bash
# Run the notebook
jupyter notebook diabetes_model_comparison.ipynb
```

## Future Improvements

* Further hyperparameter tuning for SVC and KNN
* Feature selection to reduce noise
* Ensemble model combining top performers
* Deploy the best model using Flask or FastAPI

---

Feel free to contribute or raise issues!
