# FraudGuard | Credit Card Fraud Detection

FraudGuard is a machine learning-based web application that detects potentially fraudulent credit card transactions and explains model predictions using Explainable AI.

## Features

* Credit card fraud detection using Logistic Regression.
* Interactive Streamlit dashboard.
* Fraud probability and adjustable decision threshold.
* Confusion matrix and model performance metrics.
* Transaction Explorer with CSV export.
* SHAP-based explanations for individual predictions.

## Technologies Used

* Python
* Streamlit
* Pandas and NumPy
* Scikit-learn
* SHAP
* Matplotlib

## How to Run Locally

1. Clone the repository:
   `git clone https://github.com/Harishmagodey/FraudGuard.git`

2. Install dependencies:
   `pip install -r requirements.txt`

3. Run the application:
   `streamlit run app.py`

## Project Purpose

To help users identify suspicious credit card transactions and understand why a machine learning model classifies a transaction as potentially fraudulent.

**Note:** This is an academic project. Model predictions should be reviewed and are not proof of fraud.
