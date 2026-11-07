# FraudGuard | Credit Card Fraud Detection

FraudGuard is a machine learning-based web application that detects potentially fraudulent credit card transactions and explains model predictions using Explainable AI.

**Live Demo:** [Open FraudGuard](https://fraudguard-4vu4rmaztb6eliiixms960.streamlit.app/)

## Features

- Credit card fraud detection using Logistic Regression.
- Interactive Streamlit dashboard.
- Fraud probability and adjustable decision threshold.
- Confusion matrix and model performance metrics.
- Transaction Explorer with CSV export.
- SHAP-based explanations for individual predictions.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Hugging Face Datasets

## Dataset

The dataset is hosted on Hugging Face to manage large CSV files separately from the GitHub repository.

**Dataset:** [FraudGuard Dataset](https://huggingface.co/datasets/harishmagodey/fraudguard-data)

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Harishmagodey/FraudGuard.git
   ```

2. Open the project folder:

   ```bash
   cd FraudGuard
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Project Purpose

To help users identify suspicious credit card transactions and understand why a machine learning model classifies a transaction as potentially fraudulent.

## Note

This is an academic project. Model predictions should be reviewed and are not proof of fraud.
