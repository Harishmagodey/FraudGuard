import pandas as pd
import numpy as np
import json
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("Loading local datasets...")
txn_df = pd.read_csv("seed_transactions.csv")
cust_df = pd.read_csv("clean_customers.csv")

# Merge customer demographics if missing in seed_transactions.csv
needed_cust_cols = ["customer_id", "gender", "dob", "city_pop"]
missing_cols = [c for c in ["gender", "age", "city_pop"] if c not in txn_df.columns]
if missing_cols:
    txn_df = txn_df.merge(cust_df[needed_cust_cols], on="customer_id", how="left")
    txn_df["age"] = pd.to_datetime(txn_df["trans_date_trans_time"]).dt.year - pd.to_datetime(txn_df["dob"]).dt.year

if "hour" not in txn_df.columns:
    dt = pd.to_datetime(txn_df["trans_date_trans_time"])
    txn_df["hour"] = dt.dt.hour
    txn_df["day_of_week"] = dt.dt.dayofweek

num_features = ["amt", "age", "city_pop", "hour", "day_of_week", "distance_km", "amt_to_avg_ratio"]
cat_features = ["category", "gender"]
feature_cols = num_features + cat_features

# Drop any rows with missing/NaN values before splitting
txn_df = txn_df.dropna(subset=feature_cols + ["is_fraud"])

X = txn_df[feature_cols]
y = txn_df["is_fraud"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")
}

evaluation_results = {}
trained_models = {}

for name, model in models.items():
    model.fit(X_train_proc, y_train)
    y_pred = model.predict(X_test_proc)
    evaluation_results[name] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    trained_models[name] = model
    print(f"✅ Trained {name} | F1: {evaluation_results[name]['f1_score']}")

best_model = trained_models["XGBoost"]
explainer = shap.TreeExplainer(best_model)

joblib.dump({
    "model": best_model,
    "preprocessor": preprocessor,
    "explainer": explainer,
    "feature_names": list(feature_names),
    "num_features": num_features,
    "cat_features": cat_features
}, "fraudguard_artifacts.pkl")

with open("model_metrics.json", "w") as f:
    json.dump(evaluation_results, f, indent=4)

print("🎉 Local model & SHAP explainer saved cleanly!")