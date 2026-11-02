
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

print("Loading preprocessed data...")

# Load training and testing data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze("columns")

print("Training data:", X_train.shape)

# Train Logistic Regression
print("Training the model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train, y_train)

print("Model trained successfully!")

# Use a small sample to reduce computation time
background = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)
samples = X_test.sample(
    n=min(200, len(X_test)),
    random_state=42
)

print("Calculating SHAP values...")

# Explain predictions
explainer = shap.LinearExplainer(model, background)
shap_values = explainer(samples)

# Plot 1: Overall feature importance
shap.plots.bar(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=150)
plt.show()

# Plot 2: How features influence predictions
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.show()

print("Completed! SHAP plots saved successfully.")