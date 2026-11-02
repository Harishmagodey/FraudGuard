
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression

# Load the training and testing data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze("columns")

# Train the same model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# Explain one test transaction
background = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)
explainer = shap.LinearExplainer(model, background)

transaction = X_test.iloc[[0]]
shap_values = explainer(transaction)

# Show the prediction
prediction = model.predict(transaction)[0]
probability = model.predict_proba(transaction)[0][1]

print("Predicted class:", prediction)
print("Prediction:", "Fraud" if prediction == 1 else "Legitimate")
print(f"Estimated fraud probability: {probability:.2%}")

# Explain the individual prediction
shap.plots.waterfall(shap_values[0], max_display=10)
plt.show()