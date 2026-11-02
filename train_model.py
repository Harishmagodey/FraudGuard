# Credit Card Fraud Detection
# Model training and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.exceptions import NotFittedError


# -------------------------------
# 1. Load the prepared datasets
# -------------------------------

def load_datasets():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")

    y_train = pd.read_csv("y_train.csv").squeeze("columns")
    y_test = pd.read_csv("y_test.csv").squeeze("columns")

    if set(X_train.columns) != set(X_test.columns):
        raise ValueError(
            "Training and testing features do not match."
        )

    X_test = X_test[X_train.columns]

    if len(X_train) != len(y_train):
        raise ValueError("Training data and labels have different lengths.")

    if len(X_test) != len(y_test):
        raise ValueError("Testing data and labels have different lengths.")

    if set(pd.Series(y_train).unique()) != {0, 1}:
        raise ValueError("Labels must be 0 (Legitimate) and 1 (Fraud).")

    if set(pd.Series(y_test).unique()) - {0, 1}:
        raise ValueError("Test labels must contain only 0 and 1.")

    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError(
            "Missing feature values found. Clean the data before training."
        )

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain infinite or invalid values.")

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Testing features contain infinite or invalid values.")

    print("Datasets loaded successfully.")
    print(f"Training transactions: {len(X_train):,}")
    print(f"Testing transactions:  {len(X_test):,}")
    print(f"Number of features:    {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


# -------------------------------
# 2. Train the model
# -------------------------------


def train_model(X_train, y_train):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    print("\nModel training completed.")
    print("Algorithm: StandardScaler + Logistic Regression")

    return model


# -------------------------------
# 3. Evaluate the model
# -------------------------------

def evaluate_model(model, X_test, y_test, threshold=0.30):
    fraud_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X_test)[:, fraud_index]
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    avg_precision = average_precision_score(
        y_test, probabilities
    )

    print("\n" + "=" * 45)
    print("       FRAUD DETECTION RESULTS")
    print("=" * 45)

    print(f"Decision threshold : {threshold:.2f}")
    print(f"Accuracy           : {accuracy:.2%}")
    print(f"Fraud precision    : {precision:.2%}")
    print(f"Fraud recall       : {recall:.2%}")
    print(f"Fraud F1-score     : {f1:.2%}")
    print(f"Average precision  : {avg_precision:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("Confusion Matrix:")
    print(matrix)

    # Plot the confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraud"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    ax.set_title(f"Confusion Matrix (Threshold = {threshold:.2f})")
    fig.tight_layout()
    plt.show()

    return probabilities, predictions


# -------------------------------
# 4. Show influential features
# -------------------------------

def show_feature_importance(model, feature_names, top_n=10):
    coefficients = pd.Series(
        model.named_steps["logisticregression"].coef_[0],
        index=feature_names,
    )

    importance = coefficients.abs().sort_values(
        ascending=False
    ).head(top_n)

    print(f"\nTop {top_n} influential features:")
    print(importance.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))

    importance.sort_values().plot(
        kind="barh",
        ax=ax,
    )

    ax.set_title("Top Features by Logistic Regression Coefficient")
    ax.set_xlabel("Absolute coefficient magnitude")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    plt.show()


# -------------------------------
# 5. Run the complete workflow
# -------------------------------

def main():
    try:
        X_train, X_test, y_train, y_test = load_datasets()

        model = train_model(X_train, y_train)

        
        fraud_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, fraud_index]

        print("\nTHRESHOLD COMPARISON")
        print("-" * 65)
        print(
            f"{'Threshold':<12}"
            f"{'Precision':<14}"
            f"{'Recall':<12}"
            f"{'F1-score':<12}"
            f"{'Flagged':<10}"
        )

        for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            predictions = (probabilities >= threshold).astype(int)

            precision = precision_score(
                y_test, predictions, zero_division=0
            )
            recall = recall_score(
                y_test, predictions, zero_division=0
            )
            f1 = f1_score(
                y_test, predictions, zero_division=0
            )

            print(
                f"{threshold:<12.2f}"
                f"{precision:<14.2%}"
                f"{recall:<12.2%}"
                f"{f1:<12.2%}"
                f"{predictions.sum():<10}"
            )

        evaluate_model(
            model,
            X_test,
            y_test,
            threshold=0.90,
        )



        show_feature_importance(
            model,
            X_train.columns,
            top_n=10,
        )

        print("\nModel evaluation completed successfully.")

    except (FileNotFoundError, ValueError, NotFittedError) as error:
        print(f"\nError: {error}")
    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()