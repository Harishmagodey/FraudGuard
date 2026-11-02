import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


st.set_page_config(
    page_title="FraudGuard | Credit Card Security",
    page_icon="💳",
    layout="wide",
)


# Page styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        background: linear-gradient(120deg, #101b35, #294778);
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .hero h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
        padding-bottom: 0.35rem;
    }

    .hero p {
        color: #e1e9f7;
        margin: 0.25rem 0;
    }

    div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 12px 10px;
    min-height: 105px;
    overflow: hidden;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.82rem;
    white-space: normal;
    overflow-wrap: anywhere;
}

div[data-testid="stMetricValue"] {
    font-size: clamp(1.15rem, 1.8vw, 1.8rem);
    white-space: nowrap;
}
div[data-testid="stMetric"] {
    min-width: 0;
    padding: 10px 8px;
}

div[data-testid="stMetricLabel"] {
    white-space: normal !important;
    overflow: visible !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>💳 FraudGuard</h1>
    <p><strong>Credit Card Fraud Detection</strong></p>
    <p>Explore suspicious transactions and understand model decisions
    with explainable AI.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze("columns")
    y_test = pd.read_csv("y_test.csv").squeeze("columns")

    if X_train.empty or X_test.empty:
        raise ValueError("The training or testing feature file is empty.")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature rows and label rows do not match.")

    if X_train.columns.duplicated().any():
        raise ValueError("Duplicate feature names were found.")

    if set(X_train.columns) != set(X_test.columns):
        raise ValueError(
            "Training and testing files contain different features."
        )

    X_test = X_test[X_train.columns]

    if not all(pd.api.types.is_numeric_dtype(dtype)
               for dtype in X_train.dtypes):
        raise ValueError("All model features must be numeric.")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Missing values were found in the feature files.")

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("The training data contains infinite values.")

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("The testing data contains infinite values.")

    if set(pd.unique(y_train)) != {0, 1}:
        raise ValueError("Training labels must use 0 for legitimate and 1 for fraud.")

    if not set(pd.unique(y_test)).issubset({0, 1}):
        raise ValueError("Testing labels must use 0 and 1.")

    return X_train, X_test, y_train, y_test


@st.cache_resource
def train_model(X_train, y_train):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    return model


def label_name(value):
    return "Fraud" if int(value) == 1 else "Legitimate"


def show_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraud"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=",d",
    )

    ax.set_title("Actual vs Predicted Transactions")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    fig.tight_layout()
    return fig


def show_feature_importance(model, feature_names):

    coefficients = pd.Series(
        model.named_steps["logistic_regression"].coef_[0],
        index=feature_names,
    )

    top_features = coefficients.abs().nlargest(10).index
    importance = coefficients.loc[top_features].sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot(kind="barh", ax=ax)

    ax.axvline(0, linewidth=0.8)
    ax.set_title("Features with the Largest Model Coefficients")
    ax.set_xlabel("Logistic regression coefficient")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    return fig


def show_shap_explanation(model, X_train, transaction):

    scaler = model.named_steps["scaler"]

    classifier = model.named_steps["logistic_regression"]

    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42
    )

    background_scaled = pd.DataFrame(
        scaler.transform(background),
        columns=X_train.columns
    )

    transaction_scaled = pd.DataFrame(
        scaler.transform(transaction),
        columns=transaction.columns
    )

    explainer = shap.LinearExplainer(
        classifier,
        background_scaled
    )

    explanation = explainer(transaction_scaled)

    shap.plots.waterfall(
        explanation[0],
        max_display=10,
        show=False
    )

    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.tight_layout()

    return fig


try:
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    fraud_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, fraud_index]

    # Sidebar
    st.sidebar.title("🛡️ FraudGuard")
    st.sidebar.caption("Detection and analysis workspace")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard Overview",
            "Transaction Explorer",
            "SHAP Explainability",
        ],
    )

    st.sidebar.divider()

    threshold = st.sidebar.slider(
        "Fraud decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.90,
        step=0.05,
        help=(
            "A transaction is flagged when its estimated fraud "
            "probability reaches this threshold."
        ),
    )

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
    avg_precision = average_precision_score(y_test, probabilities)

    st.sidebar.divider()
    st.sidebar.caption("Model: Logistic Regression")
    st.sidebar.caption(f"Test transactions: {len(X_test):,}")
    st.sidebar.caption(f"Input features: {X_test.shape[1]}")
    st.sidebar.caption(f"Current threshold: {threshold:.2f}")

    # Dashboard overview
    if page == "Dashboard Overview":
        st.subheader("Model Performance")
        st.caption(
            "Evaluation on the held-out test dataset. "
            "The selected threshold affects precision, recall and F1."
        )

    

        c1, c2 = st.columns(2, gap="medium")
        c3, c4 = st.columns(2, gap="medium")

        with c1:
         st.metric("Accuracy", f"{accuracy:.2%}")

        with c2:
          st.metric("Fraud Precision", f"{precision:.2%}")

        with c3:
          st.metric("Fraud Recall", f"{recall:.2%}")

        with c4:
          st.metric("Fraud F1-Score", f"{f1:.2%}")

        st.caption(
            f"Average precision (probability-based): {avg_precision:.2%}"
        )

        st.divider()

        left, right = st.columns([1, 1.15])

        fraud_count = int(predictions.sum())
        legitimate_count = len(predictions) - fraud_count

        with left:
            st.subheader("Prediction Summary")

            st.metric(
                "Flagged as Fraud",
                f"{fraud_count:,}",
            )

            st.metric(
                "Predicted Legitimate",
                f"{legitimate_count:,}",
            )

            st.info(
                "Flagged transactions are model predictions, not confirmed "
                "fraud. They should be reviewed before action is taken."
            )

        with right:
            st.subheader("Confusion Matrix")

            fig = show_confusion_matrix(y_test, predictions)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.divider()

        st.subheader("Most Influential Features")

        fig = show_feature_importance(model, X_train.columns)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.divider()

        # Fraud Detection Analysis
        st.subheader("Fraud Detection Analysis")

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("True Positives", f"{tp:,}")

        with col2:
            st.metric("False Positives", f"{fp:,}")

        with col3:
            st.metric("True Negatives", f"{tn:,}")

        with col4:
            st.metric("False Negatives", f"{fn:,}")

        st.markdown("### Model Analysis")

        if fp > 0 or fn > 0:
            st.warning(
                f"The model correctly detected {tp:,} fraudulent "
                f"transactions and {tn:,} legitimate transactions. "
                f"However, it incorrectly flagged {fp:,} legitimate "
                f"transactions as fraud and missed {fn:,} fraudulent "
                f"transactions."
            )
        else:
            st.success(
                "The model correctly classified all test transactions."
            )

        st.caption(
            "Coefficient signs indicate the direction of influence on the "
            "model's linear score. Magnitudes depend on feature scales, "
            "so they should not automatically be treated as comparable "
            "measures of real-world importance."
        )

    # Transaction explorer
    elif page == "Transaction Explorer":
        st.subheader("Transaction Explorer")
        st.write(
            "Inspect individual test transactions, filter predictions, "
            "and export the results. Row numbers start at zero."
        )

        results = pd.DataFrame({
            "Transaction Row": np.arange(len(X_test)),
            "Actual Label": [
                label_name(value) for value in np.asarray(y_test)
            ],
            "Predicted Label": [
                label_name(value) for value in predictions
            ],
            "Fraud Probability (%)": np.round(
                probabilities * 100, 2
            ),
        })

        f1_col, f2_col = st.columns([1, 1])

        with f1_col:
            search_row = st.number_input(
                "Transaction row to inspect",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                step=1,
            )

        with f2_col:
            filter_label = st.selectbox(
                "Filter by prediction",
                ["All", "Fraud", "Legitimate"],
            )

        filtered = results

        if filter_label != "All":
            filtered = results[
                results["Predicted Label"] == filter_label
            ]

        st.dataframe(
            filtered.head(1000),
            width="stretch",
            hide_index=True,
        )

        if len(filtered) > 1000:
            st.caption(
                "Showing the first 1,000 rows. Download the CSV "
                "to access all matching results."
            )

        st.download_button(
            "⬇️ Download filtered predictions",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )

        st.divider()

        row_index = int(search_row)
        transaction = X_test.iloc[[row_index]]
        row_probability = float(probabilities[row_index])
        row_prediction = int(predictions[row_index])
        row_actual = int(y_test.iloc[row_index])

        st.subheader(f"Transaction Details: Row {row_index}")

        a, b, c = st.columns(3)
        a.metric("Predicted Result", label_name(row_prediction))
        b.metric("Fraud Probability", f"{row_probability:.2%}")
        c.metric("Actual Dataset Label", label_name(row_actual))

        st.dataframe(
            transaction,
            width="stretch",
        )

        st.caption(
            "The actual label comes from the test dataset. In a live "
            "system, the true label may not be known immediately."
        )

    # SHAP explainability
    elif page == "SHAP Explainability":
        st.subheader("Explain an Individual Prediction")

        st.write(
            "Explore how feature values contribute to the model's "
            "prediction for a selected transaction."
        )

        row_index = st.number_input(
            "Select test transaction row",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1,
        )

        row_index = int(row_index)
        transaction = X_test.iloc[[row_index]]

        probability = float(probabilities[row_index])
        predicted = int(predictions[row_index])
        actual = int(y_test.iloc[row_index])

        a, b, c = st.columns(3)
        a.metric("Predicted Result", label_name(predicted))
        b.metric("Fraud Probability", f"{probability:.2%}")
        c.metric("Actual Dataset Label", label_name(actual))

        st.divider()
        st.subheader("Why did the model make this prediction?")

        with st.spinner("Preparing the SHAP explanation..."):
            fig = show_shap_explanation(
                model,
                X_train,
                transaction,
            )

            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.caption(
            "The SHAP waterfall explains contributions to the model's "
            "output, which for this logistic regression explainer is "
            "typically on the log-odds scale. It does not directly show "
            "changes in fraud probability or establish causation."
        )

        st.divider()
        st.subheader("Transaction Feature Values")
        st.dataframe(transaction, width="stretch")


except FileNotFoundError as e:
    st.error(f"Required data file not found: {e.filename}")
    st.info(
        "Keep app.py, X_train.csv, X_test.csv, y_train.csv and "
        "y_test.csv in the same project folder."
    )

except (ValueError, KeyError) as e:
    st.error(f"Please check the dataset: {e}")

except Exception as e:
    st.error("The dashboard could not complete this operation.")
    st.caption(f"Error details: {type(e).__name__}: {e}")