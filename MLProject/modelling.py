# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import warnings

import mlflow
import mlflow.sklearn

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

warnings.filterwarnings("ignore")

# =========================================================
# MLFLOW TRACKING
# =========================================================
# Tracking URI & experiment name are set via environment variables
# MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME, or fall back to defaults.
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Loan Prediction Tuning")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# =========================================================
# LOAD DATASET
# =========================================================
data_path = os.environ.get("DATA_PATH", "loan_preprocessed.csv")
data = pd.read_csv(data_path)

# =========================================================
# FEATURE DAN TARGET
# =========================================================
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# HYPERPARAMETERS (overridable via MLflow Projects params)
# =========================================================
n_estimators = int(os.environ.get("N_ESTIMATORS", 100))
max_depth_env = os.environ.get("MAX_DEPTH", "None")
max_depth = None if max_depth_env == "None" else int(max_depth_env)

# =========================================================
# START MLFLOW RUN
# =========================================================
with mlflow.start_run():

    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("f1_score",  f1)

    # ── Confusion matrix plot ───────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # ── Log model ───────────────────────────────────────
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("=" * 45)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 45)
    print("Run ID:", mlflow.active_run().info.run_id)
