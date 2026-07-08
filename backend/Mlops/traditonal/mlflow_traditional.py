import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODELS_DIR = PROJECT_ROOT / "models" / "traditional"

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

# ==========================================================
# MLflow
# ==========================================================

MLFLOW_DB = Path(__file__).resolve().parent / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


EXPERIMENT_NAME = "Traditional Models"

mlflow.set_experiment(EXPERIMENT_NAME)

# ==========================================================
# Load Test Features
# ==========================================================

print("=" * 60)
print("Loading TF-IDF test features...")
print("=" * 60)

X_test = sp.load_npz(
    EMBEDDINGS_DIR / "tfidf_test.npz"
)

labels = np.load(
    EMBEDDINGS_DIR / "labels_test.npz",
    allow_pickle=True,
)

print("TF-IDF Shape :", X_test.shape)
print("Available Labels :", labels.files)

# ==========================================================
# Load Models
# ==========================================================

print("\nLoading trained models...")

models = {
    "type": joblib.load(
        MODELS_DIR / "type_best_model.joblib"
    ),
    "queue": joblib.load(
        MODELS_DIR / "queue_best_model.joblib"
    ),
    "priority": joblib.load(
        MODELS_DIR / "priority_best_model.joblib"
    ),
}

print("Models Loaded Successfully.")

# ==========================================================
# Helper Function
# ==========================================================

def evaluate_model(model, X, y_true):
    """
    Returns evaluation metrics for a trained classifier.
    """

    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics
# ==========================================================
# Evaluate & Log to MLflow
# ==========================================================

TASKS = {
    "type": {
        "label": labels["type"],
        "model": models["type"],
        "artifact": "type_best_model.joblib",
    },
    "queue": {
        "label": labels["queue"],
        "model": models["queue"],
        "artifact": "queue_best_model.joblib",
    },
    "priority": {
        "label": labels["priority"],
        "model": models["priority"],
        "artifact": "priority_best_model.joblib",
    },
}

print("\n" + "=" * 60)
print("Evaluating Traditional Models")
print("=" * 60)

for task_name, task in TASKS.items():

    print(f"\n----- {task_name.upper()} -----")

    with mlflow.start_run(
        run_name=f"Traditional_{task_name}"
    ):

        # -------------------------------
        # Parameters
        # -------------------------------

        mlflow.log_param("Task", task_name)

        mlflow.log_param(
            "Representation",
            "TF-IDF",
        )

        mlflow.log_param(
            "Model",
            task["model"].__class__.__name__,
        )

        mlflow.log_param(
            "Feature_Count",
            X_test.shape[1],
        )

        # -------------------------------
        # Metrics
        # -------------------------------

        results = evaluate_model(
            task["model"],
            X_test,
            task["label"],
        )

        for metric_name, metric_value in results.items():

            mlflow.log_metric(
                metric_name,
                float(metric_value),
            )

        print(
            f"Accuracy : {results['accuracy']:.4f}"
        )

        print(
            f"Precision: {results['precision']:.4f}"
        )

        print(
            f"Recall   : {results['recall']:.4f}"
        )

        print(
            f"F1 Score : {results['f1_score']:.4f}"
        )

        # -------------------------------
        # Save Artifact
        # -------------------------------

        artifact_path = (
            MODELS_DIR /
            task["artifact"]
        )

        mlflow.log_artifact(
            str(artifact_path)
        )

        # -------------------------------
        # Log Model
        # -------------------------------

        mlflow.sklearn.log_model(
            sk_model=task["model"],
            name=f"{task_name}_best_model",
        )

        print("Logged Successfully.")

print("\n" + "=" * 60)
print("All Traditional Models Logged Successfully!")
print("=" * 60)