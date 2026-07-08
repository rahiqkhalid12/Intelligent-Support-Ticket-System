import os
import json
import joblib
import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = "data/processed/embeddings"
REPORT_DIR     = "reports/traditional"
MODEL_DIR      = "models/traditional"
LABEL_MAP_PATH = "data/processed/label_mappings.json"

LABEL_COLS = ["type", "queue", "priority"]

# Grid search for LinearSVC - tries multiple C values systematically
# instead of guessing one. cv=3 (not 5) to keep runtime reasonable on
# ~7600 training rows; f1_weighted matches how we pick the best model.
SVC_PARAM_GRID = {
    "C": [0.1, 0.5, 1, 2, 5, 10],
}
GRID_SEARCH_CV = 3

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_features():
    """Load pre-built TF-IDF sparse matrices for train / val / test."""
    X_train = sp.load_npz(os.path.join(EMBEDDINGS_DIR, "tfidf_train.npz"))
    X_val   = sp.load_npz(os.path.join(EMBEDDINGS_DIR, "tfidf_val.npz"))
    X_test  = sp.load_npz(os.path.join(EMBEDDINGS_DIR, "tfidf_test.npz"))
    print("Loaded TF-IDF Features")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")
    return X_train, X_val, X_test


def load_labels():
    """Load integer-encoded label arrays for train / val / test."""
    y_train = np.load(os.path.join(EMBEDDINGS_DIR, "labels_train.npz"))
    y_val   = np.load(os.path.join(EMBEDDINGS_DIR, "labels_val.npz"))
    y_test  = np.load(os.path.join(EMBEDDINGS_DIR, "labels_test.npz"))
    return y_train, y_val, y_test


def load_label_mappings():
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model definitions
# LinearSVC's C is now tuned per-target via grid search instead of a
# fixed guess - see train_linear_svc_with_grid_search().
# ---------------------------------------------------------------------------
def get_fixed_models():
    """Models trained with fixed hyperparameters (no search needed)."""
    return {
        "LogisticRegression": LogisticRegression(
            C=2,
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "MultinomialNB": MultinomialNB(),
    }


def train_linear_svc_with_grid_search(X_train, y_train):
    """Grid search over C for LinearSVC. Returns the best estimator
    already refit on the full training set (GridSearchCV does this
    automatically via refit=True, the default)."""
    base = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=3000,
    )
    search = GridSearchCV(
        base,
        SVC_PARAM_GRID,
        cv=GRID_SEARCH_CV,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  LinearSVC grid search best C : {search.best_params_['C']}")
    print(f"  LinearSVC grid search CV F1  : {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X, y, label_names=None):
    """Return a metrics dict; optionally include per-class report."""
    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="weighted", zero_division=0
    )
    report = classification_report(
        y, preds,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy"              : round(float(accuracy),  4),
        "precision"             : round(float(precision), 4),
        "recall"                : round(float(recall),    4),
        "f1_score"              : round(float(f1),        4),
        "classification_report" : report,
    }


# ---------------------------------------------------------------------------
# Per-target training loop
# ---------------------------------------------------------------------------
def train_target(target, X_train, X_val, X_test,
                 y_train, y_val, y_test, label_names=None):

    print(f"\n{'='*10} {target.upper()} {'='*10}")

    target_results = {}
    best_model    = None
    best_name     = ""
    best_val_f1   = -1.0

    # ── Fixed-hyperparameter models ─────────────────────────────────────────
    for name, model in get_fixed_models().items():
        print(f"\n  Training {name} ...")
        model.fit(X_train, y_train[target])

        val_metrics = evaluate(model, X_val, y_val[target], label_names)
        print(f"  Val F1 : {val_metrics['f1_score']}")

        target_results[name] = {"validation": val_metrics}

        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_model  = model
            best_name   = name

    # ── LinearSVC via grid search ───────────────────────────────────────────
    print(f"\n  Training LinearSVC (grid search over C) ...")
    svc_model = train_linear_svc_with_grid_search(X_train, y_train[target])
    val_metrics = evaluate(svc_model, X_val, y_val[target], label_names)
    print(f"  Val F1 : {val_metrics['f1_score']}")

    target_results["LinearSVC"] = {"validation": val_metrics}

    if val_metrics["f1_score"] > best_val_f1:
        best_val_f1 = val_metrics["f1_score"]
        best_model  = svc_model
        best_name   = "LinearSVC"

    # ── Test phase ──────────────────────────────────────────────────────────
    print(f"\n  Best Model : {best_name}")
    test_metrics = evaluate(best_model, X_test, y_test[target], label_names)
    print(f"  Test Accuracy : {test_metrics['accuracy']}")
    print(f"  Test F1 Score : {test_metrics['f1_score']}")

    target_results["best_model"]  = best_name
    target_results["test_results"] = test_metrics

    # ── Save results ─────────────────────────────────────────────────────────
    results_path = os.path.join(REPORT_DIR, f"{target}_results.json")
    with open(results_path, "w") as f:
        json.dump(target_results, f, indent=4)

    # ── Save best model ──────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"{target}_best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"  Saved model -> {model_path}")

    return best_model, best_name, test_metrics


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def save_summary(summary: dict):
    path = os.path.join(REPORT_DIR, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    X_train, X_val, X_test = load_features()
    y_train, y_val, y_test = load_labels()
    label_mappings          = load_label_mappings()

    summary = {}

    for target in LABEL_COLS:
        # Invert mapping so we can pass readable class names to sklearn
        id_to_label = {v: k for k, v in label_mappings[target].items()}
        label_names  = [id_to_label[i] for i in sorted(id_to_label)]

        _, best_name, test_metrics = train_target(
            target,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            label_names=label_names,
        )

        summary[target] = {
            "best_model" : best_name,
            "accuracy"   : test_metrics["accuracy"],
            "f1_score"   : test_metrics["f1_score"],
            "precision"  : test_metrics["precision"],
            "recall"     : test_metrics["recall"],
        }

    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    for target, metrics in summary.items():
        print(f"\n{target.upper()}")
        print(f"  Best Model : {metrics['best_model']}")
        print(f"  Accuracy   : {metrics['accuracy']}")
        print(f"  F1 Score   : {metrics['f1_score']}")

    save_summary(summary)
    print("\nDone! Results saved to", REPORT_DIR)


if __name__ == "__main__":
    main()