import os
import json
import joblib
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ==========================================================
# Paths
# ==========================================================

MODEL_DIR = "models/traditional"

REPORT_DIR = "reports/traditional"

EMBEDDINGS_DIR = "data/processed/embeddings"

LABEL_MAP_PATH = "data/processed/label_mappings.json"

TARGETS = ["type", "queue", "priority"]

os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# Load test features
# ==========================================================

X_test = sp.load_npz(
    os.path.join(
        EMBEDDINGS_DIR,
        "tfidf_test.npz"
    )
)

# ==========================================================
# Load test labels
# ==========================================================

y_test = np.load(
    os.path.join(
        EMBEDDINGS_DIR,
        "labels_test.npz"
    )
)

# ==========================================================
# Load mappings
# ==========================================================

with open(LABEL_MAP_PATH, "r") as f:
    mappings = json.load(f)

# ==========================================================
# Create confusion matrices
# ==========================================================

for target in TARGETS:

    print(f"\nProcessing {target}")

    model_path = os.path.join(
        MODEL_DIR,
        f"{target}_best_model.joblib"
    )

    model = joblib.load(model_path)

    y_true = y_test[target]

    y_pred = model.predict(X_test)

    labels = list(
        mappings[target].keys()
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    disp.plot(
        cmap="Blues",
        xticks_rotation=45,
        ax=ax
    )

    plt.title(
        f"Traditional - {target}"
    )

    plt.tight_layout()

    save_path = os.path.join(
        REPORT_DIR,
        f"{target}_confusion_matrix.png"
    )

    plt.savefig(save_path)

    plt.close()

    print(f"Saved -> {save_path}")

print("\nDone.")