import os
import json
import torch
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)

# ==========================================================
# Paths
# ==========================================================

TEST_PATH = "data/processed/test.csv"

LABEL_MAP_PATH = "data/processed/label_mappings.json"

MODEL_DIR = "models"

REPORT_DIR = "reports/bert"

TARGETS = ["type", "queue", "priority"]

MAX_LENGTH = 128

os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {DEVICE}")

# ==========================================================
# Load data
# ==========================================================

test_df = pd.read_csv(TEST_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    mappings = json.load(f)

texts = test_df["text"].tolist()

# ==========================================================
# Generate confusion matrices
# ==========================================================

for target in TARGETS:

    print(f"\nProcessing {target}")

    model_path = os.path.join(
        MODEL_DIR,
        f"distilbert_{target}"
    )

    tokenizer = DistilBertTokenizer.from_pretrained(
        model_path
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(DEVICE)

    model.eval()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    encodings = {
        k: v.to(DEVICE)
        for k, v in encodings.items()
    }

    with torch.no_grad():

        outputs = model(**encodings)

        y_pred = outputs.logits.argmax(
            dim=1
        ).cpu().numpy()

    label_map = mappings[target]

    y_true = (
        test_df[target]
        .map(label_map)
        .values
    )

    labels = list(
        label_map.keys()
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
        f"DistilBERT - {target}"
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