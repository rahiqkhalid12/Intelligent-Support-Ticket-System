"""
mlflow_distilbert.py
--------------------
Evaluates the saved DistilBERT models (type / queue / priority) on
the test set and logs parameters + metrics to MLflow.

This is a STANDALONE logging script — it does NOT retrain the models.
It reads the already-trained model folders saved by train_classifier.py
and the training hyperparameters from train_config.json (written by
train_classifier.py on its last run, so the logged params are always
accurate and never hardcoded here).

Run:
    python mlflow_distilbert.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ==========================================================
# Paths
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
MLFLOW_DB    = Path(__file__).resolve().parent / "mlflow.db"

# Training config written by train_classifier.py — ensures logged
# hyperparameters always match what was actually used for training,
# rather than being hardcoded constants in this file.
TRAIN_CONFIG_PATH = Path(__file__).resolve().parent / "train_config.json"

MODEL_FOLDERS = {
    "type":     MODELS_DIR / "distilbert_type",
    "queue":    MODELS_DIR / "distilbert_queue",
    "priority": MODELS_DIR / "distilbert_priority",
}

# Fallback hyperparameters used only if train_config.json doesn't exist
DEFAULT_CONFIG = {
    "model_name":    "distilbert-base-uncased",
    "max_length":    128,
    "batch_size":    8,
    "learning_rate": 2e-5,
    "epochs":        10,
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.005,
}

# ==========================================================
# MLflow setup
# ==========================================================
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("DistilBERT_Classifier")

# ==========================================================
# Device
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ==========================================================
# Load training config (written by train_classifier.py)
# ==========================================================
def load_train_config() -> dict:
    if TRAIN_CONFIG_PATH.exists():
        with open(TRAIN_CONFIG_PATH) as f:
            config = json.load(f)
        print(f"Loaded training config from {TRAIN_CONFIG_PATH}")
        return config
    print(f"Warning: {TRAIN_CONFIG_PATH} not found — using default config.")
    print("Add this to train_classifier.py to auto-save it:\n"
          "  json.dump(config_dict, open('train_config.json','w'), indent=2)")
    return DEFAULT_CONFIG

# ==========================================================
# Dataset
# ==========================================================
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

# ==========================================================
# Evaluation
# ==========================================================
def evaluate_model(model, dataset, batch_size, label_names=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy":  round(float(accuracy_score(all_labels, all_preds)), 4),
        "precision": round(float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)), 4),
    }

    # Per-class report saved as artifact (not logged as metrics to keep
    # MLflow UI clean — per-class breakdown is in the artifact)
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        zero_division=0,
    )
    return metrics, report

# ==========================================================
# MLflow logging for one task
# ==========================================================
def log_task(task, config, test_df, label_maps):
    model_folder = MODEL_FOLDERS[task]

    if not model_folder.exists():
        print(f"Skipping {task} — model folder not found: {model_folder}")
        return None

    print(f"\nLoading {task} model from {model_folder}...")
    tokenizer  = AutoTokenizer.from_pretrained(model_folder)
    model      = AutoModelForSequenceClassification.from_pretrained(model_folder)
    model.to(DEVICE)

    label_map   = label_maps[task]
    id_to_label = {v: k for k, v in label_map.items()}
    label_names = [id_to_label[i] for i in sorted(id_to_label)]

    labels  = test_df[task].map(label_map).tolist()
    dataset = TicketDataset(
        texts=test_df["text"].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        max_length=config.get("max_length", 128),
    )

    print("Running inference...")
    metrics, report = evaluate_model(
        model=model,
        dataset=dataset,
        batch_size=config.get("batch_size", 8),
        label_names=label_names,
    )

    with mlflow.start_run(run_name=f"DistilBERT_{task}"):

        # ── Parameters ──────────────────────────────────────
        mlflow.log_param("task",                      task)
        mlflow.log_param("model_name",                config.get("model_name", "distilbert-base-uncased"))
        mlflow.log_param("max_length",                config.get("max_length", 128))
        mlflow.log_param("batch_size",                config.get("batch_size", 8))
        mlflow.log_param("learning_rate",             config.get("learning_rate", 2e-5))
        mlflow.log_param("epochs_ceiling",            config.get("epochs", 10))
        mlflow.log_param("early_stopping_patience",   config.get("early_stopping_patience", 2))
        mlflow.log_param("early_stopping_threshold",  config.get("early_stopping_threshold", 0.005))
        mlflow.log_param("num_classes",               len(label_map))
        mlflow.log_param("test_samples",              len(test_df))
        mlflow.log_param("device",                    str(DEVICE))
        mlflow.log_param("model_path",                str(model_folder))

        # ── Metrics ─────────────────────────────────────────
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # ── Per-class report as text artifact ────────────────
        report_dir = Path(__file__).resolve().parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"distilbert_{task}_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path), artifact_path="classification_reports")

        # ── Tag for easy filtering in MLflow UI ─────────────
        mlflow.set_tag("model_type", "DistilBERT")
        mlflow.set_tag("task", task)

    # Print summary
    print(f"\n{task.upper()} — logged to MLflow")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"  {name:<10}: {value}")

    return metrics

# ==========================================================
# Main
# ==========================================================
def main():
    print("\n" + "=" * 60)
    print("DistilBERT MLflow Evaluation")
    print("=" * 60)

    config = load_train_config()

    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["text"] = test_df["text"].fillna("")
    print(f"Test samples: {len(test_df)}")

    with open(DATA_DIR / "label_mappings.json") as f:
        label_maps = json.load(f)

    summary = {}
    for task in MODEL_FOLDERS:
        result = log_task(task, config, test_df, label_maps)
        if result:
            summary[task] = result

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task, metrics in summary.items():
        print(f"\n{task.upper()}")
        for name, value in metrics.items():
            print(f"  {name:<10}: {value}")

    print("\n" + "=" * 60)
    print("MLflow UI:")
    print(f"  mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}")
    print("=" * 60)


if __name__ == "__main__":
    main()