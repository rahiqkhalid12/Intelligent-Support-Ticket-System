import os
import json
import numpy as np
import pandas as pd
import torch
import sys
import warnings

# ==========================================
# CLEAN OUTPUT (NO WARNINGS)
# ==========================================

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers.utils import logging
logging.set_verbosity_error()

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ==========================================
# Config
# ==========================================

MODEL_NAME = "distilbert-base-uncased"

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"
TEST_PATH  = "data/processed/test.csv"

LABEL_MAP_PATH = "data/processed/label_mappings.json"

TARGETS = ["type", "queue", "priority"]

MAX_LENGTH    = 128
BATCH_SIZE    = 8   # reverted from 16 - the batch=16 run scored LOWER across all
                     # three targets than the original batch=8 run, so this wasn't
                     # actually an improvement despite reducing overfitting symptoms
EPOCHS        = 10  # ceiling only - early stopping should trigger well before this
LEARNING_RATE = 2e-5

MODEL_DIR  = "models"
REPORT_DIR = "reports/bert"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# Load data once
# ==========================================

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    mappings = json.load(f)

# ==========================================
# Dataset class
# ==========================================

class TicketDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ==========================================
# Metrics
# ==========================================

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels      = pred.label_ids

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0,
    )

    return {
        "accuracy" : accuracy,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
    }

# ==========================================
# Progress Bar Callback
# ==========================================

class EpochProgressBar(TrainerCallback):

    def on_epoch_begin(self, args, state, control, **kwargs):

        self.current_epoch = int(state.epoch) + 1

        self.total_epochs = int(args.num_train_epochs)

        self.steps_per_epoch = max(
            1,
            state.max_steps // self.total_epochs
        )

        self.epoch_start_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):

        steps_done = (
            state.global_step
            - self.epoch_start_step
        )

        progress = min(
            steps_done / self.steps_per_epoch,
            1.0
        )

        self.draw_bar(progress)

    def on_epoch_end(self, args, state, control, **kwargs):

        self.draw_bar(1.0)

        print()

    def draw_bar(self, progress):

        bar_length = 20

        filled = int(
            bar_length * progress
        )

        bar = (

            "*" * filled

            + "-" * (bar_length - filled)

        )

        percent = progress * 100

        sys.stdout.write(

            f"\rEpoch {self.current_epoch}/{self.total_epochs} "

            f"[{bar}] {percent:5.1f}%"

        )

        sys.stdout.flush()

# ==========================================
# Pretty Results Callback
# ==========================================

class PrettyResultsCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):

        epoch = int(metrics["epoch"])

        print("\n")

        print("Validation Results")
        print("------------------")

        print(f"Loss      : {metrics['eval_loss']:.4f}")

        print(f"Accuracy  : {metrics['eval_accuracy']:.4f}")

        print(f"Precision : {metrics['eval_precision']:.4f}")

        print(f"Recall    : {metrics['eval_recall']:.4f}")

        print(f"F1 Score  : {metrics['eval_f1']:.4f}")

        print("-"*40)

# ==========================================
# Train one target
# ==========================================

def train_target(target):

    print(f"\n{'='*10} {target.upper()} {'='*10}")

    label_map = mappings[target]
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_labels = train_df[target].map(label_map).tolist()
    val_labels   = val_df[target].map(label_map).tolist()
    test_labels  = test_df[target].map(label_map).tolist()

    def tokenize(df):
        return tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_dataset = TicketDataset(tokenize(train_df), train_labels)
    val_dataset   = TicketDataset(tokenize(val_df),   val_labels)
    test_dataset  = TicketDataset(tokenize(test_df),  test_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_map),
    )

    training_args = TrainingArguments(
        output_dir=f"./temp/{target}",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        weight_decay=0.01,
        warmup_ratio=0.1,  # stabilizes early training - especially helpful on a
                           # small (~7600 row) training set where the first few
                           # hundred steps can otherwise be noisy/unstable
        report_to="none",

        logging_strategy="no",
        disable_tqdm=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.005
            ),
            EpochProgressBar(),
            PrettyResultsCallback(),
        ],
    )

    trainer.train()
    print("\nTraining Finished")
    print(f"Best checkpoint loaded (lowest/best eval_f1): "
          f"step {trainer.state.best_global_step}, "
          f"metric value {trainer.state.best_metric:.4f}")

    # ── Test ────────────────────────────────────────────────────────────────
    predictions = trainer.predict(test_dataset)
    y_pred      = np.argmax(predictions.predictions, axis=1)

    accuracy = accuracy_score(test_labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, average="weighted", zero_division=0,
    )

    results = {
        "accuracy" : round(float(accuracy),  4),
        "precision": round(float(precision), 4),
        "recall"   : round(float(recall),    4),
        "f1_score" : round(float(f1),        4),
    }

    with open(f"{REPORT_DIR}/{target}.json", "w") as f:
        json.dump(results, f, indent=4)

    save_path = f"{MODEL_DIR}/distilbert_{target}"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"  Test Results: {results}")

# ==========================================
# Main
# ==========================================

def main():
    for target in TARGETS:
        train_target(target)

    print("\nDone!")
    print("Models saved.")
    print("Reports saved.")


if __name__ == "__main__":
    main()