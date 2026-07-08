import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

CLEAN_PATH = "data/processed/tickets_clean.csv"
OUTPUT_DIR = "data/processed/"
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_mappings.json")


LABEL_COLS = ["type", "queue", "priority"]


STRATIFY_COL = "queue"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42


DEDUP_SIMILARITY_THRESHOLD = 0.95
DEDUP_BATCH_SIZE = 256
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded cleaned data: {df.shape}")

    for col in ["subject", "body", "answer"]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                print(f"{col}: re-filling {n_missing} NaN -> '' "
                    f"(CSV round-trip artifact)")
            df[col] = df[col].fillna("")

    return df


def build_text_field(df: pd.DataFrame) -> pd.DataFrame:

    df["text"] = (df["subject"] + " " + df["body"]).str.strip()
    return df


def drop_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["text"].str.strip() != ""].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with empty text (subject+body both empty)")
    else:
        print("No rows with empty text (as expected, body is complete)")
    return df


def remove_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    print(f"\nChecking for near-duplicates (threshold={DEDUP_SIMILARITY_THRESHOLD})...")
    print("Loading SBERT model for dedup pass...")
    model = SentenceTransformer(SBERT_MODEL_NAME)

    texts = df["text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    n = len(df)
    keep_mask = np.ones(n, dtype=bool)

    for start in range(0, n, DEDUP_BATCH_SIZE):
        end = min(start + DEDUP_BATCH_SIZE, n)
        if not keep_mask[start:end].any():
            continue

        sims = util.cos_sim(embeddings[start:end], embeddings[:end]).cpu().numpy()

        for local_i, global_i in enumerate(range(start, end)):
            if not keep_mask[global_i]:
                continue
            row_sims = sims[local_i, :global_i]
            if row_sims.size == 0:
                continue
            earlier_kept = keep_mask[:global_i]
            if (row_sims[earlier_kept] >= DEDUP_SIMILARITY_THRESHOLD).any():
                keep_mask[global_i] = False

    n_removed = (~keep_mask).sum()
    df_deduped = df[keep_mask].reset_index(drop=True)
    print(f"Near-duplicate removal: dropped {n_removed} rows "
          f"({n_removed / n * 100:.2f}%), {len(df_deduped)} rows remain")

    return df_deduped


def build_label_mappings(df: pd.DataFrame) -> dict:

    mappings = {}
    for col in LABEL_COLS:
        if col in df.columns:
            classes = sorted(df[col].dropna().unique())
            mappings[col] = {cls: i for i, cls in enumerate(classes)}
            print(f"\n{col} classes ({len(classes)}):")
            for cls, i in mappings[col].items():
                print(f"  {i}: {cls}")
    return mappings


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df["text_char_len"] = df["text"].str.len()
    df["text_word_len"] = df["text"].str.split().apply(len)
    return df


def split_data(df: pd.DataFrame):
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[STRATIFY_COL],
        random_state=RANDOM_STATE,
    )

    val_fraction_of_train_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        stratify=train_val[STRATIFY_COL],
        random_state=RANDOM_STATE,
    )
    return train, val, test


def report_split(name, split_df, full_df):
    print(f"\n{name}: {len(split_df)} rows "
          f"({len(split_df) / len(full_df):.1%} of total)")
    print(split_df[STRATIFY_COL].value_counts(normalize=True).round(3))


def main():
    df = load_data(CLEAN_PATH)
    df = build_text_field(df)
    df = drop_empty_text(df)
    df = remove_near_duplicates(df)  # MUST run before split_data()
    df = add_length_features(df)

    label_mappings = build_label_mappings(df)

    train, val, test = split_data(df)

    report_split("Train", train, df)
    report_split("Val", val, df)
    report_split("Test", test, df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_mappings, f, indent=2)
    print(f"\nSaved splits to {OUTPUT_DIR}/")
    print(f"Saved label mappings -> {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()