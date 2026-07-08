import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SPLIT_DIR = "data/processed/"
TRAIN_PATH = os.path.join(SPLIT_DIR, "train.csv")
VAL_PATH = os.path.join(SPLIT_DIR, "val.csv")
TEST_PATH = os.path.join(SPLIT_DIR, "test.csv")
LABEL_MAP_PATH = os.path.join(SPLIT_DIR, "label_mappings.json")

OUTPUT_DIR = "data/processed/embeddings"

# stop_words="english" is fine HERE - it only affects the TF-IDF
# vocabulary, not the saved `text` column (which keeps stopwords for
# the sentence-transformer model, as discussed).
TFIDF_PARAMS = dict(
    max_features=30000,       # up from 20000 - more room for trigrams
    ngram_range=(1, 3),       # up from (1, 2) - phrases like "unable to login" carry more signal
    stop_words="english",
    sublinear_tf=True,
    min_df=2,
)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast on CPU
SBERT_BATCH_SIZE = 64

LABEL_COLS = ["type", "queue", "priority"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Same CSV round-trip caveat as before - `text` should never be
    # empty (drop_empty_text in preprocessing.py guarantees this),
    # but guard anyway so a stray NaN can't break .tolist() encoding.
    df["text"] = df["text"].fillna("")
    return df


def build_tfidf_text(df: pd.DataFrame) -> "pd.Series[str]":
    """Experiment: append tag_1-tag_4 to the text used ONLY for
    TF-IDF (not for SBERT - sentence embeddings work best on natural
    language, and appending short category-like tags would just add
    noise to a model that already understands context).

    tag_4 has 'none' for missing values (set in clean.py), so it's
    safe to always include - no NaN handling needed here.

    This does NOT change the saved 'text' column itself - it only
    affects what TfidfVectorizer sees, kept local to this function so
    SBERT embeddings (built separately below) are unaffected."""
    tag_cols = [c for c in ["tag_1", "tag_2", "tag_3", "tag_4"] if c in df.columns]
    if not tag_cols:
        return df["text"]

    tags_combined = df[tag_cols].fillna("").astype(str).agg(" ".join, axis=1)
    return (df["text"] + " " + tags_combined).str.strip()


def load_label_mappings(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# TF-IDF (traditional)
# ---------------------------------------------------------------------------
def build_tfidf(train_text, val_text, test_text, output_dir):
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)

    X_train = vectorizer.fit_transform(train_text)  # fit ONLY on train
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    sp.save_npz(os.path.join(output_dir, "tfidf_train.npz"), X_train)
    sp.save_npz(os.path.join(output_dir, "tfidf_val.npz"), X_val)
    sp.save_npz(os.path.join(output_dir, "tfidf_test.npz"), X_test)
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))

    print("\nTF-IDF")
    print(f"  vocabulary size : {len(vectorizer.vocabulary_)}")
    print(f"  train shape     : {X_train.shape}")
    print(f"  val shape       : {X_val.shape}")
    print(f"  test shape      : {X_test.shape}")

    return X_train.shape, X_val.shape, X_test.shape


# ---------------------------------------------------------------------------
# Sentence embeddings (transformer-based)
# ---------------------------------------------------------------------------
def build_sentence_embeddings(train_text, val_text, test_text, output_dir):
    model = SentenceTransformer(SBERT_MODEL_NAME)

    def encode(texts, name):
        emb = model.encode(
            texts.tolist(),
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,  # unit-norm -> cosine-friendly
        )
        np.save(os.path.join(output_dir, f"sbert_{name}.npy"), emb)
        return emb

    X_train = encode(train_text, "train")
    X_val = encode(val_text, "val")
    X_test = encode(test_text, "test")

    print("\nSentence embeddings")
    print(f"  model       : {SBERT_MODEL_NAME}")
    print(f"  dim         : {X_train.shape[1]}")
    print(f"  train shape : {X_train.shape}")
    print(f"  val shape   : {X_val.shape}")
    print(f"  test shape  : {X_test.shape}")

    return X_train.shape, X_val.shape, X_test.shape


# ---------------------------------------------------------------------------
# Labels (shared by both representations)
# ---------------------------------------------------------------------------
def build_label_arrays(df, label_mappings, name, output_dir):
    """Encode type/queue/priority to ints using label_mappings.json
    (built from the FULL dataset in preprocessing.py, so ids are
    consistent across train/val/test)."""
    arrays = {}
    for col in LABEL_COLS:
        mapping = label_mappings[col]
        unmapped = set(df[col].astype(str)) - set(mapping.keys())
        if unmapped:
            raise ValueError(
                f"{name}/{col}: labels not found in label_mappings.json: {unmapped}"
            )
        arrays[col] = df[col].astype(str).map(mapping).to_numpy(dtype=np.int64)

    np.savez(os.path.join(output_dir, f"labels_{name}.npz"), **arrays)
    return arrays


# ---------------------------------------------------------------------------
# Metadata (for your report / reproducibility)
# ---------------------------------------------------------------------------
def save_metadata(tfidf_shapes, sbert_shapes, output_dir):
    metadata = {
        "tfidf": {
            "params": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in TFIDF_PARAMS.items()
            },
            "shapes": {
                "train": list(tfidf_shapes[0]),
                "val": list(tfidf_shapes[1]),
                "test": list(tfidf_shapes[2]),
            },
        },
        "sentence_embeddings": {
            "model": SBERT_MODEL_NAME,
            "shapes": {
                "train": list(sbert_shapes[0]),
                "val": list(sbert_shapes[1]),
                "test": list(sbert_shapes[2]),
            },
        },
    }
    with open(os.path.join(output_dir, "embedding_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train = load_split(TRAIN_PATH)
    val = load_split(VAL_PATH)
    test = load_split(TEST_PATH)
    label_mappings = load_label_mappings(LABEL_MAP_PATH)

    print(f"Loaded train={train.shape}, val={val.shape}, test={test.shape}")

    tfidf_shapes = build_tfidf(train["text"], val["text"], test["text"], OUTPUT_DIR)
    sbert_shapes = build_sentence_embeddings(
        train["text"], val["text"], test["text"], OUTPUT_DIR
    )

    for name, df in [("train", train), ("val", val), ("test", test)]:
        build_label_arrays(df, label_mappings, name, OUTPUT_DIR)

    save_metadata(tfidf_shapes, sbert_shapes, OUTPUT_DIR)

    print(f"\nSaved all embeddings + labels -> {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()