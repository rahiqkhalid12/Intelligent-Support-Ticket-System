import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Config - adjust to match your actual file paths / column names
# ---------------------------------------------------------------------------
RAW_PATH = "data/raw/dataset-tickets.csv"
CLEAN_PATH = "data/processed/tickets_clean.csv"

LANGUAGE_COL = "language"
KEEP_LANGUAGE = "en"

TEXT_COLS = ["subject", "body", "answer"]
TARGET_COLS = ["type", "queue", "priority"]
TAG_COLS = [f"tag_{i}" for i in range(1, 9)]  # tag_1 ... tag_8


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    print("\nLanguage distribution before filtering:")
    print(df[LANGUAGE_COL].value_counts())

    df_en = df[df[LANGUAGE_COL].str.lower() == KEEP_LANGUAGE].copy()
    print(f"\nKept {len(df_en)} English tickets "
          f"(dropped {len(df) - len(df_en)} non-English)")
    return df_en


def normalize_text(text) -> str:
    """Collapse whitespace/newlines and strip ends.
    NaN -> '' (not the literal string 'nan')."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def handle_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    subject : ~13.4% missing. body is complete, so these rows still
              carry full signal - we keep the row and fill subject
              with '' rather than a placeholder word. When subject
              and body are later concatenated, an empty subject is
              just a realistic "ticket submitted without a subject".

    body    : essentially complete; cleaned for safety anyway.

    answer  : ~0.02% missing (a few rows). 'answer' is not an input
              feature for type/queue/priority classification, so we
              don't drop rows for this - just fill with '' so no
              downstream code ever trips on a NaN.
    """
    for col in TEXT_COLS:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            df[col] = df[col].apply(normalize_text)
            if n_missing:
                print(f"{col}: filled {n_missing} missing values with ''")
    return df


def handle_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    tag_1-tag_3 : essentially complete -> any stray NaN becomes 'none'.

    tag_4       : ~10.7% missing. A NaN here most likely means "no
                  4th tag was assigned to this ticket" - that's a
                  real, meaningful category, not a data-collection
                  gap. We encode it explicitly as 'none' instead of
                  imputing a fabricated tag.

    tag_5-8     : ~49% to ~98% missing. At this sparsity there isn't
                  enough signal to use these as features, and any
                  imputation would just be invented data. We drop
                  these columns entirely rather than guess.
    """
    keep_tags = [c for c in ["tag_1", "tag_2", "tag_3", "tag_4"] if c in df.columns]
    drop_tags = [c for c in ["tag_5", "tag_6", "tag_7", "tag_8"] if c in df.columns]

    for col in keep_tags:
        n_missing = df[col].isna().sum()
        if n_missing:
            df[col] = df[col].fillna("none")
            print(f"{col}: filled {n_missing} missing values with 'none'")

    if drop_tags:
        print(f"Dropping sparse tag columns: {drop_tags}")
        df = df.drop(columns=drop_tags)

    return df


def check_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """type / queue / priority are the labels for classification.
    If any of these were missing, the row would be useless for
    supervised learning - so (only here) we drop the row."""
    existing = [c for c in TARGET_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=existing)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing target labels {existing}")
    else:
        print(f"No missing values in target columns {existing} (as expected)")
    return df


def safety_duplicate_check(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "body"])
    dropped = before - len(df)
    print(f"Duplicate safety check: dropped {dropped} duplicate rows (subject+body)")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    language : after filter_english(), every remaining row is 'en' -
               this column now has zero information for modeling.

    version  : dataset/schema metadata (not a property of the ticket
               itself) - not useful as a feature.

    Both are dropped if present.
    """
    drop_cols = [c for c in ["language", "version"] if c in df.columns]
    if drop_cols:
        print(f"Dropping zero-information / metadata columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


def main():
    df = load_data(RAW_PATH)
    df = filter_english(df)
    df = handle_text_columns(df)
    df = handle_tag_columns(df)
    df = check_target_columns(df)
    df = safety_duplicate_check(df)
    df = drop_unused_columns(df)

    print("\nRemaining missing values per column:")
    print(df.isna().sum())

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset: {df.shape} -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()