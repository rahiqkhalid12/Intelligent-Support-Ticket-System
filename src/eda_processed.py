import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"

REPORT_DIR = "reports/processed_eda"

os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    if "text" in df.columns:
        df["text"] = df["text"].fillna("")

    return df


train = load_data(TRAIN_PATH)
val = load_data(VAL_PATH)
test = load_data(TEST_PATH)

print("=" * 60)
print("SPLIT SHAPES")
print("=" * 60)
print(f"Train: {train.shape}")
print(f"Val:   {val.shape}")
print(f"Test:  {test.shape}")

# ============================================================
# VERIFY STRATIFICATION
# ============================================================

print("\n" + "=" * 60)
print("QUEUE DISTRIBUTION PER SPLIT")
print("=" * 60)

for name, split in [
    ("Train", train),
    ("Val", val),
    ("Test", test)
]:
    print(f"\n{name}:")
    print(
        split["queue"]
        .value_counts(normalize=True)
        .round(3)
    )

# ============================================================
# COMBINE SPLITS
# ============================================================

df = pd.concat(
    [train, val, test],
    ignore_index=True
)

print("\n" + "=" * 60)
print("FULL PROCESSED DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")

# ============================================================
# DATASET STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("TEXT STATISTICS")
print("=" * 60)

print(df["text_word_len"].describe().round(2))

print("\nCharacter Length Statistics:")
print(df["text_char_len"].describe().round(2))

# ============================================================
# QUEUE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("QUEUE DISTRIBUTION")
print("=" * 60)

queue_counts = df["queue"].value_counts()

print(
    df["queue"]
    .value_counts(normalize=True)
    .round(3)
)

# Class imbalance information
print("\nMost common queue:")
print(
    f"{queue_counts.idxmax()} "
    f"({queue_counts.max()} tickets)"
)

print("\nLeast common queue:")
print(
    f"{queue_counts.idxmin()} "
    f"({queue_counts.min()} tickets)"
)

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    y="queue",
    order=queue_counts.index
)

plt.title("Queue Distribution")
plt.xlabel("Count")
plt.ylabel("Queue")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "queue_distribution.png"
    )
)

plt.close()

# ============================================================
# TEXT LENGTH HISTOGRAM
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["text_word_len"],
    bins=40,
    kde=True
)

plt.title("Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "text_length_histogram.png"
    )
)

plt.close()

# ============================================================
# TEXT LENGTH BOXPLOT
# ============================================================

plt.figure(figsize=(10, 3))

sns.boxplot(
    x=df["text_word_len"]
)

plt.title("Text Length Boxplot")
plt.xlabel("Number of Words")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "text_length_boxplot.png"
    )
)

plt.close()

# ============================================================
# TEXT LENGTH BY PRIORITY
# ============================================================

# ============================================================
# TEXT LENGTH BY PRIORITY
# ============================================================

if "priority" in df.columns:

    # Order priorities naturally
    priority_order = ["low", "medium", "high"]

    # Keep only priorities that exist
    priority_order = [
        p for p in priority_order
        if p in df["priority"].unique()
    ]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="priority",
        y="text_word_len",
        order=priority_order
    )

    plt.title("Text Length by Priority")
    plt.xlabel("Priority")
    plt.ylabel("Number of Words")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORT_DIR,
            "priority_text_length.png"
        )
    )

    plt.close()
# ============================================================
# TOP WORDS (WITHOUT STOPWORDS)
# ============================================================

print("\n" + "=" * 60)
print("TOP 30 WORDS")
print("=" * 60)

word_vectorizer = CountVectorizer(
    stop_words="english"
)

X_words = word_vectorizer.fit_transform(df["text"])

word_counts = X_words.sum(axis=0).A1
words = word_vectorizer.get_feature_names_out()

top_words = pd.DataFrame(
    {
        "word": words,
        "count": word_counts
    }
)

top_words = (
    top_words
    .sort_values(
        "count",
        ascending=False
    )
    .head(30)
)

print(top_words)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=top_words,
    x="count",
    y="word"
)

plt.title("Top 30 Words")
plt.xlabel("Count")
plt.ylabel("Word")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "top_words.png"
    )
)

plt.close()

# ============================================================
# TOP BIGRAMS
# ============================================================

print("\n" + "=" * 60)
print("TOP 20 BIGRAMS")
print("=" * 60)

bigram_vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    stop_words="english"
)

X_bigrams = bigram_vectorizer.fit_transform(df["text"])

bigram_counts = X_bigrams.sum(axis=0).A1
bigrams = bigram_vectorizer.get_feature_names_out()

bigram_df = pd.DataFrame(
    {
        "bigram": bigrams,
        "count": bigram_counts
    }
)

bigram_df = (
    bigram_df
    .sort_values(
        "count",
        ascending=False
    )
    .head(20)
)

print(bigram_df)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=bigram_df,
    x="count",
    y="bigram"
)

plt.title("Top 20 Bigrams")
plt.xlabel("Count")
plt.ylabel("Bigram")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_DIR,
        "top_bigrams.png"
    )
)

plt.close()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print(f"Total rows: {len(df)}")
print(f"Total columns: {df.shape[1]}")
print(f"Plots saved to: {REPORT_DIR}")