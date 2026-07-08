import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CONFIGURATION
# ==============================

DATA_PATH = "data/raw/dataset-tickets.csv"
REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

sns.set_style("whitegrid")


# ==============================
# LOAD DATA
# ==============================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset Shape: {df.shape}")


# ==============================
# BASIC INFORMATION
# ==============================

print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

print(df.info())


# ==============================
# MISSING VALUES
# ==============================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum().sort_values(ascending=False)

missing_df = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": round((missing / len(df)) * 100, 2)
})

print(missing_df)


# ==============================
# DUPLICATES
# ==============================

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_rows = df.duplicated().sum()

print(f"Complete duplicate rows: {duplicate_rows}")

if "subject" in df.columns and "body" in df.columns:
    ticket_duplicates = df.duplicated(
        subset=["subject", "body"]
    ).sum()

    print(
        f"Duplicate tickets (subject + body): "
        f"{ticket_duplicates}"
    )


# ==============================
# LANGUAGE DISTRIBUTION
# ==============================

if "language" in df.columns:

    print("\n" + "=" * 60)
    print("LANGUAGE DISTRIBUTION")
    print("=" * 60)

    print(df["language"].value_counts())

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="language",
        order=df["language"].value_counts().index
    )

    plt.title("Language Distribution")
    plt.xlabel("Language")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/raw_language_distribution.png"
    )
    plt.close()


# ==============================
# TYPE DISTRIBUTION
# ==============================

if "type" in df.columns:

    print("\n" + "=" * 60)
    print("TYPE DISTRIBUTION")
    print("=" * 60)

    print(df["type"].value_counts())

    plt.figure(figsize=(12, 6))

    sns.countplot(
        data=df,
        y="type",
        order=df["type"].value_counts().index
    )

    plt.title("Ticket Type Distribution")
    plt.xlabel("Count")
    plt.ylabel("Type")
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/raw_type_distribution.png"
    )
    plt.close()


# ==============================
# QUEUE DISTRIBUTION
# ==============================

if "queue" in df.columns:

    print("\n" + "=" * 60)
    print("QUEUE DISTRIBUTION")
    print("=" * 60)

    print(df["queue"].value_counts())

    plt.figure(figsize=(12, 8))

    sns.countplot(
        data=df,
        y="queue",
        order=df["queue"].value_counts().index
    )

    plt.title("Queue Distribution")
    plt.xlabel("Count")
    plt.ylabel("Queue")
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/raw_queue_distribution.png"
    )
    plt.close()


# ==============================
# PRIORITY DISTRIBUTION
# ==============================

if "priority" in df.columns:

    print("\n" + "=" * 60)
    print("PRIORITY DISTRIBUTION")
    print("=" * 60)

    print(df["priority"].value_counts())

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="priority",
        order=df["priority"].value_counts().index
    )

    plt.title("Priority Distribution")
    plt.xlabel("Priority")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/raw_priority_distribution.png"
    )
    plt.close()


# ==============================
# TEXT LENGTH ANALYSIS
# ==============================

print("\n" + "=" * 60)
print("TEXT LENGTH ANALYSIS")
print("=" * 60)

if "subject" in df.columns:
    df["subject"] = df["subject"].fillna("")
    df["subject_length"] = (
        df["subject"]
        .astype(str)
        .str.split()
        .str.len()
    )

if "body" in df.columns:
    df["body"] = df["body"].fillna("")
    df["body_length"] = (
        df["body"]
        .astype(str)
        .str.split()
        .str.len()
    )

if "subject" in df.columns and "body" in df.columns:

    df["full_text"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
    )

    df["text_length"] = (
        df["full_text"]
        .str.split()
        .str.len()
    )

    print(df["text_length"].describe())

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["text_length"],
        bins=50,
        kde=True
    )

    plt.title(
        "Distribution of Ticket Lengths (Words)"
    )
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        f"{REPORT_DIR}/raw_text_lengths.png"
    )
    plt.close()


# ==============================
# SUMMARY
# ==============================

print("\n" + "=" * 60)
print("EDA FINISHED")
print("=" * 60)

print(f"Dataset shape: {df.shape}")

if "language" in df.columns:
    print(
        f"Languages found: "
        f"{df['language'].nunique()}"
    )

print(f"Plots saved in: {REPORT_DIR}")