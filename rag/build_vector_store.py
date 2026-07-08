import os
import pickle
import faiss
import numpy as np
import pandas as pd

# ==========================================================
# Paths
# ==========================================================
TRAIN_PATH = "data/processed/train.csv"
EMBEDDINGS_PATH = "data/processed/embeddings/sbert_train.npy"

OUTPUT_DIR = "models/rag"
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")
DOCS_PATH = os.path.join(OUTPUT_DIR, "documents.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load training data
# ==========================================================
train_df = pd.read_csv(TRAIN_PATH)

# ==========================================================
# Load SBERT embeddings
# ==========================================================
embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

print(f"Loaded embeddings: {embeddings.shape}")

# Safety check
assert len(train_df) == len(embeddings), (
    f"Mismatch! train_df has {len(train_df)} rows "
    f"but embeddings has {len(embeddings)} vectors."
)

# ==========================================================
# Build FAISS index
# ==========================================================
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
print(f"FAISS index saved to: {INDEX_PATH}")

# ==========================================================
# Save ticket metadata
# ==========================================================
documents = []

for _, row in train_df.iterrows():
    doc = {
        "subject": str(row.get("subject", "")),
        "text": str(row.get("text", "")),
        "answer": str(row.get("answer", "")),
        "type": str(row.get("type", "")),
        "queue": str(row.get("queue", "")),
        "priority": str(row.get("priority", "")),
        "tag_1": str(row.get("tag_1", "")),
        "tag_2": str(row.get("tag_2", "")),
        "tag_3": str(row.get("tag_3", "")),
        "tag_4": str(row.get("tag_4", "")),
    }

    documents.append(doc)

with open(DOCS_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"Documents saved to: {DOCS_PATH}")
print(f"Indexed {len(documents)} tickets successfully.")