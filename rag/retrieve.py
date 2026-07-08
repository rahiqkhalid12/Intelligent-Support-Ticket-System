import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# Paths
# ==========================================================
VECTOR_DIR = "models/rag"

INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
DOCS_PATH = os.path.join(VECTOR_DIR, "documents.pkl")

# ==========================================================
# Load SBERT model
# Must match the model used in embedding.py
# ==========================================================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# Load FAISS index
# ==========================================================
index = faiss.read_index(INDEX_PATH)

# ==========================================================
# Load ticket metadata
# ==========================================================
with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)


def retrieve(query, top_k=5):
    """
    Retrieve the top-k most similar support tickets.

    Args:
        query (str): User's support ticket.
        top_k (int): Number of similar tickets to return.

    Returns:
        list: Retrieved tickets with similarity scores.
    """

    # Encode and normalize the query
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Search FAISS
    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        doc = documents[idx].copy()

        doc["similarity_score"] = float(score)

        results.append(doc)

    return results


# ==========================================================
# Example
# ==========================================================
if __name__ == "__main__":

    query = """
    I cannot log into my account.
    It says my password is incorrect even after resetting it.
    """

    retrieved_docs = retrieve(query, top_k=5)

    print("\nTop Retrieved Tickets\n")

    for i, doc in enumerate(retrieved_docs, start=1):

        print("=" * 70)
        print(f"Result #{i}")
        print("=" * 70)

        print(f"Similarity Score : {doc['similarity_score']:.4f}")
        print(f"Type             : {doc['type']}")
        print(f"Queue            : {doc['queue']}")
        print(f"Priority         : {doc['priority']}")

        print("\nCustomer Ticket:")
        print(doc["text"])
        if "answer" in doc:
            print("\nSupport Answer:")
            print(doc["answer"])

        print()