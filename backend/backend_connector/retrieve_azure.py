import os
import numpy as np
import pandas as pd
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer

# ================================================
# Configuration
# ================================================
AZURE_SEARCH_ENDPOINT = "https://supportticketsearchaya.search.windows.net"
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "support-tickets"

# ================================================
# Step 1: Create Azure AI Search Service Index
# ================================================
def create_vector_index():
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="subject", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchableField(name="answer", type=SearchFieldDataType.String),
        SimpleField(name="queue", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="priority", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=384,
            vector_search_profile_name="my-vector-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")],
        profiles=[VectorSearchProfile(
            name="my-vector-profile",
            algorithm_configuration_name="my-hnsw"
        )]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    result = index_client.create_or_update_index(index)
    print(f"✅ Index '{result.name}' created successfully!")
    return result


# ================================================
# Step 2: Upload Embeddings to Azure AI Search
# ================================================
def upload_embeddings(train_csv_path, embeddings_npy_path):
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    train_df = pd.read_csv(train_csv_path)
    embeddings = np.load(embeddings_npy_path).astype(np.float32)

    print(f"Loaded {len(train_df)} tickets, embeddings shape: {embeddings.shape}")
    assert len(train_df) == len(embeddings), "Mismatch between tickets and embeddings!"

    BATCH_SIZE = 100
    total_uploaded = 0

    for i in range(0, len(train_df), BATCH_SIZE):
        batch_df = train_df.iloc[i:i+BATCH_SIZE]
        batch_emb = embeddings[i:i+BATCH_SIZE]

        documents = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            doc = {
                "id": str(i + j),
                "subject": str(row.get("subject", "")),
                "text": str(row.get("text", "")),
                "answer": str(row.get("answer", "")),
                "queue": str(row.get("queue", "")),
                "type": str(row.get("type", "")),
                "priority": str(row.get("priority", "")),
                "embedding": batch_emb[j].tolist()
            }
            documents.append(doc)

        search_client.upload_documents(documents)
        total_uploaded += len(documents)
        print(f"Uploaded {total_uploaded}/{len(train_df)} tickets...")

    print(f"✅ All {total_uploaded} tickets uploaded to Azure AI Search!")


# ================================================
# Step 3: Initialize Search Client + SBERT Model
# ================================================
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


# ================================================
# Step 4: Retrieve Similar Tickets (Replaces FAISS)
# ================================================
def retrieve_similar_tickets(query_text, top_k=3):
    """
    Retrieves similar support tickets from Azure AI Search
    using SBERT vector embeddings. Replaces local FAISS search.

    Args:
        query_text (str): The input ticket text to search for
        top_k (int): Number of similar tickets to return

    Returns:
        list of dicts: Similar tickets with metadata and similarity scores
    """
    query_embedding = sbert_model.encode(query_text).tolist()

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "subject", "text", "answer", "queue", "type", "priority"]
    )

    similar_tickets = []
    for r in results:
        similar_tickets.append({
            "id": r["id"],
            "subject": r["subject"],
            "text": r["text"],
            "answer": r["answer"],
            "queue": r["queue"],
            "type": r["type"],
            "priority": r["priority"],
            "score": r["@search.score"]
        })

    return similar_tickets


# ================================================
# Step 5: Test Retrieval
# ================================================
def test_retrieval():
    test_queries = [
        "I cannot login to my account, my password is not working",
        "My payment was charged but order was not placed",
        "I need to cancel my subscription"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("=" * 60)
        results = retrieve_similar_tickets(query, top_k=3)
        for i, ticket in enumerate(results):
            print(f"  #{i+1} Subject : {ticket['subject']}")
            print(f"       Type    : {ticket['type']}")
            print(f"       Queue   : {ticket['queue']}")
            print(f"       Priority: {ticket['priority']}")
            print(f"       Score   : {ticket['score']:.4f}")


# ================================================
# Main
# ================================================
if __name__ == "__main__":
    # Uncomment to rebuild index from scratch:
    # create_vector_index()
    # upload_embeddings("data/processed/train.csv",
    #                   "data/processed/embeddings/sbert_train.npy")

    test_retrieval()
