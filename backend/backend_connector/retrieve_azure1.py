import os
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer

AZURE_SEARCH_ENDPOINT = 'https://supportticketsearchaya.search.windows.net'
AZURE_SEARCH_KEY = os.environ.get('AZURE_SEARCH_KEY', 'YOUR_KEY_HERE')
INDEX_NAME = 'support-tickets'

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_similar_tickets(query_text, top_k=3):
    query_embedding = sbert_model.encode(query_text).tolist()
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields='embedding'
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=['id', 'subject', 'text', 'answer', 'queue', 'type', 'priority']
    )
    similar_tickets = []
    for r in results:
        similar_tickets.append({
            'id': r['id'],
            'subject': r['subject'],
            'text': r['text'],
            'answer': r['answer'],
            'queue': r['queue'],
            'type': r['type'],
            'priority': r['priority'],
            'score': r['@search.score']
        })
    return similar_tickets
