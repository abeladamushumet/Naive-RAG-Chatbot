import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ComplaintRetriever:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", vector_store_path="vector_store/chromadb"):
        # Load sentence embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Connect to ChromaDB persistent store
        self.client = PersistentClient(path=vector_store_path)
        self.collection = self.client.get_collection("complaints_collection")

    def embed_query(self, query_text: str):
        return self.embedding_model.encode(query_text, normalize_embeddings=True).tolist()

    def retrieve(self, query_text: str, top_k: int = 5, filters: dict = None):
        """
        Retrieve top_k most relevant chunks for the query.
        If filters are provided (e.g., {'product': 'Buy Now, Pay Later (BNPL)'}), use them.
        """
        filters = filters or {}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filters  # filters by product or other metadata
        )

        return results["documents"][0]  # List of top_k matching chunks