import os
import pandas as pd
import numpy as np
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class VectorStoreChroma:
    def __init__(self, persist_directory="vector_store/chromadb", embedding_function=None):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = PersistentClient(path=self.persist_directory)

        if "complaints_collection" in [col.name for col in self.client.list_collections()]:
            self.collection = self.client.get_collection("complaints_collection")
        else:
            self.collection = self.client.create_collection(
                name="complaints_collection",
                embedding_function=embedding_function
            )

    def add_documents(self, texts, metadatas=None, ids=None, embeddings=None):
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def query(self, query_text, n_results=5):
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def persist(self):
        # Persistence is automatic with PersistentClient
        print("Persistence is automatic with PersistentClient. No need to call persist().")

    def reset(self):
        self.collection.delete(where={})


def embed_texts(model, texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(
            batch,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings.append(emb)
    return np.vstack(embeddings)


def batch_add_documents(store, texts, metadatas, ids, embeddings, batch_size=500):
    total = len(texts)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        print(f"Adding batch {start_idx} to {end_idx} of {total}...")
        store.add_documents(
            texts[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
            ids=ids[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx]
        )


if __name__ == "__main__":
    from scripts.embedding_pipeline.embedding import load_embedding_model, embed_texts as custom_embed
    from scripts.embedding_pipeline.chunking import chunk_texts

    # Load and sample data
    df = pd.read_csv("Data/processed/filtered_complaints.csv")
    df = df.sample(n=1000, random_state=42)

    # Chunk narratives
    texts = df["Cleaned Narrative"].dropna().tolist()
    chunks = chunk_texts(texts)

    chunk_texts_only = [chunk["text"] for chunk in chunks]
    metadatas = [{"source_index": chunk["source_index"]} for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]

    # Load embedding model 
    try:
        model = load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(" Model loading failed:", e)
        exit()

    # Generate embeddings
    try:
        print(f"Embedding {len(chunk_texts_only)} chunks...")
        embeddings = embed_texts(model, chunk_texts_only, batch_size=100)
        print(f" Embeddings created. Shape: {embeddings.shape}")
    except Exception as e:
        print("Embedding failed:", e)
        exit()

    # Store vectors in Chroma
    store = VectorStoreChroma(
        persist_directory="vector_store/chromadb",
        embedding_function=None
    )

    batch_add_documents(store, chunk_texts_only, metadatas, ids, embeddings, batch_size=500)

    # No error: just a notification now
    store.persist()
    print("Vector store saved.")