from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import numpy as np

def load_embedding_model(model_path=None, model_name='all-MiniLM-L6-v2'):
    """
    Load a SentenceTransformer embedding model from local path (offline) or Hugging Face (online).

    Args:
        model_path (str, optional): Path to local model folder.
        model_name (str): Hugging Face model name (used if model_path is None).

    Returns:
        SentenceTransformer: Loaded embedding model.
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading embedding model from local path: {model_path}")
        return SentenceTransformer(model_path)
    else:
        print(f"Downloading embedding model from Hugging Face: {model_name}")
        return SentenceTransformer(model_name)

def embed_texts(model, texts):
    """
    Generate embeddings for a list of texts.

    Args:
        model (SentenceTransformer): The loaded model.
        texts (List[str]): List of text strings to embed.

    Returns:
        np.ndarray: Embedding vectors (n_texts x embedding_dim).
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings

if __name__ == "__main__":
    # Dynamically build the path to filtered_complaints.csv relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data/processed"))
    csv_path = os.path.join(base_dir, "filtered_complaints.csv")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Select sample texts
    sample_texts = df["Cleaned Narrative"].dropna().tolist()[:10]

    # Load model (try local first)
    local_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/all-MiniLM-L6-v2-local"))
    model = load_embedding_model(model_path=local_model_path)

    print(f"Embedding {len(sample_texts)} sample texts...")
    embeddings = embed_texts(model, sample_texts)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First vector (10 dims): {embeddings[0][:10]}")