from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_texts(texts, chunk_size=500, chunk_overlap=100):
    """
    Split a list of long texts into smaller overlapping chunks for embedding.

    Args:
        texts (List[str]): List of complaint narratives (cleaned).
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[dict]: A list of chunks. Each chunk is a dict with:
                    - 'text': The chunk string
                    - 'source_index': Index of original document
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or text.strip() == "":
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source_index": idx
            })

    return all_chunks