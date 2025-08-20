def format_sources(chunks, max_chunks=3):
    """
    Format retrieved chunks into a readable string for display.
    """
    if not chunks:
        return "\n\n_No supporting sources found._"

    formatted = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        formatted.append(f"**Source {i+1}:**\n{chunk.strip()}")
    return "\n\n".join(formatted)


def style_response(answer, sources):
    """
    Combine the generated answer with formatted sources.
    """
    return f"### Answer:\n{answer.strip()}\n\n---\n### Sources:\n{sources}"