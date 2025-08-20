from scripts.rag_pipeline.retriever import ComplaintRetriever
from scripts.rag_pipeline.generator import generate_answer


class ComplaintRAGPipeline:
    def __init__(
        self,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path="vector_store/chromadb"
    ):
        # Initialize the retriever with embedding model and vector DB path
        self.retriever = ComplaintRetriever(
            embedding_model_name=embedding_model_name,
            vector_store_path=vector_store_path
        )

    def ask(self, question, top_k=5):
        # Basic ask method (no filtering)
        retrieved_chunks = self.retriever.retrieve(query_text=question, top_k=top_k)
        context = "\n\n".join(retrieved_chunks)
        answer = generate_answer(context=context, question=question)
        return answer

    def ask_with_sources(self, question: str, product: str = "All", top_k: int = 5):
        """
        Enhanced method: supports product filtering and returns both answer + source chunks
        """
        filter_dict = {} if product == "All" else {"product": product}
        retrieved_chunks = self.retriever.retrieve(query_text=question, top_k=top_k, filters=filter_dict)

        context = "\n\n".join(retrieved_chunks)
        answer = generate_answer(context=context, question=question)
        return answer, retrieved_chunks  # Returning both


if __name__ == "__main__":
    pipeline = ComplaintRAGPipeline()

    user_question = "Why do customers complain about Buy Now Pay Later?"
    product = "Buy Now, Pay Later (BNPL)"

    answer, sources = pipeline.ask_with_sources(user_question, product=product)
    print("Question:", user_question)
    print("\nAnswer:\n", answer)
    print("\nSources:\n", sources[:2])  # Print first 2 chunks
