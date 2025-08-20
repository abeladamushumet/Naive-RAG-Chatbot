import sys
import os

# Add the project root to sys.path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from scripts.rag_pipeline.pipeline import ComplaintRAGPipeline
from Apps.ui_utils import format_sources, style_response  # Needed for source display

# Initialize the RAG pipeline once at startup
pipeline = ComplaintRAGPipeline()

# Define product categories
product_types = [
    "All",  
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers"
]

def answer_question(user_question: str, product: str) -> str:
    """Generate an answer for the user question using the RAG pipeline with optional product filter."""
    if not user_question.strip():
        return "Please enter a question."
    try:
        answer, sources = pipeline.ask_with_sources(user_question, product=product)
        formatted_sources = format_sources(sources)
        return style_response(answer, formatted_sources)
    except Exception as e:
        return f"Error: {e}"

# Build Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")
    gr.Markdown("Ask questions about customer complaints across financial products.")

    user_input = gr.Textbox(
        label="Your Question",
        lines=2,
        placeholder="E.g., Why are customers unhappy with BNPL?"
    )

    product_dropdown = gr.Dropdown(
        label="Product Type",
        choices=product_types,
        value="All"
    )

    output_text = gr.Textbox(label="Answer with Sources", lines=10)
    submit_btn = gr.Button("Ask")
    submit_btn.click(fn=answer_question, inputs=[user_input, product_dropdown], outputs=output_text)

    clear_btn = gr.Button("Clear")
    clear_btn.click(fn=lambda: "", inputs=None, outputs=user_input)

if __name__ == "__main__":
    demo.launch()