import os
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not set!")

print("Token loaded")

client = InferenceClient(token=hf_token)

# Use a public chat-capable model
chat_model = "mistralai/Mistral-7B-Instruct-v0.2"

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Context:\n{context}\n\n"
        "User: {question}\n"
        "Assistant:"
    )
)

def generate_answer(context: str, question: str) -> str:
    prompt = prompt_template.format(context=context, question=question)
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return resp.choices[0].message["content"]

if __name__ == "__main__":
    ctx = (
        "The Consumer Financial Protection Bureau (CFPB) collects complaint data "
        "related to financial products and services in the United States."
    )
    q = "What is the role of the CFPB?"
    print("Answer:", generate_answer(ctx, q))