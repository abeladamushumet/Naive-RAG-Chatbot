# Intelligent Complaint Analysis for Financial Services

This project implements a Retrieval-Augmented Generation (RAG) chatbot to help internal teams at **CrediTrust Financial** understand and respond to customer complaints across five key financial products:

- Credit Cards  
- Personal Loans  
- Buy Now, Pay Later (BNPL)  
- Savings Accounts  
- Money Transfers  

---

## Business Objective

CrediTrust receives thousands of customer complaints monthly. Product Managers like Asha waste hours manually reading feedback. This tool lets them:

- Ask plain-English questions  
- Get synthesized, real-time insights  
- Proactively identify major complaint trends  

---

## How It Works

1. **Complaints Data** → Cleaned and chunked  
2. **Embeddings** → Created using `all-MiniLM-L6-v2`  
3. **Vector DB** → Stored in ChromaDB  
4. **RAG Pipeline** → Retrieves top chunks + generates answer using an LLM  
5. **Gradio App** → Interactive chatbot interface for non-technical users

---

## Tech Stack

- Python
- LangChain
- SentenceTransformers
- ChromaDB
- Hugging Face (LLM)
- Gradio

---

## Project Structure

Intelligent-Complaint-Analysis/
├── App/
│ ├── app.py
│ └── ui_utils.py
├── Data/
│ ├── raw/complaints.csv
│ └── processed/filtered_complaints.csv
├── images/ 
│ └──chatbot_ui.png
├── notebooks/
│ ├── eda_preprocessing.ipynb
│ └── rag_evaluation.ipynb
├── scripts/
│ ├── data_processing/
│ ├── embedding_pipeline/
│ └── rag_pipeline/
├── vector_store/
│ └── chromadb/
├── reports/
│ ├── interim_report.md
│ ├── final_report.md
│ └── evaluation_table.md 
├── README.md
├── requirements.txt
├── LICENCE  
└── .gitignore 


---

## Setup Instructions

```bash
# Clone the repo
git clone https://github.com/abeladamushumet/intelligent-complaint-analysis.git
cd intelligent-complaint-analysis


# Install dependencies
pip install -r requirements.txt