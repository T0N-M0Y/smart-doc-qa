# smart-doc-qa

A RAG-based document Q&A system. Upload any PDF and ask questions in natural language — get accurate, source-grounded answers.

##  Features
- Upload any PDF document
- Ask questions in natural language
- Get answers grounded in document content (no hallucination)
- Powered by semantic search + LLM

## Tech Stack
- **LangChain** — RAG orchestration
- **ChromaDB** — vector database
- **HuggingFace** — embeddings (sentence-transformers)
- **Groq** — LLM inference (Llama 3.1)
- **FastAPI** — REST API (coming soon)

##  Status
Currently in active development.

##  Setup
\`\`\`bash
git clone https://github.com/T0N-M0Y/smart-doc-qa
cd docu-query
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## 👤 Author
**Md. Reja E Rabbi Tonmoy** — ML Engineer
[LinkedIn](https://www.linkedin.com/in/tonmoy-md-reja-e-rabbi/) • [GitHub](https://github.com/T0N-M0Y)