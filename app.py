from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

from src.loader import load_pdf, split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vectorstore, load_existing_vectorstore
from src.rag import answer_with_memory



# 1. Create app 
app = FastAPI(title="Smart Doc QA", description="RAG-based document Q&A API")

# 2. Load embedding model once at startup (avoid loading on every request)
embedding_model = get_embedding_model()

# 3. Chat history 
chat_history = []


# 4. Request model for question answering
class QuestionRequest(BaseModel):
    question: str


# 5. Health check endpoint
@app.get("/")
def home():
    return {"message": "Smart Doc QA API is running"}


# 6. PDF upload + index করার endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # PDF টা data folder এ save করি
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load → chunk → embed → store
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    create_vectorstore(chunks, embedding_model)

    # নতুন document এলে history clear করি
    chat_history.clear()

    return {
        "message": f"'{file.filename}' uploaded and indexed successfully",
        "pages": len(docs),
        "chunks": len(chunks),
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    vectorstore = load_existing_vectorstore(embedding_model)

    answer, sources = answer_with_memory(
        vectorstore, request.question, chat_history
    )

    chat_history.append({"question": request.question, "answer": answer})

    # প্রতিটা source chunk এর text + page সংগ্রহ করি
    source_snippets = []
    for doc in sources:
        source_snippets.append({
            "page": doc.metadata.get("page", "N/A"),
            "text": doc.page_content[:300],  # প্রথম 300 অক্ষর
        })

    return {
        "question": request.question,
        "answer": answer,
        "sources_used": len(sources),
        "sources": source_snippets,  
    }