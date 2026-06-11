from src.loader import load_pdf, split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vectorstore
from src.rag import answer_with_memory

# Setup
docs = load_pdf("data/Md_Reja_E_Rabbi_Tonmoy.pdf")  # Your file name
chunks = split_documents(docs)
embedding_model = get_embedding_model()
vectorstore = create_vectorstore(chunks, embedding_model)

# Memory 
chat_history = []


q1 = "What is the candidate's major?"
a1, _ = answer_with_memory(vectorstore, q1, chat_history)
print(f"Q1: {q1}\nA1: {a1}\n")
chat_history.append({"question": q1, "answer": a1})

q2 = "How many years of experience does he have?"
a2, _ = answer_with_memory(vectorstore, q2, chat_history)
print(f"Q2: {q2}\nA2: {a2}\n")
chat_history.append({"question": q2, "answer": a2})

q3 = "Is he suetbl for an ML role?"
a3, _ = answer_with_memory(vectorstore, q3, chat_history)
print(f"Q3: {q3}\nA3: {a3}")

q4 = "who r u?"
a4, _ = answer_with_memory(vectorstore, q4, chat_history)
print(f"Q4: {q4}\nA4: {a4}")