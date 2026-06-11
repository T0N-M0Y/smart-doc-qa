from src.loader import load_pdf, split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vectorstore
from src.rag import answer_question


# Step 1: Load
docs = load_pdf("data/Md_Reja_E_Rabbi_Tonmoy.pdf")  # Your file name

# Step 2: Chunk
chunks = split_documents(docs)

# Step 3: Embedding model
embedding_model = get_embedding_model()

# Step 4: Store
vectorstore = create_vectorstore(chunks, embedding_model)

# Test: A question to find similar chunks
question = "What is the major of the candidate? And what is his experience? And what is the suitable"  # Your question here
answer, sources = answer_question(vectorstore, question)

print(f"\n{'='*50}")
print(f"{question}")
print(f"{'='*50}")
print(f"Ans: {answer}")
