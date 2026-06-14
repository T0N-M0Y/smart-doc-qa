from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
   
    """Groq LLM (Llama 3.1) returns a concise answer based on the provided context and question."""
   
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,  # never want randomness in RAG answers, want dirct answers based on retrieved context
    )


def build_prompt():
    """Create a prompt template that instructs the LLM to answer the question based on the retrieved context."""
    
    template = """You are a helpful assistant. Answer the question 
                   based ONLY on the context below. If the answer is not in the 
                   context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    return ChatPromptTemplate.from_template(template)


def format_docs(docs):
    """the context is the retrieved documents (chunks).
     We format them as a single string to feed into the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(vectorstore, question, k=3):
    """
    Full RAG pipeline:
    Retrieve → Augment (prompt) → Generate
    """
    # 1. Retrieve: Find relevant chunks related to the question
    retrieved_docs = vectorstore.similarity_search(
        question, 
        k=k
    )
    
    context = format_docs(retrieved_docs)

    # 2. Augment: content push to prompt 
    prompt = build_prompt()

    # 3. Generate: LLM gets the prompt and generates an answer
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved_docs

def rewrite_query(question, chat_history):
    """
    User এর প্রশ্নকে retrieval-friendly করে rewrite করে।
    - Follow-up প্রশ্নে আগের context যোগ করে
    - Vague শব্দ expand করে
    """
    # History না থাকলে আর প্রশ্ন বড় হলে — rewrite এর দরকার কম
    history_text = ""
    for turn in chat_history[-3:]:  # শুধু শেষ ৩ turn (token বাঁচাতে)
        history_text += f"User: {turn['question']}\n"
        history_text += f"Assistant: {turn['answer'][:150]}\n"

    rewrite_prompt = """Given the conversation history and a follow-up 
question, rewrite the question into a standalone search query that will 
retrieve relevant information from a document.

Rules:
- Make it self-contained (resolve "he", "it", "that" using history)
- Expand vague terms (e.g., "specialization" → "skills expertise field")
- Keep it concise (one line)
- Output ONLY the rewritten query, nothing else

Conversation History:
{history}

Follow-up Question: {question}

Standalone Search Query:"""

    prompt = ChatPromptTemplate.from_template(rewrite_prompt)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    rewritten = chain.invoke({
        "history": history_text if history_text else "(none)",
        "question": question,
    })
    return rewritten.strip()


def answer_with_memory(vectorstore, question, chat_history, k=6):
    """
    Query rewriting সহ RAG।
    """
    # ১. Query Rewriting — retrieve করার আগে প্রশ্ন উন্নত করি
    search_query = rewrite_query(question, chat_history)

    # ২. Retrieve (rewritten query দিয়ে)
    retrieved_docs = vectorstore.similarity_search(search_query, k=k)
    context = format_docs(retrieved_docs)

    # ৩. History কে text এ রূপান্তর
    history_text = ""
    for turn in chat_history:
        history_text += f"User: {turn['question']}\n"
        history_text += f"Assistant: {turn['answer']}\n"

    # ৪. Answer generation (মূল প্রশ্ন দিয়ে, rewritten না)
    template = """You are a friendly assistant for a document Q&A system. 
The "Context" below is content from a document the user uploaded.

Guidelines:
- If the user greets you (hi, hello) or asks who you are, respond warmly 
  and briefly explain you answer questions about their uploaded document.

- When the user refers to "the file", "the document", "this", or asks to 
  "summarize", they mean the uploaded document in the Context.

- For document questions, answer based ONLY on the Context.

- For document questions, answer based on the Context. Look carefully 
  through ALL provided context sections before concluding information 
  isn't there. Only say "I couldn't find that in the document" if the 
  information is genuinely absent.

- Answer in the SAME language as the document content in the Context. 
  If the Context is in English, answer in English. If the document 
  question is in a different language, you may still answer in the 
  document's language.
  

Conversation History:
{history}

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question,  # মূল প্রশ্ন — natural উত্তরের জন্য
    })

    return answer, retrieved_docs