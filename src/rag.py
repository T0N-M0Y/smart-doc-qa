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


def answer_with_memory(vectorstore, question, chat_history, k=3):
    """
    Conversation memory with RAG; Chat History = Previous question-answer pairs. 
    """
    
    summary_keywords = ["summarize", "summary", "tell me about", "overview", "what is this"]
    if any(kw in question.lower() for kw in summary_keywords):
        k = 6 

    # 1. Retrieve
    
    retrieved_docs = vectorstore.similarity_search(
        question,
        k=k
    )
    context = format_docs(retrieved_docs)

    
    # 2. History conversion to text
    
    history_text = ""

    for turn in chat_history:
        history_text += f"User: {turn['question']}\n"
        history_text += f"Assistant: {turn['answer']}\n"

    # 3. Prompt এ history + context both push 
    
    template = """You are a friendly assistant for a document Q&A system. 
                    The "Context" below is content from a document the user uploaded.

                    Guidelines:
                    - If the user greets you (hi, hello) or asks who you are, respond warmly 
                      and briefly explain you can answer questions about their uploaded document.
                    - When the user refers to "the file", "the document", "this", or asks to 
                      "summarize", they mean the uploaded document in the Context.
                    - For document questions, answer based ONLY on the Context.
                    - If a document question's answer isn't in the Context, say 
                      "I couldn't find that in the document."

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
        "question": question,
    })

    return answer, retrieved_docs