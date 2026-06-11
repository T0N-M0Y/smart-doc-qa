from langchain_chroma import Chroma

def create_vectorstore(chunks, embedding_model, persist_dir="chroma_db"):
    """
     Embeddeded the chunks and stores them in ChromaDB. 
     The embedded vectors are stored on disk in the specified persist_dir,
     so that they can be loaded later without re-embedding.
    """

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    print(f" Stored {len(chunks)} chunks in ChromaDB at '{persist_dir}'")
    return vectorstore


def load_existing_vectorstore(embedding_model, persist_dir="chroma_db"):
    
    """
    Load the existing ChromaDB without re-embedding.
    """
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )
    print(f" Loaded existing ChromaDB from '{persist_dir}'")
    return vectorstore