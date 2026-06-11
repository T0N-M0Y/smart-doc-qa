from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(file_path: str):
    """
    Load a pdf file and return each page as a Document object.
    Returns a list of Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f" Loaded {len(documents)} pages from {file_path}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Documents are split into smaller chunks. chunk_size is the maximum length of each chunk and chunk_
    overlap is the amount of overlap between consecutive chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f" Split into {len(chunks)} chunks")
    return chunks


# Test for the function
if __name__ == "__main__":
    docs = load_pdf("/home/playground/Desktop/tonmoy/LLM/Rag Project/smart-doc-qa/data/Md_Reja_E_Rabbi_Tonmoy.pdf")
    chunks = split_documents(docs) 
    print(chunks[0].page_content)





