from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():

    """
    Free HuggingFace embedding model returns a 384-dimensional vector for each input text. 
    The model is "sentence-transformers/all-MiniLM-L6-v2". 
    """

    embeddings =  HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

if __name__ == "__main__":
    
    model = get_embedding_model()

    text = "I am a Machine Learning Engineer with Python experience."
    vector = model.embed_query(text)

    print(f" Text: {text}")
    print(f" Vector length (dimension): {len(vector)}")
    print(f" 1st 5 elements: {vector[:5]}")

