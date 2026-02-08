import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_xai import ChatXAI  
from langchain_classic.chains import RetrievalQA  

# 1. Load Environment Variables
load_dotenv()

def run_rag_pipeline(pdf_path, user_query):
    # 2. Load the PDF
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    # 3. Semantic Chunking
    # We use a small, local embedding model to find semantic "breakpoints" in the text
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embeddings)
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} semantic chunks.")

    # 4. Create Vector Store (Local FAISS index)
    vector_db = FAISS.from_documents(chunks, embeddings)

    # 5. Initialize Grok (xAI)
    llm = ChatXAI(
        model="grok-beta", 
        xai_api_key=os.getenv("XAI_API_KEY")
    )

    # 6. Create Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )

    # 7. Run Query
    response = qa_chain.invoke(user_query)
    return response["result"]

# Example Usage
if __name__ == "__main__":
    pdf = "Ghana Constitution.pdf"
    query = "What does the constitution say about the freedom of the media?"
    
    answer = run_rag_pipeline(pdf, query)
    print(f"\n--- Grok's Answer ---\n{answer}")