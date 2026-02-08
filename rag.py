import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Changed from ChatXAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Load Environment Variables
load_dotenv()

def run_verdict_rag():
    # 2. Load PDF
    loader = PDFPlumberLoader("Ghana Constitution.pdf")
    docs = loader.load()

    # 3. Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Semantic Chunking
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(docs)

    # 5. Create Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 6. Initialize GROQ (Using your gsk_ key)
    llm = ChatGroq(
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"), # Ensure this matches your .env
        model_name="llama-3.3-70b-versatile"     # High-performance model on Groq
    )

    # 7. Setup RAG Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # 8. Run
    query = "What does the constitution say about the freedom of the media?"
    print(f"\nVerdict Assistant: {qa.invoke(query)['result']}")

if __name__ == "__main__":
    run_verdict_rag()