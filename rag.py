import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from prompt import VERDICT_PROMPT

load_dotenv()

FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")

class VerdictRAG:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_path = os.path.join(base_dir, "Doc", "Ghana Constitution.pdf")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        self.qa_chain = self._setup_chain()

    def _build_vectorstore(self):
        # Cache FAISS index to avoid rebuilding on every startup.
        if os.path.exists(FAISS_INDEX_PATH):
            return FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
        
        loader = PDFPlumberLoader(self.pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\nArticle", "\nCHAPTER", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore

def _setup_chain(self):
    vectorstore = self._build_vectorstore()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})  # cast wider net
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=FlashrankRerank(top_n=6),  # keep more after rerank
        base_retriever=base_retriever
    )
    return RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": VERDICT_PROMPT}
    )

    def ask(self, query: str) -> str:
        response = self.qa_chain.invoke(query)
        return response["result"]