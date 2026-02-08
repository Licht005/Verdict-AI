import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from prompt import VERDICT_PROMPT 

load_dotenv()

class VerdictRAG:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_path = os.path.join(base_dir, "Doc", "Ghana Constitution.pdf")
        
        # This initializes the Bi-Encoder for vector space creation
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        self.qa_chain = self._setup_chain()

    def _setup_chain(self):
        loader = PDFPlumberLoader(self.pdf_path)
        docs = loader.load()
        
        # Better splitting for legal documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150,
            separators=["\nArticle", "\nCHAPTER", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(docs)
        
        # Build the initial Vector Space
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # STAGE 1: Broad Retrieval (Grab top 20 potential chunks)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        
        # STAGE 2: Re-ranking (Filter down to the top 3 best chunks)
        compressor = FlashrankRerank(top_n=3)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=compression_retriever,
            chain_type_kwargs={"prompt": VERDICT_PROMPT}
        )

    def ask(self, query: str):
        response = self.qa_chain.invoke(query)
        return response["result"]