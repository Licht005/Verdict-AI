import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# --- IMPORT YOUR NEW PROMPT ---
from prompt import VERDICT_PROMPT 

load_dotenv()

class VerdictRAG:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_path = os.path.join(base_dir, "Doc", "Ghana Constitution.pdf")
        
        # YES: This line creates the "Embedding Model"
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
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\nArticle", "\nCHAPTER", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(docs)
        
        # YES: This line builds the "Vector Space" (FAISS) using the embeddings
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            # Use the imported prompt here
            chain_type_kwargs={"prompt": VERDICT_PROMPT}
        )

    def ask(self, query: str):
        response = self.qa_chain.invoke(query)
        return response["result"]