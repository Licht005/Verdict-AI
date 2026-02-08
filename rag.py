import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Switched
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

class VerdictRAG:
    def __init__(self):
        # Use absolute path to avoid "File Not Found" errors
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pdf_path = os.path.join(base_dir, "Doc", "Ghana Constitution.pdf")
        
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
        
        # IMPROVEMENT: Use Recursive splitter with specific separators for legal text
        # This prevents "Article 50" from being separated from its content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\nArticle", "\nCHAPTER", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # IMPROVEMENT: Prompt that explicitly forces a distinction between Chapter and Article
        template = """You are Verdict AI, a legal assistant for Ghanaian Law developed by Lucas.
        
        CRITICAL INSTRUCTION: The user may ask for an 'Article'. Do not provide 'Chapter' information unless specifically asked for a Chapter.
        Example: If asked for Article 6, look for 'Article 6: Citizenship of Ghana'. 
        Do not return 'Chapter 6: Directive Principles'.

        Context: {context}
        Question: {question}
        
        Answer (Always cite the Article and Clause):"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            # k=10 is good for coverage, but we use 'similarity' to focus on the specific Article text
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, query: str):
        response = self.qa_chain.invoke(query)
        return response["result"]