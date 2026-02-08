import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


load_dotenv()

class VerdictRAG:
    def __init__(self):
        self.pdf_path = os.path.join("Doc", "Ghana Constitution.pdf")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        self.qa_chain = self._setup_chain()

    def _setup_chain(self):
        # Load and process
        loader = PDFPlumberLoader(self.pdf_path)
        docs = loader.load()
        
        text_splitter = SemanticChunker(self.embeddings)
        chunks = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Define the Prompt Template
        template = """You are Verdict AI, a legal assistant for Ghanaian Law.
        Developed by Lucas
        Use the context to answer. Always cite Articles.
        Context: {context}
        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, query: str):
        response = self.qa_chain.invoke(query)
        return response["result"]