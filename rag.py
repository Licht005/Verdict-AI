import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from prompt import VERDICT_PROMPT

load_dotenv()

FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")

# Maps article numbers to their topic for better retrieval
ARTICLE_TOPICS = {
    "1": "sovereignty supreme law constitution",
    "2": "enforcement constitution court",
    "3": "defence constitution coup",
    "4": "territories regions Ghana",
    "5": "regions alteration",
    "6": "citizenship born Ghana parents grandparents",
    "7": "woman marriage citizenship registration",
    "8": "renunciation citizenship",
    "9": "national symbols flag anthem",
    "10": "national symbols coat of arms seal",
}

def rewrite_query(query: str) -> str:
    """If the query is about a specific article, enrich it with known topic keywords."""
    match = re.search(r'article\s+(\d+)', query.lower())
    if match:
        num = match.group(1)
        if num in ARTICLE_TOPICS:
            return f"Article {num} {ARTICLE_TOPICS[num]}"
        return f"Article {num} section {num}."
    return query

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
        self.vectorstore = self._build_vectorstore()
        self.qa_chain = self._setup_chain()

    def _build_vectorstore(self):
        if os.path.exists(FAISS_INDEX_PATH):
            return FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)

        loader = PDFPlumberLoader(self.pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\nCHAPTER", "\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore

    def _setup_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": VERDICT_PROMPT}
        )

    def ask(self, query: str) -> str:
        enriched_query = rewrite_query(query)
        response = self.qa_chain.invoke(enriched_query)
        return response["result"]