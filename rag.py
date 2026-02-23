import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
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
        self.vectorstore = self._build_vectorstore()
        self.all_chunks = self._load_all_chunks()
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

    def _load_all_chunks(self):
        """Load all chunks from the PDF for direct article lookup."""
        loader = PDFPlumberLoader(self.pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\nCHAPTER", "\n\n", "\n", " "]
        )
        return splitter.split_documents(docs)

    def _setup_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": VERDICT_PROMPT}
        )

    def _find_article_chunks(self, article_num: str) -> list[Document]:
        """Directly scan chunks for ones that start with or contain the article number."""
        pattern = re.compile(rf'(^|\n){re.escape(article_num)}\.\n', re.MULTILINE)
        matches = []
        for chunk in self.all_chunks:
            if pattern.search(chunk.page_content):
                matches.append(chunk)
        return matches[:4]  # return up to 4 matching chunks

    def ask(self, query: str) -> str:
        # Check if query is asking about a specific article
        match = re.search(r'article\s+(\d+)', query.lower())

        if match:
            article_num = match.group(1)
            direct_chunks = self._find_article_chunks(article_num)

            if direct_chunks:
                # Build context from direct matches and send to LLM
                context = "\n\n".join([c.page_content for c in direct_chunks])
                final_prompt = VERDICT_PROMPT.format(context=context, question=query)
                response = self.llm.invoke(final_prompt)
                return response.content

        # Fallback to normal RAG for non-article queries
        response = self.qa_chain.invoke(query)
        return response["result"]