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
import json

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
CHUNKS_CACHE_PATH = os.path.join(BASE_DIR, "chunks_cache.json")

class VerdictRAG:
    def __init__(self):
        self.pdf_path = os.path.join(BASE_DIR, "Doc", "Ghana Constitution.pdf")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        self.chunks, self.vectorstore = self._build()
        self.qa_chain = self._setup_chain()

    def _build(self):
        # Load chunks from cache if available
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_CACHE_PATH):
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
            with open(CHUNKS_CACHE_PATH, "r", encoding="utf-8") as f:
                texts = json.load(f)
            chunks = [Document(page_content=t) for t in texts]
            return chunks, vectorstore

        # Build from scratch
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

        # Save chunk texts to cache
        with open(CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump([c.page_content for c in chunks], f, ensure_ascii=False)

        return chunks, vectorstore

    def _setup_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 6})
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": VERDICT_PROMPT}
        )

    def _find_article_chunks(self, article_num: str) -> list[Document]:
        pattern = re.compile(rf'(^|\n){re.escape(article_num)}\.\n', re.MULTILINE)
        # Return first match only (chunk 10 = real Article 6, not transitional provisions)
        for chunk in self.chunks:
            if pattern.search(chunk.page_content):
                return [chunk]
        return []

    def ask(self, query: str) -> str:
        match = re.search(r'article\s+(\d+)', query.lower())

        if match:
            article_num = match.group(1)
            direct_chunks = self._find_article_chunks(article_num)
            if direct_chunks:
                context = "\n\n".join([c.page_content for c in direct_chunks])
                final_prompt = VERDICT_PROMPT.format(context=context, question=query)
                response = self.llm.invoke(final_prompt)
                return response.content

        # Fallback to normal RAG
        response = self.qa_chain.invoke(query)
        return response["result"]