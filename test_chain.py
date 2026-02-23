# test_chain.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from prompt import VERDICT_PROMPT

load_dotenv()

FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

docs = vectorstore.similarity_search("what does article 6 say", k=6)
context = "\n\n".join([d.page_content for d in docs])

final_prompt = VERDICT_PROMPT.format(context=context, question="what does article 6 say")
print(final_prompt)