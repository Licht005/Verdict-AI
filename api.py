from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import VerdictRAG

rag_system: VerdictRAG = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    print("Initializing Verdict RAG system...")
    rag_system = VerdictRAG()
    print("Ready.")
    yield

app = FastAPI(title="Verdict AI", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_lawyer(request: QueryRequest):
    try:
        answer = rag_system.ask(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)