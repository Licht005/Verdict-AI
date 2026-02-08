from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import VerdictRAG  # Import your logic script

app = FastAPI()

# Initialize the RAG system globally so it loads once
print("Initializing Verdict RAG system...")
rag_system = VerdictRAG()

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