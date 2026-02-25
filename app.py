import os
import base64
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import httpx
from rag import VerdictRAG

rag_system: VerdictRAG = None
groq_client: Groq = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system, groq_client
    print("Initializing Verdict RAG system...")
    rag_system = VerdictRAG()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Ready.")
    yield

app = FastAPI(title="Verdict AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_lawyer(request: QueryRequest):
    try:
        answer = rag_system.ask(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-voice")
async def ask_voice(audio: UploadFile = File(...)):
    try:
        # Step 1: Save uploaded audio to temp file
        audio_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Step 2: Transcribe with Groq Whisper
        with open(tmp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio.filename or "audio.webm", f),
                model="whisper-large-v3",
                response_format="text"
            )
        os.unlink(tmp_path)
        question = transcription.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Step 3: Get answer from RAG
        answer = rag_system.ask(question)

        # Step 4: Convert answer to speech via ElevenLabs
        eleven_api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

        async with httpx.AsyncClient() as client:
            tts_response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": eleven_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": answer,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                timeout=30.0
            )

        if tts_response.status_code != 200:
            raise HTTPException(status_code=500, detail="TTS generation failed")

        # Step 5: Base64 encode to avoid illegal characters in HTTP headers
        q_encoded = base64.b64encode(question.encode()).decode()
        a_encoded = base64.b64encode(answer[:800].encode()).decode()

        return StreamingResponse(
            iter([tts_response.content]),
            media_type="audio/mpeg",
            headers={
                "X-Question": q_encoded,
                "X-Answer": a_encoded,
                "Access-Control-Expose-Headers": "X-Question, X-Answer"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": rag_system is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)