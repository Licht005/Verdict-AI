# Verdict AI — Ghana Constitutional Law Research Assistant

Verdict AI is an AI-powered legal research tool that lets you query Ghana's 1992 Constitution using natural language — either by typing or speaking. It retrieves exact constitutional provisions, cites specific articles and clauses, and can respond in both text and voice.

Built as a demonstration of retrieval-augmented generation (RAG) applied to a structured legal document.

---

## What It Does

- **Text queries** — Ask any question about the Constitution and get a cited, clause-level answer
- **Voice queries** — Speak your question, get a spoken response back (full STT → RAG → TTS pipeline)
- **Exact article lookup** — Queries like "What does Article 6 say?" retrieve the precise constitutional text, not a paraphrase
- **Dark, professional UI** — Built for clarity and readability in a legal research context

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3.3 70B via Groq API |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace |
| Vector Store | FAISS (CPU) |
| RAG Framework | LangChain |
| PDF Parsing | pdfplumber |
| Speech-to-Text | Whisper Large v3 via Groq API |
| Text-to-Speech | ElevenLabs Turbo v2 |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | HuggingFace Spaces (Docker) |

---

## Project Structure

```
Verdict (v2)/
│
├── api.py               # FastAPI server — /ask and /ask-voice endpoints
├── rag.py               # VerdictRAG class — article lookup + vector search fallback
├── prompt.py            # System prompt template for the LLM
├── index.html           # Frontend — text and voice UI
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container config for HuggingFace Spaces
│
├── Doc/
│   └── Ghana Constitution.pdf   # Source document
│
├── faiss_index/
│   ├── index.faiss      # Pre-built vector index
│   └── index.pkl        # Index metadata
│
└── chunks_cache.json    # Pre-split document chunks (avoids re-processing PDF on startup)
```

---

## How the RAG Works

1. Ghana's 1992 Constitution PDF is parsed and split into overlapping chunks
2. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in a FAISS index
3. On query, the system first checks if the user is asking about a specific article using regex pattern matching — if so, it retrieves that article's text directly from the cache
4. For topic-based queries (e.g. "what does the Constitution say about freedom of speech?"), it falls back to semantic vector search
5. Retrieved chunks are passed to LLaMA 3.3 70B with a structured prompt that instructs it to cite specific clauses in its response

The direct article lookup bypasses vector search entirely for article-specific queries, which significantly improves accuracy on structured legal documents where semantic similarity can be unreliable.

---

## Running Locally

### Prerequisites
- Python 3.11+
- A Groq API key (free at [groq.com](https://groq.com))
- An ElevenLabs API key (free tier at [elevenlabs.io](https://elevenlabs.io))

### Setup

```bash
# Clone the repo
git clone https://github.com/Licht005/Verdict-AI.git
cd Verdict-AI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb  # optional, defaults to George
```

### Run

```bash
python api.py
```

Then open `index.html` in your browser. The app will be available at `http://localhost:7860`.

> **Note:** On first run, if `faiss_index/` is not present, the system will build the vector index from scratch. This takes 1–2 minutes. Subsequent startups load the cached index and are near-instant.

---

## Deployment

The backend is deployed on **HuggingFace Spaces** using Docker. The frontend (`index.html`) can be served from any static host — Vercel, GitHub Pages, or directly from the browser as a local file.

Live demo: [huggingface.co/spaces/LucasLicht/verdict-ai](https://huggingface.co/spaces/LucasLicht/verdict-ai)

---

## Notes

- Voice mode requires microphone permissions in the browser
- HTTPS is required for microphone access in production (HuggingFace Spaces provides this automatically)
- The ElevenLabs free tier allows 10,000 characters/month — sufficient for demos and light use
- The Groq free tier is generous and handles both LLM inference and Whisper transcription

---

