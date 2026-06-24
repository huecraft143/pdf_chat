# PDF Chat

> A production-ready RAG pipeline that lets users chat with any PDF. Built with LangChain, ChromaDB, and FastAPI.

## Architecture

```
Frontend (Streamlit)  →  Backend (FastAPI)  →  ChromaDB (local)
                                           →  Groq LLM (cloud)
```

1. User uploads a PDF → FastAPI splits it into chunks, embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB
2. User asks a question → FastAPI retrieves the top-4 relevant chunks and passes them to Groq's `llama3-8b-8192`
3. The LLM answers grounded strictly on the retrieved context, with page citations

## Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI + LangChain |
| Vector store | ChromaDB (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq API (llama3-8b-8192) |
| Frontend | Streamlit |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
```

### 3. Run the backend

```bash
uvicorn backend.main:app --reload
```

### 4. Run the frontend (separate terminal)

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501), upload a PDF, and start chatting.

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/upload` | POST | Upload a PDF, returns `session_id` |
| `/ask` | POST | Ask a question, returns answer + sources |
