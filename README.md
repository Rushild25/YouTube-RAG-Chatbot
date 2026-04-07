<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=220&text=YouTube%20RAG%20Chatbot&fontAlign=50&fontAlignY=40&color=0:ff6a00,40:ee0979,100:00c6ff&fontColor=ffffff&animation=fadeIn" alt="YouTube RAG Chatbot banner"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=for-the-badge" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/Groq-Whisper_Flow-111827?style=for-the-badge" alt="Groq Whisper"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/Transcript_Mode-auto%20%7C%20transcript--api%20%7C%20groq--whisper-orange?style=flat-square" alt="Transcript Modes"/>
</p>

## Overview

A Retrieval-Augmented Generation (RAG) chatbot for YouTube videos using:

- FastAPI backend for ingestion and Q&A endpoints
- Streamlit frontend for easy interaction
- Qdrant vector store for semantic retrieval
- Dual transcript ingestion paths:
  - `transcript-api` (YouTube transcript API)
  - `groq-whisper` (audio transcription fallback path)
- Configurable LLM provider (`huggingface` or `groq`) for answer generation

---

## Features

- Ingest a YouTube URL and chunk transcript text for retrieval
- Ask grounded questions against ingested video context
- Track transcript source used during ingestion (`transcript-api` or `groq-whisper`)
- Automatic transcript source mode supported via `auto` (try transcript API first, then fallback when transcript is unavailable)
- Streamlit UI + FastAPI Swagger support
- Clean pipeline for switching between HuggingFace and Groq generation

---

## Project Structure

```text
YouTube-RAG-Chatbot/
  config.py
  main.py
  streamlit_app.py
  requirements.txt
  .env.example
  ingestion/
  llm/
  retrieval/
  vectorstore/
  utils/
```

---

## Prerequisites

- Python **3.11** (recommended)
- Git
- Internet access for YouTube and model/API calls
- ffmpeg (required for `groq-whisper` ingestion mode)

---

## Setup

### 1. Clone and enter the project

```powershell
git clone <your-repo-url>
cd YouTube-RAG-Chatbot
```

### 2. Create a Python 3.11 virtual environment

> Use Python 3.11 explicitly.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If your machine uses `python` directly:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and update values:

```powershell
Copy-Item .env.example .env
```

Set at least:

- `HUGGINGFACE_API_TOKEN` (if `LLM_PROVIDER=huggingface`)
- `GROQ_API_KEY` (if `LLM_PROVIDER=groq` or using `groq-whisper` mode)
- `QDRANT_URL` (default local: `http://localhost:6333`)
- `TRANSCRIPT_SOURCE_MODE` (`auto`, `transcript-api`, or `groq-whisper`)

### 5. Start Qdrant (local)

If Docker is installed:

```powershell
docker run -p 6333:6333 qdrant/qdrant
```

---

## ffmpeg Setup (for `groq-whisper` mode)

If `ffmpeg -version` works in terminal, you are done.

If not, set this in `.env`:

```env
FFMPEG_PATH=C:/full/path/to/ffmpeg.exe
```

Use forward slashes to avoid escape issues in `.env`.

---

## Run the App

### Backend (FastAPI)

```powershell
uvicorn main:app --reload
```

- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Frontend (Streamlit)

Open a second terminal (same virtual environment), then run:

```powershell
streamlit run streamlit_app.py
```

- UI: `http://localhost:8501`

---

## Ingestion Modes

`transcript_mode` can be:

- `auto`: try `transcript-api`, fallback to `groq-whisper` when transcript is unavailable
- `transcript-api`: use YouTube transcript API only
- `groq-whisper`: force Groq Whisper audio transcription

Response includes `transcript_source` so you can verify the path used.

---

## Quick API Examples

### Ingest

```json
{
  "url": "https://www.youtube.com/watch?v=JGwWNGJdvx8",
  "transcript_mode": "auto"
}
```

### Ask

```json
{
  "video_id": "JGwWNGJdvx8",
  "question": "What is this video about?",
  "top_k": 5
}
```

---

## Verification Commands

Compile-check key files:

```powershell
python -m compileall main.py streamlit_app.py ingestion llm retrieval vectorstore
```

---

## Common Issues

- `ffmpeg is required for groq-whisper mode`
  - Ensure `ffmpeg -version` works OR set `FFMPEG_PATH` correctly in `.env`

- `GROQ_API_KEY is required`
  - Add valid Groq API key in `.env`

- Ingestion/network errors for YouTube
  - Retry with stable internet
  - Check firewall/DNS/proxy settings

- Empty retrieval contexts
  - Ensure ingestion completed successfully
  - Use returned `video_id` exactly

---

<p align="center">
  <sub>FastAPI + Streamlit + Qdrant + Groq Whisper</sub>
</p>
