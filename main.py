from __future__ import annotations
from config import SETTINGS
from langchain_core.documents import Document
from ingestion.chunking import chunk_transcript
from ingestion.embedding import EmbeddingService
from ingestion.transcript_processor import normalize_transcript_lines
from ingestion.youtube_loader import fetch_transcript
from llm.generator import create_answer_generator
from retrieval.retriever import Retriever
from utils.helpers import build_chunk_id
from vectorstore.qdrant_client import QdrantVectorStore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="YouTube RAG Chatbot API", version="1.0.0")

class IngestRequest(BaseModel):
    url: str
    transcript_mode: Optional[str] = None

class IngestResponse(BaseModel):
    video_id: str
    chunks_upserted: int
    language: str
    transcript_mode: Optional[str] = None

class AskRequest(BaseModel):
    video_id: str
    question: str
    top_k: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    contexts: list[dict]


def process_video(url: str, transcript_mode: str | None = None) -> tuple[str, int, str, str] | None:
    selected_mode = (transcript_mode or SETTINGS.transcript_source_mode or "transcript-api").strip().lower()

    if selected_mode == "transcript-api":
        raw_items, lang_code, lang_label, video_id = fetch_transcript(url)
    elif selected_mode == "groq-whisper":
        raise RuntimeError("groq-whisper mode is not wired in yet.")
    else:
        raise ValueError(f"Unsupported transcript mode: {selected_mode}")
    
    raw_items, lang_code, lang_label, video_id = fetch_transcript(url)

    # print("Language: ", lang_code)

    # if not lang_code.lower().startswith("en"):
    #     print("Skipping non-english video")
    #     return None

    lines = normalize_transcript_lines(raw_items)

    chunks = chunk_transcript(
        lines=lines,
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks generated from transcript")

    embedding_service = EmbeddingService(SETTINGS.embedding_model)
    store = QdrantVectorStore(embeddings=embedding_service.embeddings)

    documents: list[Document] = []
    ids: list[str] = []
    for idx, chunk in enumerate(chunks):
        source_id = build_chunk_id(video_id, idx, chunk.text)
        ids.append(source_id)
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "video_id": video_id,
                    "chunk_id": source_id,
                    "language_detected": lang_code,
                    "language_label": lang_label,
                },
            )
        )

    store.upsert_documents(documents=documents, ids=ids)
    return video_id, len(documents), lang_code, selected_mode

@app.get("/health")
def health() -> dict:
    return{
        "status": "ok"
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest_video(payload: IngestRequest) -> IngestResponse:
    try:
        processed = process_video(payload.url.strip(), transcript_mode = payload.transcript_mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if processed is None:
        raise HTTPException(status_code=400, detail="Skipping Non-english video")
    
    video_id, chunks_upserted, lang_code, transcript_source=processed
    return IngestResponse(
        video_id=video_id,
        chunks_upserted=chunks_upserted,
        language=lang_code,
        transcript_source=transcript_source
        )


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    embedding_service= EmbeddingService(SETTINGS.embedding_model)
    retriever = Retriever(vectorstore=QdrantVectorStore(embeddings = embedding_service.embeddings))
    try:
        generator = create_answer_generator(SETTINGS.llm_provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    retrieved = retriever.retrieve(
        question=question,
        top_k=payload.top_k or SETTINGS.top_k,
        video_id=payload.video_id
    )

    if not retrieved:
        return AskResponse(
            answer="I do not have enough context from the video to answer confidently.",
            contexts=[]
        )
    
    answer = generator.generate_answer(question, retrieved)
    return AskResponse(
        answer=answer,
        contexts=retrieved
    )

def query_loop(video_id: str) -> None:
    embedding_service = EmbeddingService(SETTINGS.embedding_model)
    retriever = Retriever(vectorstore=QdrantVectorStore(embeddings=embedding_service.embeddings))
    generator = create_answer_generator(SETTINGS.llm_provider)

    print("\nAsk questions about the video. Type 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        retrieved = retriever.retrieve(question=question, top_k=SETTINGS.top_k, video_id=video_id)
        if not retrieved:
            print("Answer: I do not have enough context from the video to answer that confidently.")
            continue

        answer = generator.generate_answer(question, retrieved)
        print(f"\nAnswer: {answer}")
        for item in retrieved:
            print(
                f"[{item['video_id']}]"
            )


def main() -> None:
    print("=" * 80)
    print("YouTube RAG Chatbot Backend (CLI)")
    print("=" * 80)

    if SETTINGS.llm_provider == "huggingface" and not SETTINGS.huggingface_api_key:
        print("Warning: HUGGINGFACE_API_KEY is empty.")
    if SETTINGS.llm_provider == "groq" and not SETTINGS.groq_api_key:
        print("Warning: GROQ_API_KEY is empty.")

    url = input("Enter YouTube URL: ").strip()
    if not url:
        raise ValueError("YouTube URL is required")

    processed = process_video(url)
    if processed is None:
        return

    video_id, count, lang_code = processed
    print(f"\nIngestion complete. video_id={video_id} chunks_upserted={count} language={lang_code}")

    query_loop(video_id)


if __name__ == "__main__":
    main()
