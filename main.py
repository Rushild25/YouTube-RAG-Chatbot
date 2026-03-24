from __future__ import annotations

from config import SETTINGS
from ingestion.chunking import chunk_transcript
from ingestion.embedding import EmbeddingService
from ingestion.transcript_processor import normalize_transcript_lines
from ingestion.youtube_loader import fetch_transcript
from llm.generator import AnswerGenerator
from retrieval.retriever import Retriever
from utils.helpers import build_chunk_id, extract_youtube_video_id
from vectorstore.qdrant_client import QdrantVectorStore


def process_video(url: str) -> tuple[str, int] | None:
    # Keep robust ID extraction for early URL validation.
    _ = extract_youtube_video_id(url)
    raw_items, lang_code, lang_label, video_id = fetch_transcript(url)

    # print("Language: ", lang_code)

    if not lang_code.lower().startswith("en"):
        print("Skipping non-english video")
        return None

    lines = normalize_transcript_lines(raw_items)

    chunks = chunk_transcript(
        lines=lines,
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError("No chunks generated from transcript")

    embedding_service = EmbeddingService(SETTINGS.embedding_model)
    vectors = embedding_service.embed_texts([chunk.text for chunk in chunks], batch_size=SETTINGS.embedding_batch_size)

    records: list[dict] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        source_id = build_chunk_id(video_id, idx, chunk.text)
        records.append(
            {
                "id": source_id,
                "embedding": vector.tolist(),
                "metadata": {
                    "video_id": video_id,
                    "chunk_id": source_id,
                    "text": chunk.text,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "language_detected": lang_code,
                    "language_label": lang_label,
                },
            }
        )

    store = QdrantVectorStore()
    store.upsert(records)
    return video_id, len(records)


def query_loop(video_id: str) -> None:
    embedding_service = EmbeddingService(SETTINGS.embedding_model)
    retriever = Retriever(embedding_service=embedding_service, vectorstore=QdrantVectorStore())
    generator = AnswerGenerator()

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
        print("\nCitations:")
        for item in retrieved:
            print(
                f"[{item['video_id']} | {item['start_time']}-{item['end_time']}] "
                f"score={item['score']:.4f}"
            )


def main() -> None:
    print("=" * 80)
    print("YouTube RAG Chatbot Backend (CLI)")
    print("=" * 80)

    if not SETTINGS.huggingface_api_key:
        print("Warning: HUGGINGFACE_API_KEY is empty. LLM output may fallback to extractive mode.")

    url = input("Enter YouTube URL: ").strip()
    if not url:
        raise ValueError("YouTube URL is required")

    processed = process_video(url)
    if processed is None:
        return

    video_id, count = processed
    print(f"\nIngestion complete. video_id={video_id} chunks_upserted={count}")

    query_loop(video_id)


if __name__ == "__main__":
    main()
