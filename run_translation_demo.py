from __future__ import annotations
from ingestion import ingest_youtube_url
from translator import translate_job_hybrid
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable


DEMO_URL = "https://www.youtube.com/watch?v=etnLX7m2MiA&t=107s"
TARGET_LANGUAGE = "en"


if __name__ == "__main__":
    try:
        payload, translation_job = ingest_youtube_url(DEMO_URL, target_language=TARGET_LANGUAGE)
        print(f"Input URL: {DEMO_URL}")
        print(f"Video ID: {payload.video_id}")
        print(f"Detected language: {payload.language_code} ({payload.language_label})")
        print(f"Transcript chunks: {len(payload.chunks)}")
        print(f"Translation status before: {translation_job.status}")

        original_chunks = [c.text for c in translation_job.chunks]

        translated_job = translate_job_hybrid(
            translation_job,
            window_seconds=60.0,
            translation_batch_size=5,
            semantic_chunk_size=850,
            semantic_chunk_overlap=120,
        )
        print(f"Translation status after: {translated_job.status}")
        preview_count = 5
        for i in range(min(preview_count, len(translated_job.chunks))):
            chunk = translated_job.chunks[i]
            print(
                f"\nChunk {i+1} | approx_start={chunk.start:.2f}s | "
                f"approx_end={chunk.start + chunk.duration:.2f}s"
            )
            print(f"TRANSLATED:\n{chunk.text[:350]}")
        
        print(
            f"Engines used: primary={translated_job.metadata.get('engine_primary')} "
            f"fallback={translated_job.metadata.get('engine_fallback')}"
        )
        print(f"Translation failed batches: {translated_job.metadata.get('translation_failed_batches')}")
        print(f"Hybrid mode metadata: {translated_job.metadata.get('mode')}")
        print(f"Original line chunks: {len(original_chunks)}")
        print(f"Final semantic chunks: {len(translated_job.chunks)}")

    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        print("No transcript found for this video.")
    except VideoUnavailable:
        print("Video is unavailable/private/removed.")
    except Exception as exc:
        print(f"Demo failed: {exc}")
