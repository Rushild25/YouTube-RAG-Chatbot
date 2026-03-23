"""Demo: Export translated chunks as JSON ready for vector DB indexing."""
import os
from ingestion import ingest_youtube_url
from translator import translate_job_hybrid, write_chunks_json
import json

DEMO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Test with this known English video
TARGET_LANGUAGE = "en"
OUTPUT_FILE = "translated_chunks_en.json"

print("=" * 80)
print("EXPORT DEMO: Translate + Export to JSON")
print("=" * 80)

# Step 1: Ingest
print(f"\n[1/3] Ingesting URL: {DEMO_URL}")
payload, job = ingest_youtube_url(DEMO_URL, TARGET_LANGUAGE)
print(f"      Video ID: {job.video_id}")
print(f"      Detected language: {payload.language_label}")
print(f"      Transcript lines: {len(job.chunks)}")

# Step 2: Translate with hybrid mode
print(f"\n[2/3] Translating with hybrid mode (window=60s, batch_size=5)...")
job = translate_job_hybrid(
    job,
    window_seconds=60.0,
    translation_batch_size=5,
    semantic_chunk_size=850,
    semantic_chunk_overlap=120,
)
print(f"      Translation status: {job.status}")
print(f"      Final semantic chunks: {len(job.chunks)}")
print(f"      Primary engine: {job.metadata.get('engine_primary')}")
print(f"      Fallback engine: {job.metadata.get('engine_fallback')}")

# Step 3: Export to JSON
print(f"\n[3/3] Exporting to {OUTPUT_FILE}...")
export_result = write_chunks_json(job, OUTPUT_FILE, video_url=DEMO_URL)
print(f"      Chunks exported: {export_result['chunks_exported']}")
print(f"      File size: {os.path.getsize(OUTPUT_FILE)} bytes")

# Display sample chunks
print(f"\n" + "=" * 80)
print("SAMPLE EXPORTED CHUNKS (first 3):")
print("=" * 80)

with open(OUTPUT_FILE, "r") as f:
    data = json.load(f)

for i, chunk in enumerate(data["chunks"][:3]):
    print(f"\nChunk {i + 1}:")
    print(f"  start_time: {chunk['start_time']:.2f}s")
    print(f"  end_time: {chunk['end_time']:.2f}s")
    print(f"  duration: {chunk['duration']:.2f}s")
    print(f"  text: {chunk['text'][:100]}...")

print(f"\n" + "=" * 80)
print(f"✓ Full export written to {OUTPUT_FILE}")
print(f"✓ Ready for vector embedding + indexing")
print("=" * 80)
