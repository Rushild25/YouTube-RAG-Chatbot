from __future__ import annotations

from typing import Optional
from urllib.parse import parse_qs, urlparse

from pipeline_models import (
	TranscriptChunk,
	TranscriptPayload,
	TranslationJob,
	TranslationStatus,
	normalize_language_code,
)
from translator import translate_job_hybrid
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
	NoTranscriptFound,
	TranscriptsDisabled,
	VideoUnavailable,
)


def extract_youtube_video_id(url: str) -> Optional[str]:
	"""Extract a YouTube video ID from a URL.

	Supported forms:
	- https://www.youtube.com/watch?v=<ID>&...
	- https://m.youtube.com/watch?v=<ID>
	- https://youtube.com/watch?v=<ID>
	- https://youtu.be/<ID>?t=30
	- https://www.youtube.com/embed/<ID>
	- https://www.youtube.com/shorts/<ID>

	Returns:
		The video ID when found, otherwise None.
	"""
	if not url or not url.strip():
		return None

	raw = url.strip()

	# Allow inputs like "youtube.com/watch?v=..." without explicit scheme.
	if not raw.startswith(("http://", "https://")):
		raw = f"https://{raw}"

	parsed = urlparse(raw)
	host = parsed.netloc.lower()
	path = parsed.path.strip("/")

	def clean_id(candidate: Optional[str]) -> Optional[str]:
		if not candidate:
			return None
		# Guard against accidental trailing fragments/params in malformed inputs.
		candidate = candidate.split("?")[0].split("&")[0].split("#")[0].strip()
		return candidate or None

	# Case A: youtu.be/<ID>
	if "youtu.be" in host:
		first_segment = path.split("/")[0] if path else ""
		return clean_id(first_segment)

	# Case B/C: youtube.com/watch, /embed/<ID>, /shorts/<ID>
	if "youtube.com" in host:
		segments = [segment for segment in path.split("/") if segment]

		# /watch?v=<ID>
		if segments and segments[0] == "watch":
			query_params = parse_qs(parsed.query)
			return clean_id(query_params.get("v", [None])[0])

		# /embed/<ID>
		if len(segments) >= 2 and segments[0] == "embed":
			return clean_id(segments[1])

		# /shorts/<ID>
		if len(segments) >= 2 and segments[0] == "shorts":
			return clean_id(segments[1])

	return None


def fetch_transcript_payload(video_id: str) -> TranscriptPayload:
	"""Fetch transcript in original language and return normalized payload."""
	api = YouTubeTranscriptApi()
	transcript_list = api.list(video_id)

	# Prefer manually created captions when available, then fallback to generated.
	transcript = next(iter(transcript_list), None)
	if transcript is None:
		raise NoTranscriptFound(video_id, [], "No transcript entries found")

	fetched = transcript.fetch().to_raw_data()
	chunks: list[TranscriptChunk] = []

	for item in fetched:
		chunk_text = item.get("text", "").strip()
		if not chunk_text:
			continue
		chunks.append(
			TranscriptChunk(
				text=chunk_text,
				start=float(item.get("start", 0.0)),
				duration=float(item.get("duration", 0.0)),
			)
		)

	text = " ".join(chunk.text for chunk in chunks)

	return TranscriptPayload(
		video_id=video_id,
		language_code=getattr(transcript, "language_code", "unknown"),
		language_label=getattr(transcript, "language", "unknown"),
		is_generated=bool(getattr(transcript, "is_generated", False)),
		text=text,
		chunks=chunks,
	)


def create_translation_job(
	payload: TranscriptPayload,
	target_language: str = "en",
) -> TranslationJob:
	"""Create a translation job payload for a separate translation module."""
	source_language = normalize_language_code(payload.language_code)
	target = normalize_language_code(target_language)
	status = TranslationStatus.PENDING if source_language != target else TranslationStatus.SKIPPED
	return TranslationJob(
		video_id=payload.video_id,
		source_language=source_language,
		target_language=target,
		status=status,
		text=payload.text,
		chunks=payload.chunks,
		metadata={"source_language_label": payload.language_label},
	)


def ingest_youtube_url(url: str, target_language: str = "en") -> tuple[TranscriptPayload, TranslationJob]:
	"""Run ingestion stage only: parse URL, fetch transcript, queue translation decision."""
	video_id = extract_youtube_video_id(url)
	if not video_id:
		raise ValueError(f"Could not extract video id from URL: {url}")

	payload = fetch_transcript_payload(video_id)
	job = create_translation_job(payload, target_language=target_language)
	return payload, job


if __name__ == "__main__":
	sample_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
	try:
		payload, translation_job = ingest_youtube_url(sample_url, target_language="en")
		print(f"Video ID: {payload.video_id}")
		print(f"Detected language: {payload.language_code} ({payload.language_label})")
		print(f"Transcript chunks: {len(payload.chunks)}")
		print(f"Translation job status (before): {translation_job.status}")

		translated_job = translate_job_hybrid(
			translation_job,
			window_seconds=60.0,
			translation_batch_size=5,
			semantic_chunk_size=850,
			semantic_chunk_overlap=120,
		)
		print(f"Translation job status (after): {translated_job.status}")
		print(
			f"Translation engines: primary={translated_job.metadata.get('engine_primary')} "
			f"fallback={translated_job.metadata.get('engine_fallback')}"
		)
		print(f"Hybrid chunks generated: {len(translated_job.chunks)}")
	except TranscriptsDisabled:
		print("Transcripts are disabled for this video.")
	except NoTranscriptFound:
		print("No transcript found for this video.")
	except VideoUnavailable:
		print("Video is unavailable/private/removed.")
	except Exception as exc:
		print(f"Ingestion failed: {exc}")