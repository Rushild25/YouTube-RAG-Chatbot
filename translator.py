from __future__ import annotations
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pipeline_models import (
	TranscriptChunk,
	TranslationJob,
	TranslationStatus,
	normalize_language_code,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunked(items: list[str], size: int) -> list[list[str]]:
	return [items[index : index + size] for index in range(0, len(items), size)]


@dataclass
class _TimeWindow:
	start: float
	end: float
	text: str


def _build_time_windows(chunks: list[TranscriptChunk], window_seconds: float) -> list[_TimeWindow]:
	"""Merge caption lines into larger windows for faster translation requests."""
	if not chunks:
		return []

	windows: list[_TimeWindow] = []
	current_start = chunks[0].start
	current_end = chunks[0].start + chunks[0].duration
	buffer: list[str] = []

	for chunk in chunks:
		line = chunk.text.strip()
		if line:
			buffer.append(line)

		line_end = chunk.start + chunk.duration
		current_end = max(current_end, line_end)

		if current_end - current_start >= window_seconds:
			text = " ".join(buffer).strip()
			if text:
				windows.append(_TimeWindow(start=current_start, end=current_end, text=text))
			buffer = []
			current_start = line_end
			current_end = line_end

	if buffer:
		text = " ".join(buffer).strip()
		if text:
			windows.append(_TimeWindow(start=current_start, end=current_end, text=text))

	return windows


def _recursive_semantic_split(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
	"""Prefer RecursiveCharacterTextSplitter; fallback to a simple recursive splitter."""
	if not text.strip():
		return []

	try:
		from langchain_text_splitters import RecursiveCharacterTextSplitter

		splitter = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,
			chunk_overlap=chunk_overlap,
			separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
		)
		return [piece.strip() for piece in splitter.split_text(text) if piece.strip()]
	except Exception:  # noqa: BLE001
		# Fallback: whitespace-based splitting with overlap by words.
		words = text.split()
		if not words:
			return []

		max_words = max(40, chunk_size // 6)
		overlap_words = min(max_words // 3, max(0, chunk_overlap // 6))
		pieces: list[str] = []
		start = 0
		while start < len(words):
			end = min(len(words), start + max_words)
			piece = " ".join(words[start:end]).strip()
			if piece:
				pieces.append(piece)
			if end == len(words):
				break
			start = max(0, end - overlap_words)

		return pieces


def _map_split_text_to_time_ranges(window: _TimeWindow, splits: list[str]) -> list[TranscriptChunk]:
	"""Assign approximate time ranges to semantic split text within a translated window."""
	if not splits:
		return []

	window_text = window.text
	window_len = max(1, len(window_text))
	time_span = max(0.01, window.end - window.start)
	cursor = 0
	built: list[TranscriptChunk] = []

	for piece in splits:
		piece_clean = piece.strip()
		if not piece_clean:
			continue

		# Approximate position by searching from the current cursor.
		match_pos = window_text.find(piece_clean, cursor)
		if match_pos == -1:
			match_pos = cursor

		match_end = min(window_len, match_pos + len(piece_clean))
		start_ratio = match_pos / window_len
		end_ratio = max(start_ratio, match_end / window_len)

		start_time = window.start + (time_span * start_ratio)
		end_time = window.start + (time_span * end_ratio)
		if end_time <= start_time:
			end_time = start_time + 0.01

		built.append(
			TranscriptChunk(
				text=piece_clean,
				start=start_time,
				duration=end_time - start_time,
			)
		)
		cursor = match_end

	return built


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseTranslator(ABC):
	name = "base"

	@abstractmethod
	def translate_batch(
		self,
		texts: list[str],
		source_language: str,
		target_language: str,
	) -> list[str]:
		"""Translate a batch preserving input order."""


# ---------------------------------------------------------------------------
# Engine 1: deep-translator (Google Translate, no API key required)
# This is the recommended primary engine. Install with:
#   pip install deep-translator
# ---------------------------------------------------------------------------

class DeepGoogleTranslator(BaseTranslator):
	"""
	Wraps `deep-translator`'s GoogleTranslator.

	Advantages over local ML engines:
	  - No model downloads; works immediately after `pip install deep-translator`.
	  - Supports 100+ languages including Hindi (hi).
	  - Google quality translation.

	Batch note: GoogleTranslator accepts one text at a time, so we loop
	internally. The batch interface is preserved for API consistency.
	"""

	name = "deep-google"

	def __init__(self) -> None:
		try:
			from deep_translator import GoogleTranslator  # noqa: F401
		except ImportError as exc:
			raise RuntimeError(
				"deep-translator is not installed. Run: pip install deep-translator"
			) from exc

	def translate_batch(
		self,
		texts: list[str],
		source_language: str,
		target_language: str,
	) -> list[str]:
		if not texts:
			return []

		from deep_translator import GoogleTranslator

		# deep-translator uses "auto" for auto-detection, or an ISO 639-1 code.
		src = source_language if source_language != "unknown" else "auto"

		translator = GoogleTranslator(source=src, target=target_language)
		translated: list[str] = []
		for text in texts:
			result = translator.translate(text)
			translated.append((result or text).strip())

		if len(translated) != len(texts):
			raise RuntimeError("DeepGoogleTranslator output length mismatch")

		return translated


# ---------------------------------------------------------------------------
# Engine 2: MarianMT (local, HuggingFace transformers)
# Good offline fallback but requires ~300 MB model download per language pair.
# Install with:  pip install transformers sentencepiece torch
# ---------------------------------------------------------------------------

class MarianMTTranslator(BaseTranslator):
	name = "marianmt"

	def __init__(self) -> None:
		self._model_cache: dict[str, object] = {}

	def _load_pipeline(self, source_language: str, target_language: str):
		try:
			from transformers import pipeline
		except ImportError as exc:
			raise RuntimeError(
				"transformers is not installed. Run: pip install transformers sentencepiece torch"
			) from exc

		pair_key = f"{source_language}-{target_language}"
		if pair_key in self._model_cache:
			return self._model_cache[pair_key]

		model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"
		translator_pipeline = pipeline("translation", model=model_name)
		self._model_cache[pair_key] = translator_pipeline
		return translator_pipeline

	def translate_batch(
		self,
		texts: list[str],
		source_language: str,
		target_language: str,
	) -> list[str]:
		if not texts:
			return []

		pipe = self._load_pipeline(source_language, target_language)
		outputs = pipe(texts)
		translated = [item["translation_text"].strip() for item in outputs]
		if len(translated) != len(texts):
			raise RuntimeError("MarianMT output length mismatch")
		return translated


# ---------------------------------------------------------------------------
# Internal retry helper (unchanged)
# ---------------------------------------------------------------------------

def _attempt_batch_translation(
	translator: BaseTranslator,
	batch: list[str],
	source_language: str,
	target_language: str,
	max_retries: int,
	base_backoff_sec: float,
) -> list[str]:
	last_error: Exception | None = None

	for attempt in range(max_retries + 1):
		try:
			return translator.translate_batch(batch, source_language, target_language)
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt == max_retries:
				break
			time.sleep(base_backoff_sec * (2**attempt))

	raise RuntimeError(f"{translator.name} failed after retries: {last_error}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def translate_job(
	job: TranslationJob,
	primary_translator: BaseTranslator | None = None,
	fallback_translator: BaseTranslator | None = None,
	batch_size: int = 12,
	max_retries: int = 2,
	base_backoff_sec: float = 1.0,
) -> TranslationJob:
	"""
	Translate transcript chunks while preserving timestamps.

	Default engine order:
	  1. DeepGoogleTranslator  (primary  — lightweight, no model downloads)
	  2. MarianMTTranslator    (fallback — local ML, needs transformers + model)

	The chunk count and timing are preserved exactly.
	Only chunk.text and job.text are mutated.
	"""
	if batch_size <= 0:
		raise ValueError("batch_size must be > 0")

	source_language = normalize_language_code(job.source_language)
	target_language = normalize_language_code(job.target_language)

	if job.status == TranslationStatus.SKIPPED or source_language == target_language:
		job.status = TranslationStatus.SKIPPED
		job.metadata["reason"] = "source and target languages are the same"
		return job

	job.status = TranslationStatus.PROCESSING
	job.metadata["source_language_normalized"] = source_language
	job.metadata["target_language_normalized"] = target_language

	# --- Primary engine ---
	primary: BaseTranslator
	if primary_translator is not None:
		primary = primary_translator
	else:
		# DeepGoogleTranslator is the new default primary (no setup required).
		try:
			primary = DeepGoogleTranslator()
		except RuntimeError:
			# If deep-translator isn't installed, fall back to MarianMT as primary.
			primary = MarianMTTranslator()

	# --- Fallback engine ---
	fallback: BaseTranslator | None
	if fallback_translator is not None:
		fallback = fallback_translator
	else:
		# Use MarianMT as the default fallback (local, offline capable).
		try:
			fallback = MarianMTTranslator()
		except Exception:  # noqa: BLE001
			fallback = None

	texts = [chunk.text for chunk in job.chunks]
	batches = _chunked(texts, batch_size)

	translated_texts: list[str] = []
	failed_batches: list[int] = []
	primary_used = 0
	fallback_used = 0

	for batch_index, batch in enumerate(batches):
		translated_batch: list[str] | None = None

		try:
			translated_batch = _attempt_batch_translation(
				translator=primary,
				batch=batch,
				source_language=source_language,
				target_language=target_language,
				max_retries=max_retries,
				base_backoff_sec=base_backoff_sec,
			)
			primary_used += 1
		except Exception as primary_error:  # noqa: BLE001
			job.metadata.setdefault("primary_errors", []).append(
				{"batch": batch_index, "error": str(primary_error)}
			)
			if fallback is not None:
				try:
					translated_batch = _attempt_batch_translation(
						translator=fallback,
						batch=batch,
						source_language=source_language,
						target_language=target_language,
						max_retries=max_retries,
						base_backoff_sec=base_backoff_sec,
					)
					fallback_used += 1
				except Exception as fallback_error:  # noqa: BLE001
					job.metadata.setdefault("fallback_errors", []).append(
						{"batch": batch_index, "error": str(fallback_error)}
					)

		if translated_batch is None:
			failed_batches.append(batch_index)
			# Preserve order and chunk count even on failure by keeping original text.
			translated_texts.extend(batch)
		else:
			translated_texts.extend(translated_batch)

	for chunk, translated_text in zip(job.chunks, translated_texts):
		chunk.text = translated_text

	job.text = " ".join(item for item in translated_texts if item)
	job.metadata["engine_primary"] = primary.name
	job.metadata["engine_fallback"] = fallback.name if fallback is not None else None
	job.metadata["primary_batches_used"] = primary_used
	job.metadata["fallback_batches_used"] = fallback_used
	job.metadata["failed_batches"] = failed_batches

	if failed_batches:
		job.status = (
			TranslationStatus.PARTIAL
			if len(failed_batches) < len(batches)
			else TranslationStatus.FAILED
		)
	else:
		job.status = TranslationStatus.COMPLETED

	return job


def translate_job_hybrid(
	job: TranslationJob,
	primary_translator: BaseTranslator | None = None,
	fallback_translator: BaseTranslator | None = None,
	window_seconds: float = 45.0,
	translation_batch_size: int = 6,
	semantic_chunk_size: int = 900,
	semantic_chunk_overlap: int = 120,
	max_retries: int = 2,
	base_backoff_sec: float = 1.0,
) -> TranslationJob:
	"""Hybrid fast mode: translate larger time windows, then semantically split.

	Practical time references are preserved as approximate ranges derived from each translated window.
	"""
	if window_seconds <= 0:
		raise ValueError("window_seconds must be > 0")

	source_language = normalize_language_code(job.source_language)
	target_language = normalize_language_code(job.target_language)

	if job.status == TranslationStatus.SKIPPED or source_language == target_language:
		job.status = TranslationStatus.SKIPPED
		job.metadata["reason"] = "source and target languages are the same"
		return job

	windows = _build_time_windows(job.chunks, window_seconds=window_seconds)
	if not windows:
		job.status = TranslationStatus.FAILED
		job.metadata["error"] = "No transcript windows built for translation"
		return job

	window_job = TranslationJob(
		video_id=job.video_id,
		source_language=source_language,
		target_language=target_language,
		status=TranslationStatus.PENDING,
		text="\n\n".join(window.text for window in windows),
		chunks=[TranscriptChunk(text=window.text, start=window.start, duration=window.end - window.start) for window in windows],
		metadata={},
	)

	translated_window_job = translate_job(
		job=window_job,
		primary_translator=primary_translator,
		fallback_translator=fallback_translator,
		batch_size=translation_batch_size,
		max_retries=max_retries,
		base_backoff_sec=base_backoff_sec,
	)

	semantic_chunks: list[TranscriptChunk] = []
	failed_semantic_windows: list[int] = []

	for index, (window, translated_window_chunk) in enumerate(zip(windows, translated_window_job.chunks)):
		translated_text = translated_window_chunk.text.strip()
		if not translated_text:
			failed_semantic_windows.append(index)
			continue

		translated_window = _TimeWindow(
			start=window.start,
			end=window.end,
			text=translated_text,
		)
		splits = _recursive_semantic_split(
			translated_text,
			chunk_size=semantic_chunk_size,
			chunk_overlap=semantic_chunk_overlap,
		)
		mapped = _map_split_text_to_time_ranges(translated_window, splits)
		if not mapped:
			failed_semantic_windows.append(index)
			continue
		semantic_chunks.extend(mapped)

	job.chunks = semantic_chunks
	job.text = " ".join(chunk.text for chunk in semantic_chunks)
	job.metadata.update(
		{
			"mode": "hybrid-window-semantic",
			"window_seconds": window_seconds,
			"translation_batch_size": translation_batch_size,
			"semantic_chunk_size": semantic_chunk_size,
			"semantic_chunk_overlap": semantic_chunk_overlap,
			"source_line_chunks": len(window_job.chunks),
			"semantic_chunks": len(semantic_chunks),
			"failed_semantic_windows": failed_semantic_windows,
			"translation_stage": translated_window_job.status,
			"engine_primary": translated_window_job.metadata.get("engine_primary"),
			"engine_fallback": translated_window_job.metadata.get("engine_fallback"),
			"translation_failed_batches": translated_window_job.metadata.get("failed_batches", []),
		},
	)

	if translated_window_job.status == TranslationStatus.FAILED or not semantic_chunks:
		job.status = TranslationStatus.FAILED
	elif translated_window_job.status == TranslationStatus.PARTIAL or failed_semantic_windows:
		job.status = TranslationStatus.PARTIAL
	else:
		job.status = TranslationStatus.COMPLETED

	return job


# ---------------------------------------------------------------------------
# Export / Indexing Ready Functions
# ---------------------------------------------------------------------------


def export_chunks_to_json(
	job: TranslationJob,
	video_url: str = "",
) -> list[dict]:
	"""Export semantic chunks as JSON-ready dicts for vector DB indexing.
	
	Each chunk carries:
	  - text: translated semantic chunk
	  - start_time: float seconds
	  - end_time: float seconds
	  - duration: derived duration
	  - video_id: for reference and filtering
	  - video_url: optional full URL for UI navigation
	  - source_window_index: which time window it came from
	  - chunk_index: position in final semantic chunks (0-based)
	  - original_chunk_count: metadata about original ingestion
	  - timestamp: ISO format when exported
	"""
	if job.status not in [TranslationStatus.COMPLETED, TranslationStatus.PARTIAL, TranslationStatus.SKIPPED]:
		return []
	
	if not job.chunks:
		return []
	
	exported: list[dict] = []
	original_chunk_count = job.metadata.get("original_chunk_count", len(job.chunks))
	
	for chunk_index, chunk in enumerate(job.chunks):
		exported_chunk = {
			"text": chunk.text,
			"start_time": chunk.start,
			"end_time": chunk.start + chunk.duration,
			"duration": chunk.duration,
			"video_id": job.video_id,
			"video_url": video_url,
			"source_window_index": chunk_index,  # In hybrid mode, this is approximate window mapping
			"chunk_index": chunk_index,
			"original_chunk_count": original_chunk_count,
			"timestamp": datetime.utcnow().isoformat() + "Z",
			"translation_status": job.status,
		}
		exported.append(exported_chunk)
	
	return exported


def write_chunks_json(
	job: TranslationJob,
	filepath: str,
	video_url: str = "",
) -> dict:
	"""Write semantic chunks to JSON file ready for vector DB ingestion.
	
	Returns metadata dict with:
	  - filepath: where it was written
	  - chunks_exported: count of chunks
	  - status: job completion status
	  - timestamp: when written
	"""
	chunks_list = export_chunks_to_json(job, video_url=video_url)
	
	output_data = {
		"metadata": {
			"video_id": job.video_id,
			"video_url": video_url,
			"source_language": job.source_language,
			"target_language": job.target_language,
			"translation_status": job.status,
			"chunks_exported": len(chunks_list),
			"original_line_chunks": job.metadata.get("original_chunk_count", "unknown"),
			"final_semantic_chunks": len(chunks_list),
			"mode": job.metadata.get("mode", "unknown"),
			"window_seconds": job.metadata.get("window_seconds"),
			"semantic_chunk_size": job.metadata.get("semantic_chunk_size"),
			"timestamp": datetime.utcnow().isoformat() + "Z",
		},
		"chunks": chunks_list,
	}
	
	with open(filepath, "w", encoding="utf-8") as f:
		json.dump(output_data, f, indent=2, ensure_ascii=False)
	
	return {
		"filepath": filepath,
		"chunks_exported": len(chunks_list),
		"status": job.status,
		"timestamp": datetime.utcnow().isoformat() + "Z",
	}