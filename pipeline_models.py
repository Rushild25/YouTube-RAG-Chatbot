from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TranscriptChunk:
	text: str
	start: float
	duration: float


@dataclass
class TranscriptPayload:
	video_id: str
	language_code: str
	language_label: str
	is_generated: bool
	text: str
	chunks: list[TranscriptChunk]


@dataclass
class TranslationJob:
	video_id: str
	source_language: str
	target_language: str
	status: str
	text: str
	chunks: list[TranscriptChunk]
	metadata: dict[str, Any] = field(default_factory=dict)


class TranslationStatus:
	PENDING = "pending"
	SKIPPED = "skipped"
	PROCESSING = "processing"
	COMPLETED = "completed"
	PARTIAL = "partial"
	FAILED = "failed"


def normalize_language_code(code: Optional[str]) -> str:
	"""Normalize language codes like en-US or pt_BR to two-letter form where possible."""
	if not code:
		return "unknown"

	norm = code.strip().lower().replace("_", "-")
	if not norm:
		return "unknown"

	if "-" in norm:
		return norm.split("-", 1)[0]

	return norm
