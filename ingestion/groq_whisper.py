from __future__ import annotations
import tempfile
from pathlib import Path
from config import SETTINGS
from ingestion.audio_processor import download_audio
from groq import Groq

def _segment_to_item(segment: dict) -> dict:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start + 0.01))
    text = str(segment.get("text", "")).strip()
    return{
        "text": text,
        "start": start,
        "duration": max(0.01, end - start)
    }

def transcribe_with_groq_whisper(youtube_url: str) -> tuple[list[dict], str, str, str]:
    if not SETTINGS.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for groq-whisper mode")
    client = Groq(api_key=SETTINGS.groq_api_key)

    base_tmp = Path(SETTINGS.temp_audio_dir)
    base_tmp.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(base_tmp)) as workdir:
        audio_path, video_id = download_audio(
            youtube_url=youtube_url,
            output_dir=workdir,
            ffmpeg_path=SETTINGS.ffmpeg_path
        )

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file = (Path(audio_path).name, audio_file.read()),
                model=SETTINGS.groq_whisper_model,
                response_format="verbose_json"
            )

    language_code = str(getattr(transcription, "language", "unknown") or "unknown")

    segments = getattr(transcription,"segments", None) or []
    items: list[dict] = []
    
    for seg in segments:
        raw = seg if isinstance(seg, dict) else seg.__dict__
        item = _segment_to_item(raw)
        if item["text"]:
            items.append(item)

    if not items:
        whole_text = str(getattr(transcription, "text", "") or "").strip()
        if not whole_text:
            raise RuntimeError("Groq Whisper returned empty transcription")
        items.append({
            "text": whole_text, 
            "start":  0.0,
            "duration": 1.0
        })

    language_label = language_code
    return items, language_code, language_label, video_id