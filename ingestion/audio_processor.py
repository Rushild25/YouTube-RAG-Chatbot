from __future__ import annotations
import shutil
from pathlib import Path
from yt_dlp import YoutubeDL

def _ensure_ffmpeg_available(ffmpeg_path: str | None=None) -> None:
    if ffmpeg_path:
        candidate = Path(ffmpeg_path)
        if not candidate.exists():
            raise RuntimeError(f"FFMPEG_PATH does not exist: {ffmpeg_path}")
        return
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for groq-whisper mode. Install ffmpeg or FFMPEG_PATH."
        )
    
def download_audio(youtube_url: str, output_dir: str, ffmpeg_path: str | None = None) -> tuple[str, str]:
    _ensure_ffmpeg_available(ffmpeg_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(out_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }
        ]
    }

    if ffmpeg_path:
        ydl_opts["ffmpeg_location"] = ffmpeg_path

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
    
    video_id = str(info.get("id", "")).strip()
    if not video_id:
        raise RuntimeError("Unable to extract video id while downloading audio")
    
    audio_path = out_dir / f"{video_id}.mp3"
    if not audio_path.exists():
        candidates = list(out_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise RuntimeError("Audio file was not created by yt-dlp/ffmpeg")
        audio_path = candidates[0]

    return str(audio_path), video_id