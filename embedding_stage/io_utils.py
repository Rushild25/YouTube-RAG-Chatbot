from __future__ import annotations

import json
from typing import Any


def load_translated_chunks(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load export JSON produced by write_chunks_json()."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    metadata = data.get("metadata", {})
    chunks = data.get("chunks", [])

    if not isinstance(chunks, list):
        raise ValueError("Invalid chunks payload. Expected a list.")

    return metadata, chunks


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
