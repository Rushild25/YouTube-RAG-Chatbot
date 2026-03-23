from __future__ import annotations

import json
from typing import Iterable


def iter_vector_records(path: str) -> Iterable[dict]:
    """Yield vector-store-ready records from JSONL file.

    Each record shape:
      {
        "id": str,
        "text": str,
        "embedding": list[float],
        "metadata": {...}
      }
    """
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
