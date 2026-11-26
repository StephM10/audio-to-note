"""Shared helpers for parsing external ASR JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_safe(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_record_list(value: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in keys:
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    return []
