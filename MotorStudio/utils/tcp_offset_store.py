"""TCP 偏移配置持久化。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Sequence


DEFAULT_TCP_OFFSET = [0.0] * 6


def _config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "el_a3_sdk"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "el_a3_sdk"

    base = os.getenv("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "el_a3_sdk"
    return Path.home() / ".config" / "el_a3_sdk"


def get_tcp_offset_path() -> Path:
    return _config_dir() / "motorstudio_tcp_offset.json"


def normalize_tcp_offset(values: Sequence[float] | None) -> List[float]:
    offset = list(DEFAULT_TCP_OFFSET)
    if values is None:
        return offset
    seq = list(values)
    for idx in range(min(6, len(seq))):
        try:
            offset[idx] = float(seq[idx])
        except (TypeError, ValueError):
            offset[idx] = 0.0
    return offset


def load_tcp_offset(default: Sequence[float] | None = None) -> List[float]:
    path = get_tcp_offset_path()
    fallback = normalize_tcp_offset(default)
    if not path.exists():
        return fallback

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return fallback

    values = payload
    if isinstance(payload, dict):
        values = payload.get("tcp_offset", payload.get("offset", fallback))
    return normalize_tcp_offset(values)


def save_tcp_offset(offset: Sequence[float]) -> Path:
    path = get_tcp_offset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tcp_offset": normalize_tcp_offset(offset)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

