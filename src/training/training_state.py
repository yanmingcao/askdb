"""Training state helpers for incremental semantic training."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict


def _chromadb_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "chromadb_data"


def training_state_path() -> Path:
    return _chromadb_data_path() / "training_state.json"


def load_training_state() -> Dict[str, Any]:
    path = training_state_path()
    if not path.exists():
        return {"version": 1, "items": {}}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return {"version": 1, "items": {}}
        items = payload.get("items", {})
        if not isinstance(items, dict):
            items = {}
        return {"version": payload.get("version", 1), "items": items}
    except Exception:
        return {"version": 1, "items": {}}


def save_training_state(items: Dict[str, Dict[str, Any]]) -> None:
    path = training_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "items": items}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def reset_chromadb_data() -> None:
    data_dir = _chromadb_data_path()
    if data_dir.exists():
        shutil.rmtree(data_dir)


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
