"""Semantic store helpers."""

import json
import os
from typing import Dict, Any


def semantic_store_path() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "semantic_store.json")
    )


def load_semantic_store() -> Dict[str, Any]:
    path = semantic_store_path()
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_semantic_store(store: Dict[str, Any]) -> None:
    path = semantic_store_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2)
