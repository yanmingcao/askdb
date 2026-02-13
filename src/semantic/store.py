"""Semantic store helpers."""

import json
import os
import sys
from typing import Dict, Any


def semantic_store_path() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "semantic_store.json")
    )


def load_semantic_store() -> Dict[str, Any]:
    path = semantic_store_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        message = (
            f"Invalid JSON in semantic_store.json at line {exc.lineno}, "
            f"column {exc.colno}. Fix the JSON and try again.\n"
            f"Path: {path}"
        )
        print(message, file=sys.stderr)
        raise SystemExit(1) from exc


def save_semantic_store(store: Dict[str, Any]) -> None:
    path = semantic_store_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2)
