"""Session state persistence for conversational ask behavior."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List


def _get_session_file() -> Path:
    """Get path to session state file in chromadb_data directory."""
    base = Path(__file__).resolve().parents[2] / "chromadb_data"
    base.mkdir(exist_ok=True)
    return base / "session_state.json"


def load_session_state() -> List[Dict[str, Any]]:
    """Load conversation history (sliding window of max 5 turns).

    Returns:
        List of turn dicts, each with keys: question, sql, columns
        Empty list if no previous session exists or file is invalid
    """
    session_file = _get_session_file()
    if not session_file.exists():
        return []

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle legacy format: single turn dict -> convert to list
        if isinstance(data, dict):
            if "question" in data and "sql" in data and "columns" in data:
                if "status" not in data:
                    data["status"] = "ok"
                return [data]
            return []

        # Handle new format: list of turns
        if isinstance(data, list):
            # Validate each turn
            valid_turns = []
            for turn in data:
                if (
                    isinstance(turn, dict)
                    and "question" in turn
                    and "sql" in turn
                    and "columns" in turn
                    and isinstance(turn["columns"], list)
                ):
                    if "status" not in turn:
                        turn["status"] = "ok"
                    valid_turns.append(turn)
            return valid_turns

        return []
    except (json.JSONDecodeError, OSError):
        return []


def save_session_state(
    question: str, sql: str, columns: List[str], status: str = "ok"
) -> None:
    """Save current turn context and maintain sliding window of max 5 turns.

    Args:
        question: User's natural language question
        sql: Generated SQL query
        columns: List of column names from result DataFrame
    """
    session_file = _get_session_file()

    # Load existing turns
    turns = load_session_state()

    # Append new turn
    new_turn = {
        "question": question,
        "sql": sql,
        "columns": columns,
        "status": status,
    }
    turns.append(new_turn)

    # Keep only last 5 turns
    turns = turns[-5:]

    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)
    except OSError:
        # Silently fail if we can't write session state
        pass


def clear_session_state() -> None:
    """Clear session state file."""
    session_file = _get_session_file()
    if session_file.exists():
        try:
            session_file.unlink()
        except OSError:
            pass


def update_last_turn_status(status: str) -> bool:
    """Update status of the most recent turn. Returns True if updated."""
    session_file = _get_session_file()
    turns = load_session_state()
    if not turns:
        return False

    turns[-1]["status"] = status
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)
    except OSError:
        return False
    return True
