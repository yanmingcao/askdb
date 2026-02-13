# pyright: reportGeneralTypeIssues=false
"""Vanna AI configuration and initialization for AskDB."""

import os
import time
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI
from vanna.legacy.chromadb.chromadb_vector import ChromaDB_VectorStore

from src.config.database import get_db_utils

load_dotenv()


class AskDBVanna(ChromaDB_VectorStore):
    """Custom Vanna class using local ChromaDB + OpenAI."""

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, client: Optional[OpenAI] = None
    ):
        ChromaDB_VectorStore.__init__(self, config=config)
        self.temperature = 0.7
        if config and "temperature" in config:
            self.temperature = config["temperature"]
        if client is None:
            raise ValueError("OpenAI client must be provided")
        self.client = client
        self.last_prompt: Optional[Any] = None
        self.last_response: Optional[str] = None
        self.llm_metrics_history: List[Dict[str, Any]] = []

    def reset_llm_metrics(self) -> None:
        self.llm_metrics_history = []
        self.last_prompt = None
        self.last_response = None

    def get_llm_metrics_summary(self) -> Optional[Dict[str, Any]]:
        if not self.llm_metrics_history:
            return None
        total_prompt = sum(m.get("prompt_tokens", 0) for m in self.llm_metrics_history)
        total_completion = sum(
            m.get("completion_tokens", 0) for m in self.llm_metrics_history
        )
        total_time_ms = sum(m.get("duration_ms", 0) for m in self.llm_metrics_history)
        approx = any(m.get("approx", False) for m in self.llm_metrics_history)
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "duration_ms": total_time_ms,
            "approx": approx,
            "calls": len(self.llm_metrics_history),
        }

    def log(self, message: str, title: str = "Info") -> None:
        if title == "SQL Prompt":
            self.last_prompt = message
        if title == "LLM Response":
            self.last_response = message

        if self.config and self.config.get("verbose"):
            print(f"{title}: {message}")

    def system_message(self, message: str) -> Any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> Any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> Any:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4

        model = None
        if kwargs.get("model", None) is not None:
            model = kwargs.get("model", None)
        elif kwargs.get("engine", None) is not None:
            model = kwargs.get("engine", None)
        elif self.config is not None and "engine" in self.config:
            model = self.config["engine"]
        elif self.config is not None and "model" in self.config:
            model = self.config["model"]
        else:
            model = "gpt-3.5-turbo-16k" if num_tokens > 3500 else "gpt-3.5-turbo"

        if not model:
            model = "gpt-3.5-turbo"

        model_str = str(model)

        if self.config and self.config.get("verbose"):
            print(f"Using model {model_str} for {num_tokens:.0f} tokens (approx)")

        start = time.monotonic()
        response = self.client.chat.completions.create(
            model=model_str,
            messages=prompt,
            stop=None,
            temperature=self.temperature,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        choices = getattr(response, "choices", [])
        content_text = ""
        for choice in choices:
            choice_text = getattr(choice, "text", None)
            if choice_text is not None:
                content_text = str(choice_text)
                break

        if not content_text and choices:
            message = getattr(choices[0], "message", None)
            content_text = str(getattr(message, "content", "") or "")

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            approx = False
        else:
            prompt_tokens = int(num_tokens)
            completion_tokens = int(len(content_text) / 4)
            approx = True

        self.llm_metrics_history.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "duration_ms": duration_ms,
                "model": model_str,
                "approx": approx,
            }
        )

        return content_text


def _normalize_base_url(value: str) -> str:
    return value.strip().strip('"')


def get_vanna_config() -> Dict[str, Any]:
    """Build Vanna config dict from environment variables."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_model = os.getenv("OPENROUTER_MODEL", "").strip()
    openrouter_base = _normalize_base_url(os.getenv("OPENROUTER_BASE_URL", "").strip())

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", "").strip()
    vanna_model = os.getenv("VANNA_MODEL", "gpt-4").strip()

    provider = "openrouter" if openrouter_key else "openai"
    model = openrouter_model or openai_model or vanna_model
    api_key = openrouter_key or openai_key
    base_url = (
        openrouter_base or "https://openrouter.ai/api/v1" if openrouter_key else ""
    )

    if not api_key or api_key == "your-openai-api-key-here":
        raise ValueError(
            "LLM API key is not set. Configure OPENROUTER_API_KEY or OPENAI_API_KEY."
        )

    return {
        "api_key": api_key,
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "verbose": False,
        "path": os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "chromadb_data")
        ),
    }


def get_table_allowlist() -> Optional[list[str]]:
    """Get table allowlist from semantic store. Returns None if empty (= all tables)."""
    from src.semantic.store import load_semantic_store, semantic_store_path

    if not os.path.exists(semantic_store_path()):
        return None

    store = load_semantic_store()
    allowlist = store.get("allowlist", [])
    if not allowlist:
        return None
    return [t.strip() for t in allowlist if isinstance(t, str) and t.strip()]


def create_vanna_instance() -> AskDBVanna:
    """Create and configure a Vanna instance connected to MySQL."""
    config = get_vanna_config()
    base_url = config.get("base_url") or None
    client = OpenAI(api_key=config["api_key"], base_url=base_url)
    vn = AskDBVanna(config=config, client=client)

    db_utils = get_db_utils()
    db_utils.connect_vanna(vn)
    return vn


_vanna_instance: Optional[AskDBVanna] = None


def get_vanna() -> AskDBVanna:
    """Get or create the singleton Vanna instance."""
    global _vanna_instance
    if _vanna_instance is None:
        _vanna_instance = create_vanna_instance()
    return _vanna_instance
