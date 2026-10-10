"""Settings live in .env.local next to the app. The server never serves that file and never sends keys to the page."""
from __future__ import annotations

import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env.local"

DEFAULTS = {
    "N8N_URL": "http://localhost:5678",
    "N8N_API_KEY": "",
    "LANGFUSE_HOST": "https://cloud.langfuse.com",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "LANGFUSE_PREFIX": "langgraph",
    "OPENROUTER_API_KEY": "",
    "OLLAMA_URL": "http://localhost:11434",
    "OLLAMA_MODELS": "",
    "SUPABASE_URL": "",
    "SUPABASE_ANON_KEY": "",
}
SECRETS = {"N8N_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OPENROUTER_API_KEY", "SUPABASE_ANON_KEY"}

_lock = threading.Lock()


def read_env(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load(path: Path = ENV_FILE) -> dict[str, str]:
    with _lock:
        return {**DEFAULTS, **read_env(path)}


def save(updates: dict[str, str | None], path: Path = ENV_FILE) -> None:
    """Merge updates into the file. A value of None removes the key. Comments and order are kept."""
    for key, value in updates.items():
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError(f"{key} can't contain a line break")
    with _lock:
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else [
            "# Agent Club settings. Keep this file private; it is never served to the browser."
        ]
        pending = dict(updates)
        out: list[str] = []
        for raw in lines:
            key = raw.partition("=")[0].strip() if "=" in raw and not raw.lstrip().startswith("#") else None
            if key in pending:
                value = pending.pop(key)
                if value is not None:
                    out.append(f"{key}={value}")
                continue
            out.append(raw)
        out.extend(f"{k}={v}" for k, v in pending.items() if v is not None)
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


def mask(value: str) -> str:
    """Shows only that a key is saved, plus its last four characters."""
    return "" if not value else "••••" + value[-4:]
