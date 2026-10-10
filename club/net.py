"""Small HTTP client for the pollers (stdlib only). Errors never include headers or keys."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request


class NetError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def get(url: str, headers: dict | None = None, timeout: float = 8) -> tuple[int, bytes, float]:
    """Returns (status, body, milliseconds). Raises NetError only when the server can't be reached."""
    req = urllib.request.Request(url, headers={"User-Agent": "agent-club/0.1", **(headers or {})})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(5_000_000), (time.monotonic() - t0) * 1000
    except urllib.error.HTTPError as e:
        body = e.read(100_000) if e.fp else b""
        return e.code, body, (time.monotonic() - t0) * 1000
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        text = str(reason)
        if isinstance(reason, ConnectionRefusedError) or "10061" in text or "refused" in text.lower():
            text = "nothing is listening there; is it running?"
        elif "timed out" in text.lower():
            text = "timed out"
        raise NetError(f"can't reach {_host(url)}: {text[:80]}") from None


def get_json(url: str, headers: dict | None = None, timeout: float = 8):
    status, body, _ = get(url, headers, timeout)
    if status == 401 or status == 403:
        raise NetError(f"{_host(url)} rejected the key (HTTP {status})", status)
    if status >= 400:
        raise NetError(f"{_host(url)} answered HTTP {status}", status)
    try:
        return json.loads(body or b"null")
    except ValueError:
        raise NetError(f"{_host(url)} didn't return JSON", status) from None


def basic_auth(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc or url
