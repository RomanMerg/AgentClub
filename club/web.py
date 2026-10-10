"""HTTP: the page, the live stream, Claude Code hooks and the Connect panel's API. Local-only by design."""
from __future__ import annotations

import json
import re
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from . import config
from .config import ROOT
from .sources import check_details, load_infra

PAGES = {"/": "index.html", "/index.html": "index.html"}
MAX_BODY = 1_000_000

# which config keys each Connect card may set
SOURCE_FIELDS = {
    "n8n": {"url": "N8N_URL", "key": "N8N_API_KEY"},
    "langfuse": {"host": "LANGFUSE_HOST", "pk": "LANGFUSE_PUBLIC_KEY", "sk": "LANGFUSE_SECRET_KEY", "prefix": "LANGFUSE_PREFIX"},
    "openrouter": {"key": "OPENROUTER_API_KEY"},
    "ollama": {"url": "OLLAMA_URL", "models": "OLLAMA_MODELS"},
    "infra": {"supabaseUrl": "SUPABASE_URL", "supabaseKey": "SUPABASE_ANON_KEY"},
}
URL_KEYS = {"N8N_URL", "LANGFUSE_HOST", "OLLAMA_URL", "SUPABASE_URL"}


def host_allowed(host: str | None, port: int) -> bool:
    """Blocks DNS-rebinding: only answer requests addressed to this machine by name."""
    return host in (f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}")


def origin_allowed(origin: str | None, port: int) -> bool:
    """No Origin (curl, Claude Code hooks) is fine; a browser Origin must be this page."""
    return origin is None or origin in (f"http://localhost:{port}", f"http://127.0.0.1:{port}")


def public_sources(cfg: dict, status: dict) -> dict:
    """What the Connect panel may see: values for plain fields, masked secrets, and each source's status."""
    out = {}
    for sid, fields in SOURCE_FIELDS.items():
        vals = {f: (config.mask(cfg.get(k, "")) if k in config.SECRETS else cfg.get(k, "")) for f, k in fields.items()}
        out[sid] = {"fields": vals, **status.get(sid, {"status": "off", "detail": "Not connected"})}
    out["claude"] = status.get("claude", {"status": "waiting", "detail": "No hook events yet."})
    return out


def validate(sid: str, body: dict, cfg: dict) -> dict[str, str | None]:
    """Turns a Connect form into config updates. Empty secret fields keep the saved key."""
    if sid not in SOURCE_FIELDS:
        raise ValueError("Unknown source")
    fields = SOURCE_FIELDS[sid]
    if body.get("disconnect"):
        return {k: None for k in fields.values() if k in config.SECRETS}
    updates: dict[str, str | None] = {}
    for f, key in fields.items():
        if f not in body:
            continue
        value = str(body[f] or "").strip()
        if len(value) > 500 or "\n" in value or "\r" in value:
            raise ValueError(f"{f} looks wrong")
        if key in config.SECRETS and not value:
            continue
        if key in URL_KEYS and value and not re.match(r"^https?://", value):
            raise ValueError(f"{f} must start with http:// or https://")
        updates[key] = value
    return updates


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AgentClub/0.1"

        def log_message(self, fmt, *args):  # keep the console quiet; hooks fire a lot
            pass

        # ---------- helpers ----------
        def _guard(self) -> bool:
            if not host_allowed(self.headers.get("Host"), app.port):
                self._json(403, {"error": "Only answers on localhost"})
                return False
            return True

        def _json(self, code: int, payload: object) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("Too large")
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            return data

        # ---------- GET ----------
        def do_GET(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if path in PAGES:
                body = (ROOT / PAGES[path]).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/api/health":
                self._json(200, {"ok": True, "app": "agent-club", "version": app.version})
            elif path == "/api/snapshot":
                self._json(200, app.store.snapshot())
            elif path == "/api/sources":
                self._json(200, public_sources(config.load(), app.statuses()))
            elif path == "/api/infra/details":
                name = (parse_qs(urlparse(self.path).query).get("name") or [""])[0]
                item = next((i for i in load_infra() if i.get("name") == name), None)
                if not item:
                    self._json(404, {"error": "No such menu item"})
                else:
                    self._json(200, {"name": name, "checkedAt": int(time.time() * 1000), "rows": check_details(item, config.load())})
            elif path == "/api/stream":
                self._stream()
            else:
                self._json(404, {"error": "Not found"})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            version = -1
            try:
                while True:
                    version, snap = app.store.wait(version, timeout=15)
                    if snap is None:
                        self.wfile.write(b": ping\n\n")
                    else:
                        self.wfile.write(b"data: " + json.dumps(snap).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return

        # ---------- POST ----------
        def do_POST(self):
            if not self._guard():
                return
            path = urlparse(self.path).path
            if not origin_allowed(self.headers.get("Origin"), app.port):
                self._json(403, {"error": "Cross-site request blocked"})
                return
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})
                return
            if path == "/api/hooks/claude":
                app.claude.handle(body)
                # empty reply: hook output must never reach Claude's context
                self.send_response(204)
                self.end_headers()
            elif path.startswith("/api/sources/"):
                if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                    self._json(415, {"error": "Send JSON"})
                    return
                sid = path.rsplit("/", 1)[-1]
                try:
                    updates = validate(sid, body, config.load())
                    config.save(updates)
                except ValueError as e:
                    self._json(400, {"error": str(e)})
                    return
                status = app.runners[sid].poll_now() if sid in app.runners else {}
                self._json(200, public_sources(config.load(), app.statuses())[sid] | {"result": status})
            else:
                self._json(404, {"error": "Not found"})

    return Handler
