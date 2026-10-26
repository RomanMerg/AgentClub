"""Codex sessions -> club agents, read from the rollout logs Codex writes (~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl).

Codex has no hooks to post to us, so this tails those files. One agent per session file.
State rules (same feel as the Claude Code ones):
  task_started / a tool call      -> working (bubble shows the tool)
  an approval request             -> waiting ("Waiting on you")
  task_complete                   -> social/brag for 90 s, then idle
  turn_aborted                    -> idle
  a tool running for 2+ min       -> waiting ("Waiting on tool")
  working but silent for 5+ min   -> down
  idle for 3 h, or file untouched -> leaves the club
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePath

from .claude_hooks import DONE_TO_IDLE_S, FORGET_IDLE_S, LONG_TOOL_S, SILENT_DOWN_S, one_line
from .sources import parse_ts

SESSIONS_DIR = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"


def pretty_model(model: str | None) -> str | None:
    """gpt-6-luna -> GPT-6 Luna, gpt-5.1-codex -> GPT-5.1 Codex."""
    if not model:
        return None
    m = re.match(r"gpt-([\d.]+)-?(.*)", model)
    if not m:
        return model
    rest = " ".join(w.capitalize() for w in m.group(2).split("-") if w)
    return f"GPT-{m.group(1)}" + (f" {rest}" if rest else "")


def tool_summary(name: str, inp: object) -> tuple[str, str]:
    """(bubble title, sub line) for a Codex tool call. Its `exec` tool wraps everything in a JS snippet."""
    text = inp if isinstance(inp, str) else json.dumps(inp or {})
    if "web__run" in text or name in ("web_search", "web_search_call"):
        q = re.search(r'q:\s*"([^"]+)"', text)
        return "Searching the web", one_line(q.group(1) if q else "", 60)
    if "apply_patch" in text or name == "apply_patch":
        f = re.search(r"\*\*\* (?:Update|Add) File: (.+)", text)
        return ("Editing " + PurePath(f.group(1).strip()).name) if f else "Editing files", ""
    cmd = re.search(r'cmd:\s*"((?:[^"\\]|\\.)*)"', text)
    if cmd or name in ("exec", "shell", "exec_command", "local_shell"):
        return "Running a command", one_line(cmd.group(1).replace("\\\\", "\\") if cmd else "", 70)
    return f"Using {name or 'a tool'}", ""


@dataclass
class CodexSession:
    path: Path
    sid: str = ""
    name: str = "codex · session"
    cwd: str = ""
    model: str | None = None
    state: str = "idle"
    mood: str | None = None
    title: str = "Session started"
    sub: str = ""
    chip: str | None = None
    task: str = ""
    started_at: float = 0.0
    state_since: float = 0.0
    last_event: float = 0.0
    tool_started: float | None = None
    tool_name: str | None = None
    done_at: float | None = None
    took: float | None = None
    tokens: dict | None = None
    context: str | None = None
    offset: int = 0
    open_calls: set = field(default_factory=set)

    def set(self, state: str, at: float, *, title: str, sub: str = "", mood: str | None = None, chip: str | None = None) -> None:
        if state != self.state or mood != self.mood:
            self.state_since = at
        self.state, self.mood, self.title, self.sub, self.chip = state, mood, title, sub, chip

    # ---------- one log line ----------
    def feed(self, entry: dict) -> None:
        if not isinstance(entry, dict):
            return
        at = parse_ts(entry.get("timestamp")) or time.time()
        kind, p = entry.get("type"), entry.get("payload")
        if not isinstance(p, dict):
            return
        self.last_event = max(self.last_event, at)
        ptype = str(p.get("type") or "")
        if kind == "session_meta":
            self.sid = str(p.get("id") or p.get("session_id") or self.sid)
            self.cwd = str(p.get("cwd") or "")
            self.started_at = self.started_at or at
            self.state_since = self.state_since or at
        elif kind == "turn_context":
            self.model = p.get("model") or self.model
        elif kind == "response_item" and ptype in ("custom_tool_call", "function_call", "local_shell_call"):
            self.open_calls.add(p.get("call_id"))
            self.tool_started, self.tool_name = at, str(p.get("name") or ptype)
            title, sub = tool_summary(self.tool_name, p.get("input") or p.get("arguments") or p.get("action"))
            self.set("working", at, title=title, sub=sub)
        elif kind == "response_item" and ptype in ("custom_tool_call_output", "function_call_output", "local_shell_call_output"):
            self.open_calls.discard(p.get("call_id"))
            if not self.open_calls:
                self.tool_started = None
            if self.state == "waiting":
                self.set("working", at, title="Thinking…")
        elif kind != "event_msg":
            return
        elif ptype == "task_started":
            self.started_at, self.done_at, self.took = at, None, None
            self.set("working", at, title="Thinking…")
        elif ptype == "user_message" or (ptype == "item_completed" and (p.get("item") or {}).get("type") == "UserMessage"):
            item = p.get("item") or {}
            parts = item.get("content") or []
            text = p.get("message") or " ".join(str(c.get("text") or "") for c in parts if isinstance(c, dict))
            self.task = one_line(text, 300) or self.task
        elif "approval" in ptype or ptype == "request_user_input":
            self.set("waiting", at, title="Needs your permission", sub=one_line(p.get("command") or p.get("reason") or "", 80),
                     chip="Waiting on you")
        elif ptype == "token_count":
            info = p.get("info") or {}
            total = info.get("total_token_usage") or {}
            if total:
                self.tokens = {"in": int(total.get("input_tokens") or 0), "out": int(total.get("output_tokens") or 0)}
            last, window = info.get("last_token_usage") or {}, info.get("model_context_window")
            if last and window:
                self.context = f"Context window: {max(0, round(100 - int(last.get('input_tokens') or 0) / int(window) * 100))}% left"
        elif ptype == "task_complete":
            self.done_at = at
            ms = p.get("duration_ms")
            self.took = ms / 1000 if isinstance(ms, (int, float)) else (at - self.started_at if self.started_at else None)
            self.tool_started, self.open_calls = None, set()
            self.set("social", at, mood="brag", title="Finished", sub=one_line(self.task, 70), chip="Done · bragging")
        elif ptype == "turn_aborted":
            self.tool_started, self.open_calls = None, set()
            self.set("idle", at, title="Stopped", sub="")

    # ---------- watchdog ----------
    def watch(self, now: float) -> None:
        if self.state in ("working", "waiting") and self.tool_started and now - self.tool_started > LONG_TOOL_S \
                and self.chip != "Waiting on you":
            mins = int((now - self.tool_started) // 60)
            self.set("waiting", now, title=f"{self.tool_name or 'A tool'} is still running", sub=f"{mins} min so far", chip="Waiting on tool")
        elif self.state == "working" and not self.tool_started and now - self.last_event > SILENT_DOWN_S:
            mins = int((now - self.last_event) // 60)
            self.set("down", now, title=f"No log lines for {mins} min", sub=f"last: {self.title}")
        if self.state == "social" and self.mood == "brag" and self.done_at and now - self.done_at > DONE_TO_IDLE_S:
            self.set("idle", now, title="Waiting for your next prompt")

    def agent(self) -> dict:
        return {
            "name": self.name,
            "src": "Codex session",
            "state": self.state,
            "mood": self.mood,
            "title": self.title,
            "sub": self.sub,
            "chip": self.chip,
            "task": self.task or "No prompt yet",
            "model": pretty_model(self.model),
            "tokens": self.tokens,
            "usage": self.context,
            "startedAt": int(self.started_at * 1000),
            "stateSince": int(self.state_since * 1000),
            "took": round(self.took) if self.took is not None else None,
        }


class Codex:
    """Polls the rollout folder. No config: it's on whenever Codex has written a session today or yesterday."""
    id = "codex"
    interval = 3

    def __init__(self, root: Path = SESSIONS_DIR, clock=time.time) -> None:
        self.root, self.clock = root, clock
        self.sessions: dict[Path, CodexSession] = {}

    def enabled(self, cfg: dict) -> bool:
        return self.root.is_dir()

    def recent_files(self, now: float) -> list[Path]:
        files = []
        for days in (0, 1):
            d = datetime.fromtimestamp(now) - timedelta(days=days)
            folder = self.root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
            if folder.is_dir():
                files += [f for f in folder.glob("rollout-*.jsonl") if now - f.stat().st_mtime < FORGET_IDLE_S]
        return files

    def poll(self, cfg: dict) -> dict[str, dict]:
        now = self.clock()
        live = set(self.recent_files(now))
        for gone in [p for p in self.sessions if p not in live]:
            del self.sessions[gone]
        for path in live:
            s = self.sessions.get(path) or self._new(path)
            self._read(s)
        agents = {}
        for s in self.sessions.values():
            s.watch(now)
            if s.state == "idle" and now - s.last_event > FORGET_IDLE_S:
                continue
            agents[f"codex:{s.sid or s.path.stem}"] = s.agent()
        return agents

    def detail(self, agents: dict) -> str:
        n = len(agents)
        return f"{n} session{'s' if n != 1 else ''} in the last 3 h" if n else "No Codex session in the last 3 h"

    def _new(self, path: Path) -> CodexSession:
        s = CodexSession(path=path)
        self.sessions[path] = s
        self._read(s)
        base = "codex · " + (PurePath(s.cwd).name or "session")
        taken = {o.name for o in self.sessions.values() if o is not s}
        s.name = base if base not in taken else f"{base} #{(s.sid or path.stem)[-4:]}"
        return s

    @staticmethod
    def _read(s: CodexSession) -> None:
        try:
            with s.path.open("rb") as f:
                f.seek(s.offset)
                chunk = f.read()
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        s.offset += end + 1
        for raw in chunk[: end + 1].splitlines():
            try:
                s.feed(json.loads(raw))
            except ValueError:
                continue
