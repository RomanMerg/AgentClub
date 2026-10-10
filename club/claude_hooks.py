"""Claude Code hook events -> club agents: one per session, plus one per subagent while it runs.

Hooks POST each event's JSON (the same JSON Claude Code passes to hook commands on stdin).
State rules:
  UserPromptSubmit / PreToolUse / PostToolUse -> working (bubble shows the tool in use)
  Notification about a permission              -> waiting ("Waiting on you")
  Notification "waiting for your input"        -> idle
  Stop                                         -> social/brag for 90 s, then idle
  Task/Agent tool                              -> parent social/watch, a subagent agent dances
  3 failed tool calls in a row                 -> error
  a tool running for 2+ min                    -> waiting ("Waiting on tool")
  working but silent for 5+ min                -> down
  SessionEnd                                   -> leaves the club
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePath

LONG_TOOL_S = 120
SILENT_DOWN_S = 300
DONE_TO_IDLE_S = 90
CHILD_LINGER_S = 30
FORGET_IDLE_S = 3 * 3600
TRANSCRIPT_EVERY_S = 15
SUBAGENT_TOOLS = {"Task", "Agent"}


def one_line(text: object, n: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def short_path(p: str) -> str:
    parts = PurePath(p).parts
    return "/".join(parts[-2:]) if parts else ""


def tool_summary(tool: str, inp: dict | None) -> tuple[str, str]:
    """(bubble title, sub line) for a tool call."""
    inp = inp or {}
    name = PurePath(inp.get("file_path") or inp.get("notebook_path") or "").name
    if tool == "Bash":
        return "Running a command", one_line(inp.get("description") or inp.get("command"), 70)
    if tool == "Read":
        return f"Reading {name}", short_path(inp.get("file_path", ""))
    if tool in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"Editing {name}", short_path(inp.get("file_path") or inp.get("notebook_path") or "")
    if tool == "Write":
        return f"Writing {name}", short_path(inp.get("file_path", ""))
    if tool in ("Grep", "Glob"):
        return "Searching the code", one_line(inp.get("pattern"), 60)
    if tool == "WebFetch":
        host = re.sub(r"^https?://", "", str(inp.get("url", ""))).split("/")[0]
        return "Reading a web page", host
    if tool == "WebSearch":
        return "Searching the web", one_line(inp.get("query"), 60)
    if tool == "TodoWrite":
        return "Planning the next steps", ""
    if tool.startswith("mcp__"):
        parts = tool.split("__")
        server = parts[1] if len(parts) > 1 else "MCP"
        return f"Using {server}", parts[-1] if len(parts) > 2 else ""
    return f"Using {tool}", ""


def pretty_model(model: str | None) -> str | None:
    """claude-opus-5-5 -> Opus 5.5, claude-sonnet-4-5-20250929 -> Sonnet 4.5."""
    if not model:
        return None
    parts = [p for p in model.replace("claude-", "").split("-") if not re.fullmatch(r"\d{8}", p)]
    words = [p for p in parts if not p.isdigit()]
    nums = [p for p in parts if p.isdigit()]
    if not words:
        return model
    return " ".join(w.capitalize() for w in words) + (" " + ".".join(nums) if nums else "")


class TranscriptUsage:
    """Reads a session transcript (JSONL) incrementally and totals token usage per assistant message."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.offset = 0
        self.by_id: dict[str, dict] = {}
        self.model: str | None = None
        self.last_context = 0

    def update(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        for raw in chunk[: end + 1].splitlines():
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            msg = entry.get("message") if isinstance(entry, dict) else None
            if not isinstance(msg, dict) or entry.get("type") != "assistant":
                continue
            usage = msg.get("usage") or {}
            mid = msg.get("id") or entry.get("uuid") or str(len(self.by_id))
            self.by_id[mid] = usage  # streamed parts repeat the same id; keep the latest
            self.model = msg.get("model") or self.model
            ctx = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            if ctx:
                self.last_context = ctx

    def totals(self) -> tuple[int, int]:
        tin = tout = 0
        for u in self.by_id.values():
            tin += sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            tout += int(u.get("output_tokens") or 0)
        return tin, tout

    def context_left(self) -> str | None:
        if not self.last_context:
            return None
        window = 1_000_000 if (self.model and "[1m]" in self.model) or self.last_context > 200_000 else 200_000
        return f"Context window: {max(0, round(100 - self.last_context / window * 100))}% left"


@dataclass
class Session:
    sid: str
    aid: str
    name: str
    parent: "Session | None" = None
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
    errors_in_row: int = 0
    transcript: TranscriptUsage | None = None
    transcript_read_at: float = 0.0
    children: list["Session"] = field(default_factory=list)
    child_count: int = 0


class ClaudeTracker:
    def __init__(self, store, clock=time.time) -> None:
        self.store = store
        self.clock = clock
        self.sessions: dict[str, Session] = {}
        self.lingering: list[Session] = []  # finished subagents still bragging
        self.lock = threading.RLock()
        self.events_seen = 0
        self.last_event_at: float | None = None

    # ---------- events ----------
    def handle(self, ev: dict) -> None:
        if not isinstance(ev, dict):
            return
        name = ev.get("hook_event_name") or ""
        sid = str(ev.get("session_id") or "")
        if not sid or not name:
            return
        now = self.clock()
        with self.lock:
            self.events_seen += 1
            self.last_event_at = now
            if name == "SessionEnd":
                self._end(sid)
                return
            s = self.sessions.get(sid) or self._new_session(sid, ev, now)
            s.last_event = now
            if ev.get("transcript_path") and not s.transcript:
                s.transcript = TranscriptUsage(ev["transcript_path"])
            tool = ev.get("tool_name") or ""
            inp = ev.get("tool_input") if isinstance(ev.get("tool_input"), dict) else {}

            if name == "SessionStart":
                self._set(s, "idle", now, title="Session started", sub="")
            elif name == "UserPromptSubmit":
                s.task = one_line(ev.get("prompt"), 300)
                s.started_at, s.done_at, s.took, s.errors_in_row = now, None, None, 0
                self._set(s, "working", now, title="Thinking…", sub="")
            elif name == "PreToolUse" and tool in SUBAGENT_TOOLS:
                self._spawn_child(s, inp, now)
            elif name == "PreToolUse":
                target = s.children[-1] if s.children else s
                target.last_event = now
                target.tool_started, target.tool_name = now, tool
                title, sub = tool_summary(tool, inp)
                self._set(target, "working", now, title=title, sub=sub)
            elif name in ("PostToolUse", "PostToolUseFailure") and tool in SUBAGENT_TOOLS:
                self._finish_child(s, now)
            elif name in ("PostToolUse", "PostToolUseFailure"):
                target = s.children[-1] if s.children else s
                target.last_event = now
                target.tool_started = None
                failed = name == "PostToolUseFailure" or _response_failed(ev.get("tool_response"))
                target.errors_in_row = target.errors_in_row + 1 if failed else 0
                if target.errors_in_row >= 3:
                    self._set(target, "error", now, title=f"{target.errors_in_row} tool calls failed in a row",
                              sub=f"last: {tool}", chip="Failing")
                elif target.state != "working":
                    title, sub = tool_summary(tool, inp)
                    self._set(target, "working", now, title=title, sub=sub)
                self._maybe_read_transcript(s, now)
            elif name == "Notification":
                msg = str(ev.get("message") or "")
                kind = str(ev.get("notification_type") or "")
                if kind == "idle_prompt" or "waiting for your input" in msg.lower():
                    self._set(s, "idle", now, title="Waiting for your next prompt", sub="")
                else:
                    self._set(s, "waiting", now, title="Needs your permission" if "permission" in msg.lower() else "Needs you",
                              sub=one_line(msg, 80), chip="Waiting on you")
            elif name == "SubagentStop":
                if s.children:
                    self._finish_child(s, now)
            elif name == "Stop":
                s.done_at = now
                s.took = (now - s.started_at) if s.started_at else None
                s.tool_started = None
                self._set(s, "social", now, mood="brag", title="Finished", sub=one_line(s.task, 70), chip="Done · bragging")
                self._maybe_read_transcript(s, now, force=True)
            elif name == "PreCompact":
                self._set(s, "working", now, title="Compacting its memory", sub="")
            self._publish(s)

    def tick(self) -> None:
        """Watchdog, called every few seconds: long tools, silent sessions, bragging that's gone on long enough."""
        now = self.clock()
        with self.lock:
            for s in list(self.sessions.values()):
                for c in list(s.children):
                    self._watch(c, now)
                self._watch(s, now)
                if s.state == "social" and s.mood == "brag" and s.done_at and now - s.done_at > DONE_TO_IDLE_S:
                    self._set(s, "idle", now, title="Waiting for your next prompt", sub="")
                if s.state == "idle" and now - s.last_event > FORGET_IDLE_S:
                    self._end(s.sid)
                    continue
                self._publish(s)
            for c in list(self.lingering):
                if now - (c.done_at or now) > CHILD_LINGER_S:
                    self.lingering.remove(c)
                    self.store.drop(c.aid)

    def status(self) -> dict:
        with self.lock:
            n = len(self.sessions)
            if not self.events_seen:
                return {"status": "waiting", "detail": "No hook events yet. Add the hook config to Claude Code."}
            ago = int(self.clock() - (self.last_event_at or 0))
            return {"status": "ok", "detail": f"{n} session{'s' if n != 1 else ''} · last event {ago}s ago"}

    # ---------- internals ----------
    def _new_session(self, sid: str, ev: dict, now: float) -> Session:
        base = "claude · " + (PurePath(str(ev.get("cwd") or "")).name or "session")
        name = base if all(s.name != base for s in self.sessions.values()) else f"{base} #{sid[:4]}"
        s = Session(sid=sid, aid=f"claude:{sid}", name=name, started_at=now, state_since=now, last_event=now)
        self.sessions[sid] = s
        return s

    def _set(self, s: Session, state: str, now: float, *, title: str, sub: str, mood: str | None = None, chip: str | None = None) -> None:
        if state != s.state or mood != s.mood:
            s.state_since = now
        s.state, s.mood, s.title, s.sub, s.chip = state, mood, title, sub, chip

    def _spawn_child(self, s: Session, inp: dict, now: float) -> None:
        s.child_count += 1
        kind = str(inp.get("subagent_type") or "subagent")
        c = Session(sid=s.sid, aid=f"{s.aid}:{s.child_count}", name=f"claude · {kind}", parent=s,
                    task=one_line(inp.get("description") or inp.get("prompt"), 300), started_at=now, state_since=now, last_event=now)
        self._set(c, "working", now, title="Starting up", sub=one_line(inp.get("description"), 60))
        s.children.append(c)
        self._set(s, "social", now, mood="watch", title="Waiting on its subagent", sub=one_line(inp.get("description"), 60),
                  chip="Waiting on subagent")

    def _finish_child(self, s: Session, now: float) -> None:
        if not s.children:
            return
        c = s.children.pop()
        c.done_at, c.took = now, now - c.started_at
        self._set(c, "social", now, mood="brag", title="Finished", sub=one_line(c.task, 70), chip="Done · bragging")
        self._put(c)
        self.lingering.append(c)
        if not s.children:
            self._set(s, "working", now, title="Back from its subagent", sub="")

    def _watch(self, s: Session, now: float) -> None:
        if s.state in ("working", "waiting") and s.tool_started and now - s.tool_started > LONG_TOOL_S and s.chip != "Waiting on you":
            mins = int((now - s.tool_started) // 60)
            self._set(s, "waiting", now, title=f"{s.tool_name or 'A tool'} is still running", sub=f"{mins} min so far", chip="Waiting on tool")
        elif s.state == "working" and not s.tool_started and not s.children and now - s.last_event > SILENT_DOWN_S:
            mins = int((now - s.last_event) // 60)
            self._set(s, "down", now, title=f"No events for {mins} min", sub=f"last: {s.title}")

    def _maybe_read_transcript(self, s: Session, now: float, force: bool = False) -> None:
        if not s.transcript or (not force and now - s.transcript_read_at < TRANSCRIPT_EVERY_S):
            return
        s.transcript_read_at = now

        def work() -> None:
            try:
                s.transcript.update()
            except OSError:
                return
            with self.lock:
                if s.sid in self.sessions:
                    self._publish(s)

        threading.Thread(target=work, daemon=True).start()

    def _end(self, sid: str) -> None:
        s = self.sessions.pop(sid, None)
        if not s:
            return
        for c in s.children:
            self.store.drop(c.aid)
        self.store.drop(s.aid)

    def _publish(self, s: Session) -> None:
        self._put(s)
        for c in s.children:
            self._put(c)

    def _put(self, s: Session) -> None:
        tr = (s.parent or s).transcript
        tokens = None
        if tr and tr.by_id and not s.parent:
            tin, tout = tr.totals()
            tokens = {"in": tin, "out": tout}
        self.store.put(s.aid, {
            "source": "claude",
            "name": s.name,
            "src": f"Claude Code subagent of {s.parent.name}" if s.parent else "Claude Code session",
            "state": s.state,
            "mood": s.mood,
            "title": s.title,
            "sub": s.sub,
            "chip": s.chip,
            "task": s.task or "No prompt yet",
            "model": pretty_model(tr.model) if tr else None,
            "tokens": tokens,
            "usage": tr.context_left() if tr and not s.parent else None,
            "startedAt": int(s.started_at * 1000),
            "stateSince": int(s.state_since * 1000),
            "took": round(s.took) if s.took is not None else None,
        })


def _response_failed(resp: object) -> bool:
    if isinstance(resp, dict):
        return bool(resp.get("is_error") or resp.get("error") or resp.get("interrupted"))
    return False
