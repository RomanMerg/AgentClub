"""Shared state: every agent the club shows, plus source status and infra checks. Readers wait for changes."""
from __future__ import annotations

import copy
import threading
import time


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self) -> None:
        self._cond = threading.Condition(threading.RLock())
        self._version = 0
        self._agents: dict[str, dict] = {}
        self._meta: dict[str, object] = {"sources": {}, "infra": [], "openrouter": None}

    # ---- writes (each bumps the version only when something actually changed) ----
    def put(self, agent_id: str, agent: dict) -> None:
        with self._cond:
            old = self._agents.get(agent_id)
            agent = {**agent, "id": agent_id}
            if old is not None and old.get("state") == agent.get("state") and "stateSince" not in agent:
                agent["stateSince"] = old.get("stateSince")
            agent.setdefault("stateSince", now_ms())
            if old != agent:
                self._agents[agent_id] = agent
                self._bump()

    def drop(self, agent_id: str) -> None:
        with self._cond:
            if self._agents.pop(agent_id, None) is not None:
                self._bump()

    def replace_source(self, source: str, agents: dict[str, dict]) -> None:
        """A poller's full view of its source: add/update these, remove the rest of that source."""
        with self._cond:
            for agent_id in [k for k, a in self._agents.items() if a.get("source") == source and k not in agents]:
                del self._agents[agent_id]
                self._bump()
            for agent_id, agent in agents.items():
                self.put(agent_id, {**agent, "source": source})

    def set_meta(self, key: str, value: object) -> None:
        with self._cond:
            if self._meta.get(key) != value:
                self._meta[key] = copy.deepcopy(value)
                self._bump()

    def set_source_status(self, source: str, status: dict) -> None:
        with self._cond:
            sources = dict(self._meta["sources"])  # type: ignore[arg-type]
            if sources.get(source) != status:
                sources[source] = dict(status)
                self._meta["sources"] = sources
                self._bump()

    def _bump(self) -> None:
        self._version += 1
        self._cond.notify_all()

    # ---- reads ----
    def agents(self, source: str | None = None) -> list[dict]:
        with self._cond:
            return [copy.deepcopy(a) for a in self._agents.values() if source is None or a.get("source") == source]

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "agents": [copy.deepcopy(a) for a in self._agents.values()],
                "sources": copy.deepcopy(self._meta["sources"]),
                "infra": copy.deepcopy(self._meta["infra"]),
                "openrouter": copy.deepcopy(self._meta["openrouter"]),
                "serverTime": now_ms(),
            }

    def wait(self, last_version: int, timeout: float) -> tuple[int, dict | None]:
        """Blocks until the version moves past last_version (or timeout). Returns (version, snapshot or None)."""
        with self._cond:
            if self._version == last_version:
                self._cond.wait(timeout)
            if self._version == last_version:
                return last_version, None
            return self._version, self.snapshot()
