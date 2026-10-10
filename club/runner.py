"""Runs each source on its own thread and records its status for the Connect panel."""
from __future__ import annotations

import threading
import time

from . import config, net
from .store import Store, now_ms


class Runner(threading.Thread):
    def __init__(self, source, store: Store, on_result=None) -> None:
        super().__init__(daemon=True, name=f"poll-{source.id}")
        self.source = source
        self.store = store
        self.on_result = on_result
        self.wake = threading.Event()
        self.poll_lock = threading.Lock()
        self.status: dict = {"status": "off", "detail": "Not connected"}

    def run(self) -> None:
        while True:
            self.poll_now()
            self.wake.wait(self.source.interval)
            self.wake.clear()

    def poll_now(self) -> dict:
        with self.poll_lock:
            cfg = config.load()
            sid = self.source.id
            if not self.source.enabled(cfg):
                self._set({"status": "off", "detail": "Not connected"})
                if sid not in ("openrouter", "infra"):
                    self.store.replace_source(sid, {})
                return self.status
            try:
                result = self.source.poll(cfg)
            except net.NetError as e:
                self._set({"status": "error", "detail": str(e), "lastOk": self.status.get("lastOk")})
                return self.status
            except Exception as e:  # a bad payload shouldn't kill the thread
                self._set({"status": "error", "detail": f"unexpected reply ({type(e).__name__})", "lastOk": self.status.get("lastOk")})
                return self.status
            if isinstance(result, dict) and sid not in ("openrouter",):
                self.store.replace_source(sid, result)
            if self.on_result:
                self.on_result(result)
            self._set({"status": "ok", "detail": self.source.detail(result), "lastOk": now_ms()})
            return self.status

    def _set(self, status: dict) -> None:
        self.status = status
        self.store.set_source_status(self.source.id, status)


class Ticker(threading.Thread):
    """Calls fn every `every` seconds (the Claude hook watchdog)."""

    def __init__(self, fn, every: float) -> None:
        super().__init__(daemon=True, name="ticker")
        self.fn, self.every = fn, every

    def run(self) -> None:
        while True:
            time.sleep(self.every)
            try:
                self.fn()
            except Exception:  # keep ticking
                pass
