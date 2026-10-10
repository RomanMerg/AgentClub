"""Agent Club server: python -m club [--port 8765]"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer

from .claude_hooks import ClaudeTracker
from .runner import Runner, Ticker
from .sources import Infra, Langfuse, N8n, Ollama, OpenRouter
from .store import Store
from .web import make_handler


class App:
    version = "0.1"

    def __init__(self, port: int) -> None:
        self.port = port
        self.store = Store()
        self.claude = ClaudeTracker(self.store)
        self.langfuse = Langfuse()
        self.runners = {
            "n8n": Runner(N8n(), self.store),
            "langfuse": Runner(self.langfuse, self.store),
            "openrouter": Runner(OpenRouter(), self.store, on_result=self._credit),
            "ollama": Runner(Ollama(), self.store),
            "infra": Runner(Infra(), self.store, on_result=lambda r: self.store.set_meta("infra", r)),
        }

    def _credit(self, result: dict) -> None:
        self.langfuse.credit = result.get("credit")
        self.store.set_meta("openrouter", result)

    def statuses(self) -> dict:
        return {**{k: r.status for k, r in self.runners.items()}, "claude": self.claude.status()}

    def start(self) -> None:
        for r in self.runners.values():
            r.start()

        def watchdog():
            self.claude.tick()
            self.store.set_source_status("claude", self.claude.status())

        Ticker(watchdog, 5).start()


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent Club: your agents as clubgoers")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app = App(args.port)
    app.start()
    # no address reuse: on Windows it would let two servers share the port and the old one keep answering
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    except OSError:
        raise SystemExit(f"Port {args.port} is already in use. Close the other server (an older start.cmd window?) and try again.")
    server.daemon_threads = True
    print(f"Agent Club: http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
