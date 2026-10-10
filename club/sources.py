"""Pollers: n8n executions, Langfuse traces (LangGraph runs), OpenRouter credit, Ollama models, infra health.

Each source has enabled(cfg) and poll(cfg). Agent sources return {agent_id: agent}; the others return data
for the store's meta. The map_* functions are pure so they can be tested without a network.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from . import net
from .config import ROOT

NO_LLM = "n/a (no LLM steps)"


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def ago(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s}s" if s < 60 else f"{s // 60} min" if s < 3600 else f"{s // 3600} h {s % 3600 // 60} min"


def short(text: object, n: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ======================= n8n =======================
N8N_RUNNING_DOWN_S = 15 * 60
N8N_ERROR_WINDOW_S = 30 * 60
N8N_DONE_WINDOW_S = 90
N8N_RECENT_S = 2 * 3600


def map_n8n(workflows: list[dict], executions: list[dict], now: float, error_info: dict[str, tuple[str, str]]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for ex in sorted(executions, key=lambda e: parse_ts(e.get("startedAt")) or 0, reverse=True):
        latest.setdefault(str(ex.get("workflowId")), ex)
    agents: dict[str, dict] = {}
    for wf in workflows:
        wid, name, active = str(wf.get("id")), str(wf.get("name") or "workflow"), bool(wf.get("active"))
        ex = latest.get(wid)
        started = parse_ts(ex.get("startedAt")) if ex else None
        stopped = parse_ts(ex.get("stoppedAt")) if ex else None
        status = (ex or {}).get("status") or ""
        base = {"name": f"n8n · {short(name, 26)}", "src": f"n8n workflow{' (active)' if active else ''}",
                "task": f"Workflow “{name}”", "model": None, "tokens": None, "usage": NO_LLM, "mood": None, "chip": None,
                "startedAt": int((started or now) * 1000), "took": None}
        if ex and status in ("running", "new") and started:
            if now - started > N8N_RUNNING_DOWN_S:
                agent = {**base, "state": "down", "title": f"Running for {ago(now - started)}", "sub": f"execution #{ex.get('id')}"}
            else:
                agent = {**base, "state": "working", "title": f"Running execution #{ex.get('id')}", "sub": f"started {ago(now - started)} ago"}
        elif ex and status == "waiting":
            agent = {**base, "state": "waiting", "title": "Paused in a Wait node", "sub": f"execution #{ex.get('id')}", "chip": "Wait node"}
        elif ex and status in ("error", "crashed") and stopped and now - stopped < N8N_ERROR_WINDOW_S:
            msg, node = error_info.get(str(ex.get("id")), ("", ""))
            agent = {**base, "state": "error", "title": short(msg, 60) or f"Execution #{ex.get('id')} failed",
                     "sub": f"node “{node}”" if node else f"execution #{ex.get('id')}", "chip": "Execution failed"}
        elif ex and status == "success" and stopped and now - stopped < N8N_DONE_WINDOW_S:
            agent = {**base, "state": "social", "mood": "brag", "title": f"Finished execution #{ex.get('id')}",
                     "sub": f"took {ago(stopped - (started or stopped))}", "chip": "Done · bragging",
                     "took": round(stopped - (started or stopped))}
        elif active or (stopped and now - stopped < N8N_RECENT_S):
            last = f"last run {ago(now - stopped)} ago" if stopped else "no runs yet"
            agent = {**base, "state": "idle", "title": "Waiting for its trigger", "sub": last}
        else:
            continue
        agents[f"n8n:{wid}"] = agent
    return agents


class N8n:
    id = "n8n"
    interval = 5

    def __init__(self) -> None:
        self.error_cache: dict[str, tuple[str, str]] = {}
        self.workflows: list[dict] = []
        self.workflows_at = 0.0

    def enabled(self, cfg: dict) -> bool:
        return bool(cfg.get("N8N_API_KEY") and cfg.get("N8N_URL"))

    def poll(self, cfg: dict) -> dict[str, dict]:
        base, headers = cfg["N8N_URL"].rstrip("/"), {"X-N8N-API-KEY": cfg["N8N_API_KEY"], "Accept": "application/json"}
        now = time.time()
        if now - self.workflows_at > 60:
            self.workflows = (net.get_json(f"{base}/api/v1/workflows?limit=250", headers) or {}).get("data") or []
            self.workflows_at = now
        executions = (net.get_json(f"{base}/api/v1/executions?limit=100", headers) or {}).get("data") or []
        for ex in executions:
            eid = str(ex.get("id"))
            if ex.get("status") in ("error", "crashed") and eid not in self.error_cache:
                stopped = parse_ts(ex.get("stoppedAt"))
                if stopped and now - stopped < N8N_ERROR_WINDOW_S:
                    self.error_cache[eid] = self._error_detail(base, headers, eid)
        return map_n8n(self.workflows, executions, now, self.error_cache)

    @staticmethod
    def _error_detail(base: str, headers: dict, eid: str) -> tuple[str, str]:
        try:
            ex = net.get_json(f"{base}/api/v1/executions/{quote(eid)}?includeData=true", headers) or {}
        except net.NetError:
            return "", ""
        result = ((ex.get("data") or {}).get("resultData") or {})
        err = result.get("error") or {}
        return str(err.get("message") or ""), str(result.get("lastNodeExecuted") or err.get("node", {}).get("name") or "")

    def detail(self, agents: dict) -> str:
        return f"{len(agents)} workflow{'s' if len(agents) != 1 else ''} in the club"


# ======================= Langfuse (LangGraph runs) =======================
LF_WINDOW_S = 30 * 60
LF_DOWN_S = 5 * 60
LF_DONE_S = 120
LF_IDLE_S = 10 * 60
LF_MAX_RUNS = 6


def trace_task(inp: object) -> str:
    if isinstance(inp, str):
        return short(inp, 200)
    if isinstance(inp, dict):
        msgs = inp.get("messages")
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") in ("user", "human") or (isinstance(m, dict) and m.get("type") == "human"):
                    return "Customer asked: “" + short(m.get("content"), 180) + "”"
        for k in ("question", "input", "message", "query"):
            if inp.get(k):
                return short(inp[k], 200)
        return short(json.dumps(inp, ensure_ascii=False), 200)
    return "LangGraph run"


def map_trace(trace: dict, observations: list[dict], now: float, prefix: str, credit: str | None) -> dict | None:
    started = parse_ts(trace.get("timestamp")) or now
    times = [t for o in observations for t in (parse_ts(o.get("endTime")), parse_ts(o.get("startTime"))) if t]
    last_activity = max(times) if times else started
    tin = tout = 0
    model = None
    for o in observations:
        u = o.get("usageDetails") or o.get("usage") or {}
        tin += int(u.get("input") or 0)
        tout += int(u.get("output") or 0)
        if o.get("type") == "GENERATION" and o.get("model"):
            model = o["model"]
    base = {"name": f"{prefix} · run {str(trace.get('id'))[:4]}", "src": "LangGraph run (Langfuse)", "task": trace_task(trace.get("input")),
            "model": model, "tokens": {"in": tin, "out": tout} if (tin or tout) else None, "usage": credit or "—",
            "mood": None, "chip": None, "startedAt": int(started * 1000), "took": None}
    errored = [o for o in observations if o.get("level") == "ERROR"]
    latest = max(observations, key=lambda o: parse_ts(o.get("startTime")) or 0, default=None)
    if errored:
        e = errored[-1]
        msg = str(e.get("statusMessage") or "")
        if "429" in msg or "rate limit" in msg.lower():
            return {**base, "state": "waiting", "title": "Rate limited (429)", "sub": short(e.get("name"), 50), "chip": "Waiting on API"}
        return {**base, "state": "error", "title": f"{short(e.get('name'), 40)} failed", "sub": short(msg, 80), "chip": "Run failed"}
    if trace.get("output") is not None:
        took = max(0.0, last_activity - started)
        if now - last_activity < LF_DONE_S:
            return {**base, "state": "social", "mood": "brag", "title": "Answered the customer", "sub": f"took {ago(took)}",
                    "chip": "Done · bragging", "took": round(took)}
        if now - last_activity < LF_IDLE_S:
            return {**base, "state": "idle", "title": "Run finished", "sub": "", "took": round(took)}
        return None
    if now - last_activity > LF_DOWN_S:
        return {**base, "state": "down", "title": f"No new steps for {ago(now - last_activity)}",
                "sub": f"last step: {short(latest.get('name'), 40)}" if latest else ""}
    return {**base, "state": "working", "title": short(latest.get("name"), 50) if latest else "Starting",
            "sub": model or (latest or {}).get("type", "").lower()}


class Langfuse:
    id = "langfuse"
    interval = 10

    def __init__(self) -> None:
        self.final: dict[str, tuple[dict, list[dict]]] = {}
        self.credit: str | None = None

    def enabled(self, cfg: dict) -> bool:
        return bool(cfg.get("LANGFUSE_PUBLIC_KEY") and cfg.get("LANGFUSE_SECRET_KEY") and cfg.get("LANGFUSE_HOST"))

    def poll(self, cfg: dict) -> dict[str, dict]:
        host = cfg["LANGFUSE_HOST"].rstrip("/")
        headers = {"Authorization": net.basic_auth(cfg["LANGFUSE_PUBLIC_KEY"], cfg["LANGFUSE_SECRET_KEY"])}
        now = time.time()
        since = datetime.fromtimestamp(now - LF_WINDOW_S, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        traces = (net.get_json(f"{host}/api/public/traces?limit=20&fromTimestamp={quote(since)}", headers) or {}).get("data") or []
        traces = sorted(traces, key=lambda t: parse_ts(t.get("timestamp")) or 0, reverse=True)[:LF_MAX_RUNS]
        agents: dict[str, dict] = {}
        for tr in traces:
            tid = str(tr.get("id"))
            if tid in self.final:
                tr, obs = self.final[tid]
            else:
                obs = (net.get_json(f"{host}/api/public/observations?traceId={quote(tid)}&limit=100", headers) or {}).get("data") or []
                if tr.get("output") is not None:
                    self.final[tid] = (tr, obs)
            agent = map_trace(tr, obs, now, cfg.get("LANGFUSE_PREFIX") or "langgraph", self.credit)
            if agent:
                agents[f"langfuse:{tid}"] = agent
        return agents

    def detail(self, agents: dict) -> str:
        return f"{len(agents)} recent run{'s' if len(agents) != 1 else ''}"


# ======================= OpenRouter =======================
def credit_text(key: dict, credits: dict | None = None) -> str:
    """The key's own spending limit and, when available, the account balance; they are different numbers."""
    parts = []
    remaining, limit, usage = key.get("limit_remaining"), key.get("limit"), key.get("usage")
    if remaining is not None and limit is not None:
        parts.append(f"key limit ${float(remaining):.2f} of ${float(limit):.2f} left")
    elif usage is not None:
        parts.append(f"key used ${float(usage):.2f} (no limit)")
    if credits and credits.get("total_credits") is not None:
        balance = float(credits["total_credits"]) - float(credits.get("total_usage") or 0)
        parts.append(f"account ${balance:.2f}")
    return "OpenRouter: " + " · ".join(parts) if parts else "OpenRouter: connected"


class OpenRouter:
    id = "openrouter"
    interval = 60

    def enabled(self, cfg: dict) -> bool:
        return bool(cfg.get("OPENROUTER_API_KEY"))

    def poll(self, cfg: dict) -> dict:
        auth = {"Authorization": f"Bearer {cfg['OPENROUTER_API_KEY']}"}
        key = (net.get_json("https://openrouter.ai/api/v1/key", auth) or {}).get("data") or {}
        try:
            credits = (net.get_json("https://openrouter.ai/api/v1/credits", auth) or {}).get("data") or {}
        except net.NetError:
            credits = None   # some keys can't read the account balance; the key limit is still useful
        return {"credit": credit_text(key, credits), "label": key.get("label")}

    def detail(self, result: dict) -> str:
        return result.get("credit", "")


# ======================= Ollama =======================
OLLAMA_KEEP_ALIVE_S = 300
OLLAMA_BUSY_S = 20


def map_ollama(models: list[str], loaded: list[dict], reachable: bool, now: float) -> dict[str, dict]:
    agents: dict[str, dict] = {}
    by_name = {str(m.get("name") or m.get("model")): m for m in loaded}
    for name in models:
        base = {"name": f"ollama · {name}", "src": "Ollama (local model)", "task": "Answers local prompts", "model": name,
                "tokens": None, "usage": "Local · free", "mood": None, "chip": None, "startedAt": int(now * 1000), "took": None}
        if not reachable:
            agents[f"ollama:{name}"] = {**base, "state": "down", "title": "Ollama isn’t running", "sub": "start Ollama to wake it"}
            continue
        m = by_name.get(name)
        if not m:
            agents[f"ollama:{name}"] = {**base, "state": "idle", "title": "Asleep (not loaded)", "sub": ""}
            continue
        expires = parse_ts(m.get("expires_at"))
        last_use = (expires - OLLAMA_KEEP_ALIVE_S) if expires else None
        if last_use and now - last_use < OLLAMA_BUSY_S:
            agents[f"ollama:{name}"] = {**base, "state": "working", "title": "Generating", "sub": "loaded in memory"}
        else:
            left = f"unloads in {ago(expires - now)}" if expires and expires > now else ""
            agents[f"ollama:{name}"] = {**base, "state": "idle", "title": "Loaded, waiting", "sub": left}
    return agents


class Ollama:
    id = "ollama"
    interval = 5

    def enabled(self, cfg: dict) -> bool:
        return bool(cfg.get("OLLAMA_URL") and cfg.get("OLLAMA_MODELS", "").strip())

    def poll(self, cfg: dict) -> dict[str, dict]:
        models = [m.strip() for m in cfg["OLLAMA_MODELS"].split(",") if m.strip()]
        try:
            loaded = (net.get_json(cfg["OLLAMA_URL"].rstrip("/") + "/api/ps", timeout=3) or {}).get("models") or []
            reachable = True
        except net.NetError:
            loaded, reachable = [], False
        self.reachable = reachable
        return map_ollama(models, loaded, reachable, time.time())

    def detail(self, agents: dict) -> str:
        return "Ollama is running" if getattr(self, "reachable", False) else "Ollama isn’t running on this PC"


# ======================= infra health (the menu behind the bar) =======================
INFRA_FILE = ROOT / "infra.json"
EXAMPLE_FILE = ROOT / "infra.example.json"
DEFAULT_INFRA = json.loads(EXAMPLE_FILE.read_text(encoding="utf-8")) if EXAMPLE_FILE.exists() else []


class DockerDown(Exception):
    pass


def docker_states(names: list[str]) -> dict[str, dict]:
    """Container name -> docker State (Status, Health, StartedAt). Missing containers are absent."""
    if not names:
        return {}
    try:
        res = subprocess.run(["docker", "inspect", *names], capture_output=True, text=True, timeout=8,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except FileNotFoundError:
        raise DockerDown("Docker isn’t installed") from None
    except subprocess.TimeoutExpired:
        raise DockerDown("Docker didn’t answer") from None
    out = res.stdout.strip()
    if not out.startswith("["):
        raise DockerDown("Docker isn’t running")
    try:
        items = json.loads(out)
    except ValueError:
        raise DockerDown("Docker gave an unreadable answer") from None
    return {str(c.get("Name", "")).lstrip("/"): c.get("State") or {} for c in items}


def short_container(name: str, siblings: list[str]) -> str:
    """twenty-server-1 -> server (drops the prefix the group shares and the -1 replica suffix)."""
    base = re.sub(r"-\d+$", "", name)
    prefix = os.path.commonprefix([re.sub(r"-\d+$", "", n) for n in siblings]) if len(siblings) > 1 else ""
    return base[len(prefix):].lstrip("-") or base


def summarize_containers(names: list[str], states: dict[str, dict]) -> tuple[bool | None, str, str]:
    """(ok, short problem text or '', one detail line per container). ok None means 'starting'."""
    parts, problems, starting = [], [], []
    for n in names:
        label = short_container(n, names)
        st = states.get(n)
        if st is None:
            parts.append(f"{label} missing"); problems.append(f"{label} missing")
            continue
        status, health = st.get("Status", "?"), (st.get("Health") or {}).get("Status")
        if status != "running":
            parts.append(f"{label} {status}"); problems.append(f"{label} {status}")
        elif health == "unhealthy":
            parts.append(f"{label} unhealthy"); problems.append(f"{label} unhealthy")
        elif health == "starting":
            parts.append(f"{label} starting"); starting.append(f"{label} starting")
        else:
            parts.append(f"{label} {health or 'running'}")
    ok = False if problems else (None if starting else True)
    return ok, (problems or starting or [""])[0], ", ".join(parts)


def load_infra(path: Path = INFRA_FILE) -> list[dict]:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_INFRA, indent=2) + "\n", encoding="utf-8")
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return DEFAULT_INFRA
    return items if isinstance(items, list) else DEFAULT_INFRA


def fill(template: str, cfg: dict) -> tuple[str, list[str]]:
    missing = [k for k in re.findall(r"\{([A-Z0-9_]+)\}", template) if not cfg.get(k)]
    return re.sub(r"\{([A-Z0-9_]+)\}", lambda m: cfg.get(m.group(1), ""), template), missing


def check_http(item: dict, cfg: dict) -> dict:
    url, missing = fill(item.get("url", ""), cfg)
    headers = {}
    for k, v in (item.get("headers") or {}).items():
        headers[k], more = fill(v, cfg)
        missing += more
    out = {"name": item.get("name", url), "ok": None, "code": None, "ms": None, "detail": "", "checkedAt": int(time.time() * 1000)}
    if missing and item.get("fallback"):
        # e.g. no n8n API key yet: check plain availability instead and say so
        fb = check_http({**item.get("fallback"), "name": out["name"]}, cfg)
        fb["detail"] = "; ".join(x for x in (fb["detail"], f"no {', '.join(sorted(set(missing)))} yet, so only availability was checked") if x)
        fb["keyless"] = True
        return fb
    if missing:
        out["detail"] = "needs " + ", ".join(sorted(set(missing)))
        return out
    try:
        code, body, ms = net.get(url, headers, timeout=8)
    except net.NetError as e:
        out.update(ok=False, detail=str(e))
        return out
    expect = item.get("expect")
    if expect:
        # e.g. Supabase: a Postgres permission error (42501) still proves the database answered
        ok = code in expect.get("status", [200]) and (code == 200 or expect.get("bodyContains", "") in body.decode("utf-8", "replace"))
    else:
        ok = 200 <= code < 400
    out.update(ok=ok, code=code, ms=round(ms))
    if item.get("auth") and code in (401, 403):
        out.update(label="API key rejected", detail=f"HTTP {code}: the server is up but refused the key")
        return out
    if item.get("auth") and ok:
        out["detail"] = "API key accepted"
    if ok and code >= 400:
        out["detail"] = expect.get("note", "") if expect else ""
        out["label"] = f"up · {round(ms)} ms"
        return out
    if code == 530:
        out["detail"] = "Cloudflare can’t reach the origin (tunnel down?)"
    elif code >= 400:
        out["detail"] = f"HTTP {code}"
    return out


def container_rows(names: list[str], states: dict[str, dict]) -> list[dict]:
    """One drill-down row per container, with Docker's own health-check output when it is failing."""
    rows = []
    for n in names:
        st = states.get(n)
        row = {"name": n, "ok": None, "label": "", "detail": "", "code": None, "ms": None}
        if st is None:
            row.update(ok=False, label="missing", detail="no container with this name")
        else:
            status, health = st.get("Status", "?"), st.get("Health") or {}
            hs = health.get("Status")
            last = (health.get("Log") or [{}])[-1]
            if status != "running":
                row.update(ok=False, label=status, detail=f"exit code {st.get('ExitCode')}")
            elif hs == "unhealthy":
                # last non-empty line: the actual error, not a progress meter
                lines = [ln.strip() for ln in str(last.get("Output") or "").splitlines() if ln.strip()]
                out = lines[-1] if lines else ""
                row.update(ok=False, label="unhealthy", detail=f"health check: {out[:140]}" if out else "health check failing")
            elif hs == "starting":
                row.update(ok=None, label="starting")
            else:
                row.update(ok=True, label=hs or "running", detail="" if hs else "no health check configured")
        rows.append(row)
    return rows


def check_details(item: dict, cfg: dict) -> list[dict]:
    """The on-demand drill-down for one board line (nothing here runs in the background)."""
    subs = item.get("details") or []
    names = sorted({n for s in subs for n in (s.get("docker") or [])})
    try:
        states, err = docker_states(names), None
    except DockerDown as e:
        states, err = {}, str(e)
    rows: list[dict] = []
    for sub in subs:
        if sub.get("docker"):
            if err:
                rows.extend({"name": n, "ok": False, "label": err, "detail": "", "code": None, "ms": None} for n in sub["docker"])
            else:
                rows.extend(container_rows(sub["docker"], states))
        else:
            r = check_http(sub, cfg)
            r.setdefault("label", "")
            rows.append(r)
    return rows


def check_one(item: dict, cfg: dict, states: dict[str, dict] | None, docker_error: str | None = None) -> dict:
    """One board line: the HTTP check (if any) plus the item's Docker containers (if any)."""
    out = check_http(item, cfg) if item.get("url") else {"name": item.get("name", "?"), "ok": True, "code": None, "ms": None,
                                                          "detail": "", "checkedAt": int(time.time() * 1000)}
    out["hasDetails"] = bool(item.get("details"))
    names = item.get("docker") or []
    if not names:
        return out
    if docker_error:
        out.update(ok=False, label=docker_error, detail="; ".join(x for x in (out["detail"], docker_error) if x))
        return out
    c_ok, problem, detail = summarize_containers(names, states or {})
    out["detail"] = "; ".join(x for x in (out["detail"], "containers: " + detail) if x)
    if out["ok"] is False:
        return out                                   # the HTTP failure is the headline
    if c_ok is not True:
        out.update(ok=c_ok, label=problem)           # e.g. "server starting", "worker exited"
    else:
        up = len(names)
        head = "key ok" if item.get("auth") and not out.get("keyless") else out["code"]
        out["label"] = f"{head} · {up}/{up} up" if head else f"{up}/{up} up"
    return out


class Infra:
    id = "infra"
    interval = 30

    def enabled(self, cfg: dict) -> bool:
        return True

    def poll(self, cfg: dict) -> list[dict]:
        items = load_infra()
        names = sorted({n for it in items for n in (it.get("docker") or [])})
        try:
            states, err = docker_states(names), None
        except DockerDown as e:
            states, err = {}, str(e)
        return [check_one(item, cfg, states, err) for item in items]

    def detail(self, result: list[dict]) -> str:
        bad = [r["name"] for r in result if r["ok"] is False]
        return f"{len(result) - len(bad)}/{len(result)} healthy" + (f" · down: {', '.join(bad)}" if bad else "")
