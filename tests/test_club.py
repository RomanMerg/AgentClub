import json
import tempfile
import unittest
from pathlib import Path

from club import config
from club.claude_hooks import ClaudeTracker, TranscriptUsage, pretty_model, tool_summary
from club.sources import map_n8n, map_ollama, map_trace, trace_task, credit_text, fill, check_one, short_container, summarize_containers
from club.store import Store
from club.web import host_allowed, origin_allowed, validate, public_sources

T0 = 1_800_000_000.0


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def iso(ts):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class ConfigTest(unittest.TestCase):
    def test_save_merges_keeps_comments_and_removes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env.local"
            p.write_text("# keep me\nN8N_URL=http://x:1\nOLD=1\n", encoding="utf-8")
            config.save({"N8N_API_KEY": "abc123", "OLD": None, "N8N_URL": "http://y:2"}, p)
            text = p.read_text(encoding="utf-8")
            self.assertIn("# keep me", text)
            self.assertNotIn("OLD=", text)
            self.assertEqual(config.read_env(p), {"N8N_URL": "http://y:2", "N8N_API_KEY": "abc123"})

    def test_save_rejects_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                config.save({"N8N_API_KEY": "a\nEVIL=1"}, Path(d) / ".env.local")

    def test_mask_shows_only_last_four(self):
        self.assertEqual(config.mask(""), "")
        self.assertEqual(config.mask("sk-or-1234567890"), "••••7890")


class StoreTest(unittest.TestCase):
    def test_replace_source_removes_missing_and_keeps_state_since(self):
        s = Store()
        s.replace_source("n8n", {"n8n:1": {"state": "idle"}, "n8n:2": {"state": "working"}})
        since = {a["id"]: a["stateSince"] for a in s.agents()}
        s.replace_source("n8n", {"n8n:1": {"state": "idle", "title": "x"}})
        agents = s.agents()
        self.assertEqual([a["id"] for a in agents], ["n8n:1"])
        self.assertEqual(agents[0]["stateSince"], since["n8n:1"])

    def test_wait_times_out_without_changes(self):
        s = Store()
        v, snap = s.wait(0, timeout=0.01)
        self.assertIsNone(snap)


class ClaudeHooksTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = Store()
        self.tr = ClaudeTracker(self.store, clock=self.clock)

    def ev(self, name, **kw):
        self.tr.handle({"hook_event_name": name, "session_id": "s1", "cwd": "C:/proj/my-app", **kw})

    def agent(self, aid="claude:s1"):
        return next((a for a in self.store.agents() if a["id"] == aid), None)

    def test_prompt_tool_and_stop(self):
        self.ev("UserPromptSubmit", prompt="Fix the login bug\nplease")
        a = self.agent()
        self.assertEqual((a["state"], a["name"], a["task"]), ("working", "claude · my-app", "Fix the login bug please"))
        self.ev("PreToolUse", tool_name="Edit", tool_input={"file_path": "C:/proj/backend/auth.py"})
        self.assertEqual(self.agent()["title"], "Editing auth.py")
        self.clock.t += 42
        self.ev("Stop")
        a = self.agent()
        self.assertEqual((a["state"], a["mood"], a["took"]), ("social", "brag", 42))
        self.clock.t += 91
        self.tr.tick()
        self.assertEqual(self.agent()["state"], "idle")

    def test_permission_notification_is_waiting(self):
        self.ev("UserPromptSubmit", prompt="deploy")
        self.ev("Notification", message="Claude needs your permission to use Bash")
        a = self.agent()
        self.assertEqual((a["state"], a["chip"]), ("waiting", "Waiting on you"))
        self.ev("Notification", message="Claude is waiting for your input")
        self.assertEqual(self.agent()["state"], "idle")

    def test_subagent_runs_as_its_own_agent(self):
        self.ev("UserPromptSubmit", prompt="release")
        self.ev("PreToolUse", tool_name="Task", tool_input={"subagent_type": "test-runner", "description": "Run the tests"})
        self.assertEqual((self.agent()["state"], self.agent()["mood"]), ("social", "watch"))
        child = self.agent("claude:s1:1")
        self.assertEqual((child["name"], child["state"]), ("claude · test-runner", "working"))
        self.ev("PreToolUse", tool_name="Bash", tool_input={"command": "pytest -q"})
        self.assertEqual(self.agent("claude:s1:1")["title"], "Running a command")
        self.assertEqual(self.agent()["state"], "social")  # the parent keeps waiting
        self.ev("PostToolUse", tool_name="Task", tool_response={})
        self.assertEqual(self.agent("claude:s1:1")["mood"], "brag")
        self.assertEqual(self.agent()["state"], "working")
        self.clock.t += 31
        self.tr.tick()
        self.assertIsNone(self.agent("claude:s1:1"))

    def test_three_failures_in_a_row_is_error(self):
        self.ev("UserPromptSubmit", prompt="x")
        for _ in range(3):
            self.ev("PostToolUse", tool_name="Bash", tool_response={"is_error": True})
        self.assertEqual(self.agent()["state"], "error")
        self.ev("PostToolUse", tool_name="Bash", tool_response={"stdout": "ok"})
        self.assertEqual(self.agent()["state"], "working")

    def test_long_tool_is_waiting_and_silence_is_down(self):
        self.ev("UserPromptSubmit", prompt="build")
        self.ev("PreToolUse", tool_name="Bash", tool_input={"command": "npm run build"})
        self.clock.t += 121
        self.tr.tick()
        self.assertEqual((self.agent()["state"], self.agent()["chip"]), ("waiting", "Waiting on tool"))
        self.ev("PostToolUse", tool_name="Bash", tool_response={})
        self.assertEqual(self.agent()["state"], "working")
        self.clock.t += 301
        self.tr.tick()
        self.assertEqual(self.agent()["state"], "down")

    def test_session_end_removes_and_same_folder_gets_suffix(self):
        self.ev("SessionStart")
        self.tr.handle({"hook_event_name": "SessionStart", "session_id": "abcd1234", "cwd": "C:/proj/my-app"})
        names = sorted(a["name"] for a in self.store.agents())
        self.assertEqual(names, ["claude · my-app", "claude · my-app #abcd"])
        self.ev("SessionEnd")
        self.assertIsNone(self.agent())

    def test_ignores_junk(self):
        self.tr.handle({"hello": 1})
        self.tr.handle([])
        self.assertEqual(self.store.agents(), [])

    def test_transcript_usage_dedupes_streamed_parts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            u = {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0, "output_tokens": 5}
            rows = [{"type": "assistant", "message": {"id": "m1", "model": "claude-opus-5-5", "usage": u}},
                    {"type": "assistant", "message": {"id": "m1", "model": "claude-opus-5-5", "usage": {**u, "output_tokens": 7}}},
                    {"type": "user", "message": {"content": "hi"}}]
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            t = TranscriptUsage(str(p))
            t.update()
            self.assertEqual(t.totals(), (110, 7))
            self.assertEqual(t.context_left(), "Context window: 100% left")
            self.assertEqual(pretty_model(t.model), "Opus 5.5")

    def test_helpers(self):
        self.assertEqual(pretty_model("claude-sonnet-4-5-20250929"), "Sonnet 4.5")
        self.assertEqual(tool_summary("mcp__supabase__execute_sql", {}), ("Using supabase", "execute_sql"))
        self.assertEqual(tool_summary("Grep", {"pattern": "TODO"}), ("Searching the code", "TODO"))


class SourcesTest(unittest.TestCase):
    def test_n8n_states(self):
        now = T0
        wfs = [{"id": "1", "name": "Lead sync", "active": True}, {"id": "2", "name": "Invoice OCR", "active": True},
               {"id": "3", "name": "Follow-up", "active": True}, {"id": "4", "name": "Old", "active": False},
               {"id": "5", "name": "Report", "active": True}, {"id": "6", "name": "Stuck", "active": True}]
        exs = [{"id": "10", "workflowId": "1", "status": "running", "startedAt": iso(now - 12)},
               {"id": "11", "workflowId": "2", "status": "error", "startedAt": iso(now - 70), "stoppedAt": iso(now - 60)},
               {"id": "12", "workflowId": "3", "status": "waiting", "startedAt": iso(now - 600)},
               {"id": "13", "workflowId": "5", "status": "success", "startedAt": iso(now - 40), "stoppedAt": iso(now - 30)},
               {"id": "14", "workflowId": "6", "status": "running", "startedAt": iso(now - 1000)}]
        a = map_n8n(wfs, exs, now, {"11": ("HTTP 500 from OCR", "Parse invoice")})
        self.assertEqual(a["n8n:1"]["state"], "working")
        self.assertEqual((a["n8n:2"]["state"], a["n8n:2"]["title"], a["n8n:2"]["sub"]), ("error", "HTTP 500 from OCR", "node “Parse invoice”"))
        self.assertEqual(a["n8n:3"]["state"], "waiting")
        self.assertNotIn("n8n:4", a)
        self.assertEqual((a["n8n:5"]["state"], a["n8n:5"]["mood"], a["n8n:5"]["took"]), ("social", "brag", 10))
        self.assertEqual(a["n8n:6"]["state"], "down")

    def test_trace_states(self):
        now = T0
        tr = {"id": "8f2c99", "timestamp": iso(now - 20), "input": {"messages": [{"role": "user", "content": "Price for a salon?"}]}, "output": None}
        gen = {"type": "GENERATION", "name": "calculate_roi", "model": "gpt-4o-mini", "startTime": iso(now - 5), "endTime": None,
               "usageDetails": {"input": 900, "output": 120}}
        a = map_trace(tr, [gen], now, "advisor", "OpenRouter credit: $41.20 left")
        self.assertEqual((a["state"], a["name"], a["title"], a["tokens"]), ("working", "advisor · run 8f2c", "calculate_roi", {"in": 900, "out": 120}))
        self.assertEqual(a["task"], "Customer asked: “Price for a salon?”")
        stale = map_trace(tr, [{**gen, "startTime": iso(now - 400)}], now, "advisor", None)
        self.assertEqual(stale["state"], "down")
        failed = map_trace(tr, [{**gen, "level": "ERROR", "statusMessage": "429 Too Many Requests"}], now, "advisor", None)
        self.assertEqual(failed["state"], "waiting")
        done = map_trace({**tr, "output": "answer"}, [{**gen, "endTime": iso(now - 3)}], now, "advisor", None)
        self.assertEqual((done["state"], done["mood"]), ("social", "brag"))
        self.assertIsNone(map_trace({**tr, "output": "a"}, [{**gen, "startTime": iso(now - 3700), "endTime": iso(now - 3600)}], now, "advisor", None))

    def test_ollama_states(self):
        now = T0
        self.assertEqual(map_ollama(["qwen3.5:9b"], [], False, now)["ollama:qwen3.5:9b"]["state"], "down")
        self.assertEqual(map_ollama(["qwen3.5:9b"], [], True, now)["ollama:qwen3.5:9b"]["title"], "Asleep (not loaded)")
        busy = [{"name": "qwen3.5:9b", "expires_at": iso(now + 295)}]
        self.assertEqual(map_ollama(["qwen3.5:9b"], busy, True, now)["ollama:qwen3.5:9b"]["state"], "working")
        warm = [{"name": "qwen3.5:9b", "expires_at": iso(now + 120)}]
        self.assertEqual(map_ollama(["qwen3.5:9b"], warm, True, now)["ollama:qwen3.5:9b"]["state"], "idle")

    def test_container_summary(self):
        names = ["twenty-server-1", "twenty-worker-1", "twenty-db-1"]
        healthy = {"Status": "running", "Health": {"Status": "healthy"}}
        self.assertEqual(short_container("twenty-server-1", names), "server")
        ok, problem, detail = summarize_containers(names, {"twenty-server-1": healthy, "twenty-worker-1": {"Status": "running"}, "twenty-db-1": healthy})
        self.assertEqual((ok, problem), (True, ""))
        self.assertEqual(detail, "server healthy, worker running, db healthy")
        ok, problem, _ = summarize_containers(names, {"twenty-server-1": {"Status": "running", "Health": {"Status": "starting"}},
                                                      "twenty-worker-1": {"Status": "running"}, "twenty-db-1": healthy})
        self.assertEqual((ok, problem), (None, "server starting"))
        ok, problem, _ = summarize_containers(names, {"twenty-server-1": healthy, "twenty-worker-1": {"Status": "exited"}})
        self.assertEqual((ok, problem), (False, "worker exited"))

    def test_board_line_combines_http_and_containers(self):
        item = {"name": "n8n", "docker": ["n8n-app", "n8n-app-redis"]}
        up = {"n8n-app": {"Status": "running"}, "n8n-app-redis": {"Status": "running", "Health": {"Status": "healthy"}}}
        self.assertEqual(check_one(item, {}, up)["label"], "2/2 up")
        down = check_one(item, {}, {}, "Docker isn’t running")
        self.assertEqual((down["ok"], down["label"]), (False, "Docker isn’t running"))

    def test_expected_error_counts_as_healthy(self):
        from unittest import mock
        from club.sources import check_http
        item = {"name": "DB", "url": "https://x/rest/v1/t", "expect": {"status": [200, 401], "bodyContains": "42501", "note": "db answered"}}
        with mock.patch("club.net.get", return_value=(401, b'{"code":"42501"}', 12.0)):
            r = check_http(item, {})
        self.assertEqual((r["ok"], r["code"], r["detail"]), (True, 401, "db answered"))
        with mock.patch("club.net.get", return_value=(401, b'{"message":"Invalid API key"}', 12.0)):
            self.assertFalse(check_http(item, {})["ok"])
        with mock.patch("club.net.get", return_value=(503, b"{}", 12.0)):
            self.assertFalse(check_http(item, {})["ok"])

    def test_parent_line_checks_the_api_key_and_falls_back_without_one(self):
        from unittest import mock
        from club.sources import check_http
        item = {"name": "n8n", "url": "{N8N_URL}/api/v1/workflows?limit=1", "headers": {"X-N8N-API-KEY": "{N8N_API_KEY}"},
                "auth": True, "fallback": {"url": "{N8N_URL}/healthz"}}
        cfg = {"N8N_URL": "http://localhost:5678"}
        with mock.patch("club.net.get", return_value=(200, b'{"status":"ok"}', 5.0)) as g:
            r = check_http(item, cfg)
        self.assertEqual(g.call_args[0][0], "http://localhost:5678/healthz")
        self.assertTrue(r["ok"] and r["keyless"])
        self.assertIn("only availability was checked", r["detail"])
        with mock.patch("club.net.get", return_value=(401, b"{}", 5.0)):
            r = check_http(item, {**cfg, "N8N_API_KEY": "bad"})
        self.assertEqual((r["ok"], r["label"]), (False, "API key rejected"))
        with mock.patch("club.net.get", return_value=(200, b'{"data":[]}', 5.0)):
            r = check_one({**item, "docker": ["n8n-app"]}, {**cfg, "N8N_API_KEY": "good"}, {"n8n-app": {"Status": "running"}})
        self.assertEqual((r["ok"], r["label"], r["detail"]), (True, "key ok · 1/1 up", "API key accepted; containers: n8n-app running"))

    def test_drill_down_rows_explain_unhealthy_containers(self):
        from club.sources import container_rows
        rows = container_rows(["n8n-app", "gone"], {"n8n-app": {"Status": "running", "Health": {"Status": "unhealthy",
                              "Log": [{"ExitCode": 1, "Output": "/bin/sh: curl: not found\n"}]}}})
        self.assertEqual((rows[0]["ok"], rows[0]["label"], rows[0]["detail"]), (False, "unhealthy", "health check: /bin/sh: curl: not found"))
        self.assertEqual((rows[1]["ok"], rows[1]["label"]), (False, "missing"))

    def test_small_helpers(self):
        self.assertEqual(trace_task("hello"), "hello")
        self.assertEqual(credit_text({"limit_remaining": 2.7227, "limit": 3, "usage": 0.2773}, {"total_credits": 5, "total_usage": 0.7624}),
                         "OpenRouter: key limit $2.72 of $3.00 left · account $4.24")
        self.assertEqual(credit_text({"usage": 1.5}), "OpenRouter: key used $1.50 (no limit)")
        self.assertEqual(fill("{SUPABASE_URL}/rest/v1/", {"SUPABASE_URL": "https://x"}), ("https://x/rest/v1/", []))
        self.assertEqual(fill("{SUPABASE_ANON_KEY}", {})[1], ["SUPABASE_ANON_KEY"])


class WebTest(unittest.TestCase):
    def test_host_and_origin_checks(self):
        self.assertTrue(host_allowed("localhost:8765", 8765))
        self.assertFalse(host_allowed("evil.example:8765", 8765))
        self.assertTrue(origin_allowed(None, 8765))
        self.assertTrue(origin_allowed("http://localhost:8765", 8765))
        self.assertFalse(origin_allowed("https://evil.example", 8765))

    def test_validate_keeps_saved_secret_when_blank_and_checks_urls(self):
        self.assertEqual(validate("n8n", {"url": "http://localhost:5678", "key": ""}, {}), {"N8N_URL": "http://localhost:5678"})
        with self.assertRaises(ValueError):
            validate("n8n", {"url": "javascript:alert(1)"}, {})
        with self.assertRaises(ValueError):
            validate("nope", {}, {})
        self.assertEqual(validate("openrouter", {"disconnect": True}, {}), {"OPENROUTER_API_KEY": None})

    def test_public_sources_never_leaks_keys(self):
        cfg = {**config.DEFAULTS, "N8N_API_KEY": "secret-n8n-key-9999", "LANGFUSE_SECRET_KEY": "sk-lf-abcd"}
        out = json.dumps(public_sources(cfg, {}), ensure_ascii=False)
        self.assertNotIn("secret-n8n-key", out)
        self.assertNotIn("sk-lf-abcd", out)
        self.assertIn("••••9999", out)


if __name__ == "__main__":
    unittest.main()
