# Agent Club

Your AI agents as clubgoers. Where they hang out shows what they're doing:

| State | In the club |
|---|---|
| Working | dancing |
| Blocked (waiting on you, a long command, an API) | bathroom line |
| Idle | couch, drink, shisha |
| Waiting on another agent / just finished | at the bar |
| Error | throwing up by the bin |
| Not responding | face-down on the floor, red ring |

**Guest list** (left side): untick an agent to keep them out of the club; tick to let them walk back in. Remembered per browser.

## Run it

Double-click `start.cmd` (or `python -m club --port 8765`) and open http://localhost:8765.
Python 3.11+ only, no packages to install. Without the server the page runs on demo data.

## Connect sources

Open **Connect agents** at the top of the page.

- **Claude Code**: copy the hook config from the panel into `~/.claude/settings.json`. No key.
- **n8n**: URL + API key (n8n → Settings → n8n API).
- **LangGraph via Langfuse**: host + public and secret key.
- **OpenRouter**: API key (shows credit left).
- **Ollama**: URL + model names. No key.
- **Bar menu (infra health)**: the checks are in `infra.json`. Each line can have a `url` (HTTP check), a `docker`
  list of container names (state and health via `docker inspect`), or both. Optional per line: `expect` (status codes that
  still count as up), `auth` + `fallback` (an API-key check that falls back to plain availability), and `details`: sub-checks
  that only run when you press Details on that line in the card (no background polling).

Keys are saved in `.env.local` (git-ignored). You can also edit that file directly; the server re-reads it on every poll.
Variables: `N8N_URL`, `N8N_API_KEY`, `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PREFIX`,
`OPENROUTER_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODELS`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`. The server only listens on 127.0.0.1, refuses other
hostnames and cross-site requests, and never sends keys back to the page.

## Files

- `index.html`: the whole page (Three.js scene, HUD, jukebox)
- `club/`: the server (`claude_hooks.py` hook state machine, `sources.py` pollers, `web.py` HTTP)
- `tests/`: `python -m unittest discover -s tests`
- Add `?debug` to the URL to expose the agent engine as `window.__club` for testing.
