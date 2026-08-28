# JobSynk AI Interviews — interviewer worker

The live voice interviewer for `/ai-interviews`. A [LiveKit Agents](https://docs.livekit.io/agents/) worker
that joins the candidate's room, runs a spoken mock interview grounded in their résumé and job description,
and posts the transcript back to `resumeai-backend`, which builds the report.

```
browser ──(LiveKit room, audio)──▶ this worker ──▶ LiveKit Inference (STT · LLM · TTS)
   │                                    │
   └── POST /interviews/{id}/start ─────┼── GET  /internal/interviews/{id}/context   (briefing)
       (backend mints token + dispatch) └── POST /internal/interviews/{id}/finalize  (transcript)
```

- `agent.py` — the `Interviewer` agent, turn handling, timekeeper, `end_interview` tool, worker entry point
- `prompts.py` — instructions (stages, seniority/type calibration, grounding rules), timekeeper notes
- `backend_client.py` — the two worker-only backend calls (shared secret in `X-Interview-Agent-Key`)
- `tests/` — text-mode smoke tests against the real LLM (grounding, role discipline, tool ending, latency)

## Setup

```powershell
uv sync
uv run -m livekit.agents download-files     # local turn-detector fallback weights
copy .env.example .env.local                # then fill it in
```

`.env.local`

| Var | Meaning |
|---|---|
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Same LiveKit Cloud project the backend mints tokens for |
| `JOBSYNK_BACKEND_URL` | Backend base URL (`http://localhost:8010` locally) |
| `INTERVIEW_AGENT_SECRET` | Must equal the backend's `INTERVIEW_AGENT_SECRET` |
| `INTERVIEW_AGENT_NAME` | Dispatch name, default `jobsynk-interviewer` — must equal the backend's `INTERVIEW_AGENT_NAME` |
| `INTERVIEW_LLM_MODEL` / `INTERVIEW_STT_MODEL` / `INTERVIEW_TTS_MODEL` / `INTERVIEW_TTS_VOICE` | Optional overrides (LiveKit Inference model ids) |

## Run

```powershell
uv run python agent.py dev        # register with LiveKit Cloud; the backend dispatches it per interview
uv run python agent.py console    # talk to Sam from the terminal (uses a fake session id → needs the backend)
uv run python agent.py start      # production mode
uv run pytest -s                  # smoke tests (real LLM calls, ~1 minute)
```

The worker must be running for interviews to work: with an explicit `agent_name`, LiveKit only dispatches it
when a room token carries a matching `RoomAgentDispatch` (which `POST /interviews/{id}/start` adds).

## Model choice

`bench_models.py` on the real interviewer prompt, 4 turns, text mode (LLM time-to-first-token only —
STT/TTS are excluded), measured 2026-08-28. None of these asked more than one question per turn:

| Model | mean TTFT | max TTFT |
|---|---|---|
| `google/gemini-2.5-flash-lite` | 1.22 s | 1.37 s |
| **`openai/gpt-4.1-mini`** (default) | **1.32 s** | 1.51 s |
| `openai/gpt-4.1-nano` | 1.45 s | 2.60 s |
| `openai/gpt-oss-120b` | 1.46 s | 1.76 s |
| `google/gemini-3-flash` | 1.65 s | 2.08 s |
| `xai/grok-4-1-fast-non-reasoning` | 3.35 s | 3.88 s |

`gpt-4.1-mini` is the default: within 0.1 s of the fastest and the strongest at the rules that matter here
(one question per turn, never coach the candidate, never invent résumé facts) — those are what the smoke
tests assert. Swap with `INTERVIEW_LLM_MODEL` and re-run `uv run pytest -s` before trusting a new model.

## Latency & turn-taking

- Semantic end-of-turn (`inference.TurnDetector`) with dynamic endpointing 0.7–6 s: a thinking pause does not end
  the candidate's turn; a finished sentence does.
- Adaptive interruption + `resume_false_interruption`: back-channels ("mhm") don't stop Sam; if Sam starts on a
  false end-of-turn and the candidate keeps talking, Sam yields and resumes later.
- Preemptive generation: the reply is generated while the turn detector is still confirming, so speech starts
  right after the candidate stops. `tests/test_interviewer.py::test_llm_latency_budget` enforces the LLM
  time-to-first-token budget; STT/TTS/EOU latencies are logged per turn (`latency ...` lines).

## Deploy

`Dockerfile` builds a standalone worker image (Railway/Fly/any container host). Give it the env vars above,
no ports needed. Scale by running more replicas.
