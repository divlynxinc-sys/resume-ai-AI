# ResumeAI AI Service

Single-file FastAPI service (`main.py`, ~1300 lines) that does all LLM generation. Called only by `../resumeai-backend` (never directly by the frontend). See the frontend CLAUDE.md for the cross-repo overview.

## LLM backends

Two modes, switched by env (`USE_HOSTED_LLM = bool(LLM_API_KEY)`):

- **Hosted (prod, Railway)**: OpenAI-compatible Chat Completions API. `LLM_API_KEY` (or `GROQ_API_KEY`), `LLM_BASE_URL` (default `https://api.groq.com/openai/v1`), `LLM_MODEL` (default **`openai/gpt-oss-120b`** since 2026-08-28 — Groq retired `llama-3.3-70b-versatile`, which now 404s; update the Railway env too), `LLM_TIMEOUT` (120s). No GPU needed.
- **Local (docker-compose dev)**: Ollama. `OLLAMA_API`, `OLLAMA_MODEL` (default `qwen2.5:7b-instruct-q4_K_M`), `OLLAMA_TIMEOUT` (600s).

There is **no token counting or usage metering here** — quotas are enforced upstream in the backend (`enforce_usage_limit`).

## Endpoints

- `POST /generate_resume` — non-streaming JSON: optimizes a structured resume against a job description, returns `{resume, ats_final_result}` (ATS scoring included). This is what the backend's `POST /resumes/{id}/ai/optimize` calls.
- `POST /generate_cover_letter` — streams plain-text chunks.
- `POST /generate_qa_answers` — streams; interview Q&A from job description (resume optional).
- `POST /generate_hr_email` — streams; N email drafts from context (resume optional).
- `POST /interview/report` — non-streaming JSON (added 2026-08-28): transcript of a live AI interview → per-question scores/feedback + overall report; `overall_score` computed in Python with the shared weights. Called by the backend's `utils/interviews.py::generate_report`.
- File parsing helpers use `pdfplumber` / `python-docx` for uploaded resumes.

## `interview_agent/` — the live interviewer (separate process)

A LiveKit Agents worker (uv project: `uv sync`, `uv run python agent.py dev|start`, `uv run pytest -s`) that joins the candidate's LiveKit room and conducts the spoken interview (STT/LLM/TTS via LiveKit Inference). Prompts in `interview_agent/prompts.py`, turn-taking/latency config in `agent.py::TURN_HANDLING`, backend calls in `backend_client.py`. It is the 4th deployable (own `Dockerfile`); it must be running under `INTERVIEW_AGENT_NAME=jobsynk-interviewer` with the same LiveKit project as the backend. Never verify its LiveKit APIs from memory — introspect the installed `livekit-agents` (`.venv`) first.

## Conventions

- Pydantic models at the top of `main.py` define the resume shape (`Experience`, `Project`, `Education`, ...). The backend's `resume_ai_adapter.py` must stay in sync with these — `job_description` is required by `ResumeRequest` even when unused (callers set it to `""`).
- Prompts are inline f-strings in each endpoint; streaming responses yield raw text chunks (no SSE framing).
