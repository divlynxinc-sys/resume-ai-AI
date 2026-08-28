# Standalone image for the AI Interviews worker (LiveKit Agents). Separate from
# the AI service (../main.py): it is a long-running worker, not an HTTP server.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY . .

# Turn-detector fallback weights, so a cold start never blocks on a download.
RUN uv run -m livekit.agents download-files

CMD ["uv", "run", "python", "agent.py", "start"]
