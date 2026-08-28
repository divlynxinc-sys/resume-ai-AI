"""
Latency bench: LLM time-to-first-token per candidate model, on the real
interviewer prompt, text mode, three turns each. Not a test — run it when
choosing INTERVIEW_LLM_MODEL:

    uv run python bench_models.py
    uv run python bench_models.py openai/gpt-4.1-mini google/gemini-2.5-flash-lite
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from typing import Any, Dict, List

from livekit.agents import AgentSession

from agent import Interviewer
from backend_client import InterviewContext
from tests.test_interviewer import RESUME

DEFAULT_MODELS = [
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
    "openai/gpt-oss-120b",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3-flash",
    "xai/grok-4-1-fast-non-reasoning",
]

TURNS = [
    "Hi, I'm Ayesha. I've spent four years on React and TypeScript products.",
    "At Nimbus Labs I led the Flightdeck dashboard rebuild and cut load time from six seconds to under two.",
    "The hardest part was convincing the backend team to paginate the flight list API. I built a prototype to show the difference.",
]


def _ttft_of(history: Any) -> List[float]:
    from agent import turn_latency

    return [lat["llm_ttft"] for lat in (turn_latency(item) for item in history.items) if lat and lat["llm_ttft"] > 0]


async def bench(model: str) -> Dict[str, Any]:
    ctx = InterviewContext(
        session_id="bench",
        role_title="Frontend Developer",
        interview_type="general",
        seniority="mid",
        duration_minutes=15,
        question_target=4,
        candidate_name="Ayesha",
        resume=RESUME,
        job_description="Own the React design system and improve Core Web Vitals across the customer portal.",
    )
    ttfts: List[float] = []
    replies: List[str] = []
    started = time.perf_counter()
    try:
        async with AgentSession() as session:
            await session.start(Interviewer(ctx, on_finish=None, llm_model=model), capture_run=True)
            for turn in TURNS:
                result = await session.run(user_input=turn)
                for ev in result.events:
                    item = getattr(ev, "item", None)
                    if item is not None and getattr(item, "role", None) == "assistant":
                        replies.append(item.text_content or "")
            await asyncio.sleep(0.5)
            ttfts = _ttft_of(session.history)
    except Exception as e:  # noqa: BLE001
        return {"model": model, "error": str(e)[:160]}
    return {
        "model": model,
        "turns": len(ttfts),
        "mean": statistics.fmean(ttfts) if ttfts else None,
        "max": max(ttfts) if ttfts else None,
        "wall": time.perf_counter() - started,
        "sample": replies[-1][:140] if replies else "",
        "multi_question_turns": sum(1 for r in replies if r.count("?") > 1),
    }


async def main(models: List[str]) -> None:
    rows = [await bench(m) for m in models]
    print("\nmodel                              turns  mean_ttft  max_ttft  wall   >1 question turns")
    for r in rows:
        if "error" in r or not r.get("turns"):
            print(f"{r['model']:<34} ERROR {r.get('error') or 'no completed turns (timeouts?)'}")
            continue
        print(f"{r['model']:<34} {r['turns']:>5}  {r['mean']:>8.2f}s {r['max']:>8.2f}s {r['wall']:>5.1f}s  {r['multi_question_turns']}")
    for r in rows:
        if "sample" in r:
            print(f"\n[{r['model']}] last reply: {r['sample']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or DEFAULT_MODELS))
