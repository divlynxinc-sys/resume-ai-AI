"""
JobSynk AI Interviews — the live interviewer.

One LiveKit Agents worker, dispatched by name (`jobsynk-interviewer`) whenever
the backend mints a room token for an interview. Pipeline: LiveKit Inference
STT -> LLM -> TTS, semantic end-of-turn detection so the candidate is never cut
off mid-thought, and preemptive generation so the reply starts the moment they
finish. When the interview ends the transcript is posted back to the backend,
which builds the report; no audio is ever stored.

Run:  uv run python agent.py dev      (connect to LiveKit Cloud)
      uv run python agent.py console  (talk to it from this terminal)
      uv run pytest                   (text-mode smoke tests, real LLM)

All LiveKit APIs used here were checked against the installed livekit-agents
1.6.x package, not recalled from memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    StopResponse,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    llm,
    room_io,
)

from backend_client import BackendClient, InterviewContext
from prompts import (
    CANDIDATE_LEFT_REASON,
    HARD_STOP_MESSAGE,
    OPENING_INSTRUCTION,
    SILENCE_CHECKIN,
    build_instructions,
    timekeeper_note,
)

load_dotenv(".env.local")
logger = logging.getLogger("interview-agent")

AGENT_NAME = os.getenv("INTERVIEW_AGENT_NAME", "jobsynk-interviewer")
LLM_MODEL = os.getenv("INTERVIEW_LLM_MODEL", "openai/gpt-4.1-mini")
STT_MODEL = os.getenv("INTERVIEW_STT_MODEL", "assemblyai/universal-3-5-pro")
TTS_MODEL = os.getenv("INTERVIEW_TTS_MODEL", "fishaudio/s2.1-pro")
TTS_VOICE = os.getenv("INTERVIEW_TTS_VOICE", "fa4c9eb3dccc4806b382b40d61c6b10a")

# Low temperature: the interviewer should be consistent and stick to the briefing.
LLM_TEMPERATURE = 0.35
# Seconds past the planned length before the interviewer stops the interview itself.
HARD_STOP_GRACE_S = 90
# Silence before the interviewer checks in (user_state -> "away"), and how many times per interview.
SILENCE_TIMEOUT_S = 25.0
MAX_SILENCE_CHECKINS = 2

# Turn handling is the part that decides "does the candidate feel cut off". These
# values are also asserted by tests/test_interviewer.py so they can't drift silently.
TURN_HANDLING: Dict[str, Any] = {
    # Semantic end-of-turn model: waits through mid-sentence pauses, ends the turn on real completion.
    "turn_detection": "livekit-turn-detector",
    "endpointing": {"mode": "dynamic", "min_delay": 0.7, "max_delay": 6.0},
    # Adaptive: "mhm" / "right" while Sam is talking is not an interruption; a real one is.
    # If the candidate keeps talking after a false end-of-turn, Sam stops and lets them finish.
    "interruption": {"mode": "adaptive", "resume_false_interruption": True, "false_interruption_timeout": 2.0},
    # Start generating the reply while the turn detector is still confirming end of turn.
    "preemptive_generation": {"enabled": True},
}

FinishCallback = Callable[[str, List[Dict[str, Any]]], Awaitable[None]]


class Interviewer(Agent):
    def __init__(
        self,
        context: InterviewContext,
        *,
        on_finish: Optional[FinishCallback] = None,
        llm_model: str = LLM_MODEL,
    ) -> None:
        super().__init__(
            instructions=build_instructions(context),
            llm=inference.LLM(model=llm_model, extra_kwargs={"temperature": LLM_TEMPERATURE}),
        )
        self.context = context
        self._on_finish = on_finish
        self._started_mono = time.monotonic()
        self._started_wall = time.time()
        self._finished = asyncio.Event()
        self._finish_reason: Optional[str] = None
        self._finish_lock = asyncio.Lock()
        self._timer_task: Optional[asyncio.Task[None]] = None
        self._silence_checkins = 0

    # --- lifecycle -----------------------------------------------------------------

    @property
    def finished(self) -> asyncio.Event:
        return self._finished

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started_mono

    async def on_enter(self) -> None:
        self._started_mono = time.monotonic()
        self._started_wall = time.time()
        self._timer_task = asyncio.create_task(self._hard_stop_timer())
        self.session.generate_reply(instructions=OPENING_INSTRUCTION)

    async def on_exit(self) -> None:
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        # turn_ctx is the per-generation copy, so this note steers only this reply and never
        # accumulates in the persisted history.
        turn_ctx.add_message(
            role="system",
            content=timekeeper_note(
                elapsed_s=self.elapsed_s,
                duration_minutes=self.context.duration_minutes,
                question_target=self.context.question_target,
                main_questions_asked=self.main_questions_asked(),
            ),
        )

    # --- tools ------------------------------------------------------------------------

    @function_tool
    async def end_interview(self, context: RunContext, farewell: str) -> None:
        """End the interview. Call this only at the closing stage: after the candidate has had a chance to add
        anything, when the timekeeper says time is up, or when the candidate asks to stop.

        Args:
            farewell: The exact one to three sentence goodbye to say out loud. Thank the candidate and
                mention that their feedback report will be ready in a moment. Do not ask a question.
        """
        asyncio.create_task(self._close_out(context.speech_handle, farewell))
        # Nothing else should be generated after this tool: the farewell is spoken by _close_out.
        raise StopResponse()

    async def _close_out(self, tool_handle: Any, farewell: str) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(tool_handle.wait_for_playout(), timeout=15)
        with contextlib.suppress(Exception):
            speech = self.session.say(farewell, allow_interruptions=False)
            await asyncio.wait_for(speech.wait_for_playout(), timeout=60)
        await self.finish("completed")

    async def _hard_stop_timer(self) -> None:
        await asyncio.sleep(self.context.duration_minutes * 60 + HARD_STOP_GRACE_S)
        if self._finished.is_set():
            return
        logger.info("hard stop reached for %s", self.context.session_id)
        with contextlib.suppress(Exception):
            current = self.session.current_speech
            if current is not None:
                await asyncio.wait_for(current.wait_for_playout(), timeout=30)
            speech = self.session.say(HARD_STOP_MESSAGE, allow_interruptions=False)
            await asyncio.wait_for(speech.wait_for_playout(), timeout=60)
        await self.finish("time_limit")

    async def silence_checkin(self) -> None:
        if self._finished.is_set() or self._silence_checkins >= MAX_SILENCE_CHECKINS:
            return
        if self.session.agent_state not in ("listening", "idle"):
            return
        self._silence_checkins += 1
        self.session.generate_reply(instructions=SILENCE_CHECKIN)

    async def finish(self, reason: str) -> None:
        async with self._finish_lock:
            if self._finished.is_set():
                return
            self._finish_reason = reason
            self._finished.set()
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        with contextlib.suppress(Exception):
            logger.info("interview %s ended (%s); %s", self.context.session_id, reason, format_latency(latency_summary(self.session.history)))
        if self._on_finish is None:
            return
        try:
            await self._on_finish(reason, self.transcript())
        except Exception:  # noqa: BLE001
            logger.exception("on_finish failed for %s", self.context.session_id)

    # --- transcript -------------------------------------------------------------------

    def transcript(self) -> List[Dict[str, Any]]:
        """Conversation so far as [{role, text, at}], from the session's own history."""
        out: List[Dict[str, Any]] = []
        for item in self.session.history.items:
            if getattr(item, "type", None) != "message":
                continue
            role = getattr(item, "role", None)
            if role not in ("user", "assistant"):
                continue
            text = (item.text_content or "").strip()
            if not text:
                continue
            created = float(getattr(item, "created_at", self._started_wall) or self._started_wall)
            out.append({"role": role, "text": text, "at": round(max(0.0, created - self._started_wall), 1)})
        return out

    def main_questions_asked(self) -> int:
        """Rough count: assistant turns containing a question, excluding the opening."""
        asked = sum(1 for t in self.transcript() if t["role"] == "assistant" and "?" in t["text"])
        return max(0, asked - 1)


# --- session wiring ----------------------------------------------------------------------


def _noise_cancellation() -> Any:
    try:
        from livekit.plugins import ai_coustics

        return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S)
    except Exception:  # noqa: BLE001 — optional; the interview works without it
        logger.info("ai_coustics noise cancellation unavailable, continuing without it")
        return None


def build_session() -> AgentSession:
    return AgentSession(
        stt=inference.STT(model=STT_MODEL, language="en"),
        tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing=TURN_HANDLING["endpointing"],
            interruption=TURN_HANDLING["interruption"],
            preemptive_generation=TURN_HANDLING["preemptive_generation"],
        ),
        user_away_timeout=SILENCE_TIMEOUT_S,
    )


def _session_id_from(ctx: JobContext) -> str:
    metadata = getattr(ctx.job, "metadata", "") or ""
    with contextlib.suppress(Exception):
        data = json.loads(metadata)
        if isinstance(data, dict) and data.get("session_id"):
            return str(data["session_id"])
    room = ctx.room.name or ""
    if room.startswith("interview-"):
        return room[len("interview-"):]
    raise RuntimeError("no interview session id in job metadata or room name")


def turn_latency(item: Any) -> Optional[Dict[str, float]]:
    """
    Per-turn latency report attached to each assistant ChatMessage (livekit-agents 1.7+).

    `MetricsReport` behaves like a mapping and only carries the fields that applied to
    that turn — in text mode (tests) that is `llm_node_ttft` alone, in a real room the
    audio ones too. Returns None when the item is not a measured assistant turn.
    """
    if getattr(item, "role", None) != "assistant":
        return None
    m = getattr(item, "metrics", None)
    if m is None:
        return None
    if not isinstance(m, dict):
        m = m.model_dump() if hasattr(m, "model_dump") else dict(m)
    if not m:
        return None

    def val(key: str) -> float:
        try:
            return float(m.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "end_of_turn": val("end_of_turn_delay"),
        "transcription": val("transcription_delay"),
        "llm_ttft": val("llm_node_ttft"),
        "tts_ttfb": val("tts_node_ttfb"),
        "e2e": val("e2e_latency"),
    }


def latency_summary(history: llm.ChatContext) -> Optional[Dict[str, float]]:
    """Mean per-turn latencies over the whole conversation (metrics are final once a turn has played out)."""
    turns = [lat for lat in (turn_latency(item) for item in history.items) if lat]
    if not turns:
        return None
    keys = ("end_of_turn", "transcription", "llm_ttft", "tts_ttfb", "e2e")
    summary = {k: sum(t[k] for t in turns) / len(turns) for k in keys}
    summary["turns"] = float(len(turns))
    summary["e2e_max"] = max(t["e2e"] for t in turns)
    return summary


def format_latency(summary: Optional[Dict[str, float]]) -> str:
    if not summary:
        return "latency: no turns"
    return (
        f"latency over {int(summary['turns'])} turns: e2e mean={summary['e2e']:.2f}s max={summary['e2e_max']:.2f}s "
        f"(end_of_turn={summary['end_of_turn']:.2f}s transcription={summary['transcription']:.2f}s "
        f"llm_ttft={summary['llm_ttft']:.2f}s tts_ttfb={summary['tts_ttfb']:.2f}s)"
    )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def interview(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    session_id = _session_id_from(ctx)
    backend = BackendClient.from_env()

    try:
        context = await backend.fetch_context(session_id)
    except Exception as e:  # noqa: BLE001
        logger.error("could not load interview %s: %s", session_id, e)
        ctx.shutdown(reason="interview context unavailable")
        return

    async def on_finish(reason: str, transcript: List[Dict[str, Any]]) -> None:
        try:
            await backend.finalize(session_id, transcript, reason)
        finally:
            with contextlib.suppress(Exception):
                await ctx.delete_room()

    agent = Interviewer(context, on_finish=on_finish)
    session = build_session()

    @session.on("user_state_changed")
    def _on_user_state(ev: Any) -> None:
        if ev.new_state == "away":
            asyncio.create_task(agent.silence_checkin())

    async def _on_shutdown(reason: str) -> None:
        # The candidate closed the tab / lost connection before Sam wrapped up.
        await agent.finish(CANDIDATE_LEFT_REASON)

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=_noise_cancellation()),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
