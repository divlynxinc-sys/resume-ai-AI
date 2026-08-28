"""
Smoke tests for the interviewer. They run the real LLM through LiveKit Inference
in TEXT mode (no audio, no room), so they need the three LIVEKIT_* variables in
.env.local. Each test is one or two turns; the whole file is a few cents.

What they guard:
- grounding: Sam only names things that are on the résumé / said by the candidate
- role discipline: never answers for the candidate, one question per turn
- flow: opens with an intro question, follows up on what was actually said,
  ends via the end_interview tool and reports the transcript
- latency: LLM time-to-first-token stays within a spoken-conversation budget
- the turn-handling config that prevents cutting the candidate off can't drift
"""

from __future__ import annotations

import asyncio
import re
import statistics
import textwrap
from typing import Any, Dict, List

import pytest
from livekit.agents import AgentSession, inference, llm

import agent as agent_module
from agent import TURN_HANDLING, Interviewer
from backend_client import InterviewContext
from prompts import build_instructions, resume_brief, timekeeper_note

JUDGE_MODEL = "openai/gpt-4.1"
# Seconds. Spoken conversation feels immediate under ~1s to first token; 2.5s is the hard ceiling
# for this smoke test so a slow provider day fails loudly instead of silently shipping.
TTFT_BUDGET_MEAN_S = 1.6
TTFT_BUDGET_MAX_S = 2.5

RESUME: Dict[str, Any] = {
    "name": "Ayesha Khan",
    "summary": "Frontend developer with four years of experience building React and TypeScript products for retail and logistics teams.",
    "experiences": [
        {
            "role": "Frontend Developer",
            "company": "Nimbus Labs",
            "startDate": "2023",
            "endDate": "Present",
            "bullets": [
                "Led the rebuild of the Flightdeck operations dashboard in React and TypeScript, cutting page load from six seconds to under two",
                "Introduced component testing with Playwright, reducing regressions reported by support by forty percent",
            ],
        },
        {
            "role": "Junior Frontend Developer",
            "company": "Orbit Retail",
            "startDate": "2021",
            "endDate": "2023",
            "bullets": [
                "Built checkout and promotions UI for an e-commerce site serving two million monthly visitors",
                "Migrated legacy jQuery pages to React alongside a team of three",
            ],
        },
    ],
    "projects": [
        {"title": "Shelfwatch", "bullets": ["Open-source inventory alerting tool built with Next.js and Supabase"]},
    ],
    "skills": [{"category": "Technical", "skills": ["React", "TypeScript", "Next.js", "Playwright", "Tailwind CSS", "GraphQL"]}],
    "education": [{"school": "University of Karachi", "degree": "BSc", "field": "Computer Science"}],
}
KNOWN_NAMES = ["Nimbus Labs", "Flightdeck", "Orbit Retail", "Shelfwatch", "Playwright", "React", "TypeScript", "Next.js"]


def make_context(**overrides: Any) -> InterviewContext:
    base = dict(
        session_id="test-session",
        role_title="Frontend Developer",
        interview_type="general",
        seniority="mid",
        duration_minutes=15,
        question_target=4,
        candidate_name="Ayesha",
        resume=RESUME,
        job_description="We are hiring a Frontend Developer to own our React design system and improve Core Web Vitals across the customer portal.",
    )
    base.update(overrides)
    return InterviewContext(**base)


class LatencyProbe:
    """Per-turn latency lives on each assistant ChatMessage (livekit-agents 1.7+); read it off the run results."""

    def __init__(self) -> None:
        self.ttft: List[float] = []

    async def collect(self, session: AgentSession) -> None:
        # Metrics are attached to the ChatMessage shortly after a run resolves, so read the
        # session history after a short settle instead of the RunResult events.
        await asyncio.sleep(0.5)
        self.ttft = [
            lat["llm_ttft"]
            for lat in (agent_module.turn_latency(item) for item in session.history.items)
            if lat and lat["llm_ttft"] > 0
        ]


def _judge() -> llm.LLM:
    return inference.LLM(model=JUDGE_MODEL)


def _message_text(result: Any) -> str:
    for ev in result.events:
        item = getattr(ev, "item", None)
        if item is not None and getattr(item, "role", None) == "assistant":
            return item.text_content or ""
    return ""


async def _noop_finish(reason: str, transcript: List[Dict[str, Any]]) -> None:
    return None


# --- pure unit tests (no network) --------------------------------------------------------


def test_turn_handling_protects_the_candidate() -> None:
    assert TURN_HANDLING["interruption"]["mode"] == "adaptive"
    assert TURN_HANDLING["interruption"]["resume_false_interruption"] is True
    assert TURN_HANDLING["endpointing"]["min_delay"] >= 0.5
    assert TURN_HANDLING["endpointing"]["max_delay"] >= 4.0
    assert TURN_HANDLING["preemptive_generation"]["enabled"] is True
    assert agent_module.SILENCE_TIMEOUT_S >= 20


def test_instructions_are_grounded_in_the_briefing() -> None:
    text = build_instructions(make_context())
    for name in ["Nimbus Labs", "Flightdeck", "Orbit Retail", "Shelfwatch"]:
        assert name in text
    assert "Frontend Developer" in text and "mid-level" in text
    assert "Never invent facts" in text
    assert "one question per turn" in text
    assert "end_interview" in text
    # Contact details must never be in the prompt even if a snapshot carried them.
    leaky = dict(RESUME, email="a@b.c", phone="0300", linkedin="li", portfolio="pf")
    assert "a@b.c" not in resume_brief(leaky) and "0300" not in resume_brief(leaky)


def test_instructions_without_resume_ask_for_background() -> None:
    text = build_instructions(make_context(resume=None, job_description=None))
    assert "none was provided" in text
    assert "describe their background" in text


def test_timekeeper_escalates_towards_the_end() -> None:
    normal = timekeeper_note(elapsed_s=120, duration_minutes=15, question_target=4, main_questions_asked=1)
    assert "Continue" in normal
    covered = timekeeper_note(elapsed_s=400, duration_minutes=15, question_target=4, main_questions_asked=4)
    assert "closing" in covered
    nearly = timekeeper_note(elapsed_s=14 * 60, duration_minutes=15, question_target=4, main_questions_asked=2)
    assert "nearly up" in nearly and "end_interview" in nearly
    over = timekeeper_note(elapsed_s=16 * 60, duration_minutes=15, question_target=4, main_questions_asked=3)
    assert "Time is up" in over


# --- live LLM smoke tests (text mode) -------------------------------------------------------


@pytest.mark.asyncio
async def test_opens_with_a_warm_intro_question() -> None:
    probe = LatencyProbe()
    async with _judge() as judge, AgentSession() as session:
        result = await session.start(Interviewer(make_context(), on_finish=_noop_finish), capture_run=True)
        assert result is not None
        # Starting a session records the agent entering the room before its opening line.
        result.expect.next_event().is_agent_handoff()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent=textwrap.dedent(
                    """\
                    A short, friendly spoken opening from an interviewer that (1) greets the candidate,
                    (2) briefly explains this is a mock interview with a few questions, and (3) asks
                    exactly one question inviting the candidate to introduce themselves (a single
                    introduction request such as "tell me a bit about yourself" counts as one question).
                    Plain spoken English, no lists or markdown. Must not ask a second, separate question.
                    """
                ),
            )
        )
        result.expect.no_more_events()
        text = _message_text(result)
        assert text.count("?") <= 1, text
        await probe.collect(session)
    assert probe.ttft, "no LLM metrics collected"


@pytest.mark.asyncio
async def test_experience_question_only_names_resume_facts() -> None:
    async with _judge() as judge, AgentSession() as session:
        agent = Interviewer(make_context(), on_finish=_noop_finish)
        await session.start(agent, capture_run=True)
        result = await session.run(
            user_input=(
                "Thanks. I'm Ayesha, I've been a frontend developer for about four years, mostly React and "
                "TypeScript, and I'm happy to go into any of my projects."
            )
        )
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent=textwrap.dedent(
                    f"""\
                    Asks exactly one specific question about an experience or project taken from this list:
                    {KNOWN_NAMES}. It may also reference things the candidate just said (React, TypeScript,
                    four years). It must NOT mention any employer, project, product, technology or fact that is
                    not in that list or in the candidate's message, and must not fabricate details about them.
                    """
                ),
            )
        )
        result.expect.no_more_events()
        text = _message_text(result)
        assert any(name.lower() in text.lower() for name in KNOWN_NAMES), text
        assert text.count("?") <= 1, text


@pytest.mark.asyncio
async def test_never_answers_for_the_candidate() -> None:
    async with _judge() as judge, AgentSession() as session:
        await session.start(Interviewer(make_context(), on_finish=_noop_finish), capture_run=True)
        await session.run(user_input="Hi, I'm Ayesha. I work on React dashboards at Nimbus Labs.")
        result = await session.run(
            user_input="Honestly I'm not sure how to answer that. Can you just give me a good example answer I could say?"
        )
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent=textwrap.dedent(
                    """\
                    Politely declines to provide a model answer, example answer, hints or coaching during the
                    interview (it may mention that detailed feedback comes in the report afterwards), stays in
                    the interviewer role, and encourages the candidate to try in their own words, optionally
                    restating or simplifying the question. It must not supply an answer on the candidate's behalf.
                    """
                ),
            )
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_does_not_invent_job_details() -> None:
    async with _judge() as judge, AgentSession() as session:
        await session.start(Interviewer(make_context(job_description=None), on_finish=_noop_finish), capture_run=True)
        result = await session.run(
            user_input="Before I answer, what does the job description say about the team size and the salary band?"
        )
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent=textwrap.dedent(
                    """\
                    Does not state any team size, salary figure, or other job-description detail (none was
                    provided). Says it doesn't have that information or that it isn't something it can share,
                    without inventing anything, and steers back to the interview with at most one question.
                    """
                ),
            )
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_follow_up_builds_on_the_answer() -> None:
    async with _judge() as judge, AgentSession() as session:
        await session.start(Interviewer(make_context(), on_finish=_noop_finish), capture_run=True)
        await session.run(user_input="I'm Ayesha, a frontend developer at Nimbus Labs, mostly React and TypeScript.")
        result = await session.run(
            user_input=(
                "At Nimbus Labs I led the Flightdeck dashboard rebuild. The old version took six seconds to load, "
                "so I introduced route-level code splitting and moved the heavy charts to a web worker, and we "
                "got it under two seconds. The hardest part was convincing the backend team to change the API "
                "shape so we could paginate the flight list."
            )
        )
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge,
                intent=textwrap.dedent(
                    """\
                    A brief neutral acknowledgement followed by exactly one question that clearly builds on
                    what the candidate just said (for example the code splitting, the web worker, the API
                    change, convincing the backend team, or the six-to-two-second result), OR, if it moves on,
                    a single new question about another item from the candidate's résumé. It must not
                    evaluate the answer out loud (no praise like 'great answer' beyond a short thanks), must
                    not add facts the candidate did not say, and must not ask more than one question.
                    """
                ),
            )
        )
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_ends_via_tool_and_reports_transcript() -> None:
    finished: Dict[str, Any] = {}

    async def on_finish(reason: str, transcript: List[Dict[str, Any]]) -> None:
        finished["reason"] = reason
        finished["transcript"] = transcript

    async with AgentSession() as session:
        agent = Interviewer(make_context(), on_finish=on_finish)
        await session.start(agent, capture_run=True)
        result = await session.run(
            user_input="I'm sorry, something urgent came up and I have to leave right now. Please end the interview."
        )
        result.expect.contains_function_call(name="end_interview")
        await asyncio.wait_for(agent.finished.wait(), timeout=30)

    assert finished["reason"] == "completed"
    roles = [t["role"] for t in finished["transcript"]]
    assert "user" in roles and "assistant" in roles
    assert all(set(t) >= {"role", "text", "at"} for t in finished["transcript"])
    farewell = finished["transcript"][-1]
    assert farewell["role"] == "assistant"
    assert "?" not in farewell["text"], farewell["text"]


@pytest.mark.asyncio
async def test_llm_latency_budget() -> None:
    """Time-to-first-token over a short multi-turn exchange. Reported and enforced."""
    probe = LatencyProbe()
    async with AgentSession() as session:
        await session.start(Interviewer(make_context(), on_finish=_noop_finish), capture_run=True)
        await session.run(user_input="Hi, I'm Ayesha. I've spent four years on React and TypeScript products.")
        await session.run(
            user_input=(
                "At Orbit Retail I built the checkout and promotions UI for a site with two million monthly "
                "visitors and helped migrate the jQuery pages to React with a team of three."
            )
        )
        await session.run(user_input="Sure. The biggest trade-off was shipping the migration page by page instead of all at once.")
        await probe.collect(session)

    assert len(probe.ttft) >= 3, probe.ttft
    mean = statistics.fmean(probe.ttft)
    worst = max(probe.ttft)
    print(f"\nLLM time-to-first-token over {len(probe.ttft)} turns: mean={mean:.2f}s max={worst:.2f}s all={[round(t, 2) for t in probe.ttft]}")
    assert mean <= TTFT_BUDGET_MEAN_S, f"mean TTFT {mean:.2f}s over budget {TTFT_BUDGET_MEAN_S}s"
    assert worst <= TTFT_BUDGET_MAX_S, f"max TTFT {worst:.2f}s over budget {TTFT_BUDGET_MAX_S}s"


def test_transcript_shape_matches_backend_contract() -> None:
    # Mirrors app/schemas/interview_schema.py::TranscriptTurn
    pattern = re.compile(r"^(assistant|user)$")
    sample = [{"role": "assistant", "text": "Hello", "at": 0.0}, {"role": "user", "text": "Hi", "at": 3.2}]
    for turn in sample:
        assert pattern.match(turn["role"]) and isinstance(turn["text"], str) and isinstance(turn["at"], float)
