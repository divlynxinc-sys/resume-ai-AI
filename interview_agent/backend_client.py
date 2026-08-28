"""
HTTP client for the two worker-only backend endpoints
(resumeai-backend/app/routers/interviews.py, `internal_router`).

Authenticated with the INTERVIEW_AGENT_SECRET shared secret. `finalize` retries
because a lost transcript means a lost interview.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("interview-agent.backend")


@dataclass
class InterviewContext:
    session_id: str
    role_title: str
    interview_type: str = "general"
    seniority: str = "mid"
    duration_minutes: int = 15
    question_target: int = 4
    candidate_name: str = ""
    resume: Optional[Dict[str, Any]] = None
    job_description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, session_id: str, data: Dict[str, Any]) -> "InterviewContext":
        return cls(
            session_id=session_id,
            role_title=str(data.get("role_title") or "the role"),
            interview_type=str(data.get("interview_type") or "general"),
            seniority=str(data.get("seniority") or "mid"),
            duration_minutes=int(data.get("duration_minutes") or 15),
            question_target=int(data.get("question_target") or 4),
            candidate_name=str(data.get("candidate_name") or ""),
            resume=data.get("resume") if isinstance(data.get("resume"), dict) else None,
            job_description=data.get("job_description") or None,
        )


class BackendClient:
    def __init__(self, base_url: str, secret: str, *, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.secret = secret
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "BackendClient":
        base = os.getenv("JOBSYNK_BACKEND_URL", "http://localhost:8010")
        secret = os.getenv("INTERVIEW_AGENT_SECRET", "")
        if not secret:
            raise RuntimeError("INTERVIEW_AGENT_SECRET is not set")
        return cls(base, secret)

    def _headers(self) -> Dict[str, str]:
        return {"X-Interview-Agent-Key": self.secret}

    async def fetch_context(self, session_id: str) -> InterviewContext:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/internal/interviews/{session_id}/context", headers=self._headers())
            r.raise_for_status()
            return InterviewContext.from_payload(session_id, r.json())

    async def finalize(
        self,
        session_id: str,
        transcript: List[Dict[str, Any]],
        ended_reason: str,
        *,
        note: str | None = None,
        attempts: int = 4,
    ) -> Dict[str, Any]:
        payload = {"transcript": transcript, "ended_reason": ended_reason, "note": note}
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    r = await client.post(
                        f"{self.base_url}/internal/interviews/{session_id}/finalize",
                        headers=self._headers(),
                        json=payload,
                    )
                    r.raise_for_status()
                    return r.json()
            except Exception as e:  # noqa: BLE001 — retry everything, this is the one call that must land
                last_error = e
                logger.warning("finalize attempt %s/%s failed for %s: %s", attempt, attempts, session_id, e)
                await asyncio.sleep(min(8.0, 1.5 * attempt))
        raise RuntimeError(f"could not finalize interview {session_id}: {last_error}")
