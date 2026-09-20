"""A typed Python client for the HTTP API.

Parses responses into the same :mod:`mission_control.schemas` models the server
serialises from, so the contract is defined once and neither side can drift
from it.

Used by the CLI, and importable directly for scripts::

    mc = MissionControl(token=...)
    proposal = mc.run_matcher("mis-0001")
    mc.offer_proposal("mis-0001", proposal)

Errors arrive as :class:`ApiError`, which keeps the machine-readable code and
whatever the specific failure knew — notably ``available`` on an illegal
transition, so a caller can react rather than just report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

import httpx

from . import schemas as s


class ApiError(Exception):
    """A non-2xx response, with the server's error code preserved.

    ``code`` is the stable identifier — ``SELF_APPROVAL``, ``NOT_FOUND``,
    ``ILLEGAL_TRANSITION`` — because §6.4's whole point is that "you lack the
    permission" and "you hold it but this case is denied" are different
    answers, and a status code alone cannot tell them apart.
    """

    def __init__(self, status: int, payload: Mapping[str, Any]) -> None:
        self.status = status
        self.code = str(payload.get("error", "ERROR"))
        self.message = str(payload.get("message", ""))
        self.detail = dict(payload)
        super().__init__(f"{self.code}: {self.message}")

    @property
    def available(self) -> list[str]:
        """Events that *would* have been legal, on an illegal transition."""
        return list(self.detail.get("available", []))

    @property
    def reason(self) -> str | None:
        """Why a precondition failed, on a guard failure."""
        got = self.detail.get("reason")
        return str(got) if got else None


def make_plan(
    requirements: Sequence[s.RequirementIn],
    *,
    start: datetime,
    end: datetime,
    site: str,
) -> s.PlanIn:
    return s.PlanIn(
        requirements=list(requirements),
        window=s.WindowIn(start=start, end=end),
        site=site,
    )


def requirement(
    label: str,
    skills: Mapping[str, str],
    *,
    count: int = 1,
    mandatory: bool = True,
    id: str | None = None,
) -> s.RequirementIn:
    """``requirement("Pilot", {"pilot": "PROFICIENT"})``.

    All the skills of **one person** — a Flight Surgeon needing Medicine and
    Physiology is one requirement with two entries, not two requirements (§7.1).
    """
    return s.RequirementIn(
        label=label,
        skills=[
            s.SkillRequirementIn(skill_id=k, min_level=v) for k, v in skills.items()
        ],
        count=count,
        mandatory=mandatory,
        id=id,
    )


class MissionControl:
    """One client, one identity. :meth:`as_actor` gives you another."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        token: str | None = None,
        *,
        http: httpx.Client | None = None,
        simulated_now: datetime | None = None,
        timeout: float = 30.0,
    ) -> None:
        """``http`` supplies a pre-built transport when you have one.

        Tests pass Starlette's ``TestClient`` — itself an ``httpx.Client`` — so
        the whole stack runs in-process with no server to start and no port to
        pick. Real callers can pass a client configured with retries, proxies
        or mutual TLS. Left unset, a plain one is built.
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.simulated_now = simulated_now
        self._http = http or httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- plumbing

    def as_actor(self, token: str) -> MissionControl:
        """A client for a different identity, over the same connection settings.

        Scripts need this constantly: a demo offers as the Lead and accepts as
        five different crew members.
        """
        clone = MissionControl.__new__(MissionControl)
        clone.base_url = self.base_url
        clone.token = token
        clone.simulated_now = self.simulated_now
        clone._http = self._http
        return clone

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.simulated_now is not None:
            headers["X-Simulated-Now"] = self.simulated_now.isoformat()
        return headers

    def _call(self, method: str, path: str, **kwargs) -> Any:
        response = self._http.request(
            method, path, headers=self._headers(), **kwargs
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = {"error": "HTTP_ERROR", "message": response.text}
            raise ApiError(response.status_code, payload)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MissionControl:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- meta and auth

    def health(self) -> dict:
        return self._call("GET", "/health")

    def me(self) -> s.MeOut:
        return s.MeOut.model_validate(self._call("GET", "/me"))

    def bootstrap(self, tenant_id: str, name: str, director_name: str) -> s.TokenOut:
        return s.TokenOut.model_validate(
            self._call(
                "POST", "/bootstrap",
                json={"tenant_id": tenant_id, "name": name, "director_name": director_name},
            )
        )

    def token_for(self, member_id: str) -> s.TokenOut:
        """Mint a token for another member. Dev-only server-side."""
        return s.TokenOut.model_validate(
            self._call("POST", "/tokens", json={"member_id": member_id})
        )

    # -- organisation

    def members(self) -> list[s.MemberOut]:
        return [s.MemberOut.model_validate(m) for m in self._call("GET", "/members")]

    def add_member(self, name: str, role: str) -> s.MemberOut:
        return s.MemberOut.model_validate(
            self._call("POST", "/members", json={"name": name, "role": role})
        )

    def deactivate_member(self, member_id: str, reason: str) -> s.MemberOut:
        return s.MemberOut.model_validate(
            self._call(
                "POST", f"/members/{member_id}/deactivate", json={"reason": reason}
            )
        )

    def skills(self) -> list[s.SkillOut]:
        return [s.SkillOut.model_validate(x) for x in self._call("GET", "/skills")]

    def add_skill(self, name: str, skill_id: str | None = None) -> s.SkillOut:
        return s.SkillOut.model_validate(
            self._call("POST", "/skills", json={"name": name, "id": skill_id})
        )

    def audit(self) -> list[s.EventOut]:
        return [s.EventOut.model_validate(e) for e in self._call("GET", "/audit")]

    # -- crew

    def crew_profile(self, member_id: str) -> s.CrewProfileOut:
        return s.CrewProfileOut.model_validate(self._call("GET", f"/crew/{member_id}"))

    def set_skills(self, member_id: str, skills: Mapping[str, str]) -> s.CrewProfileOut:
        return s.CrewProfileOut.model_validate(
            self._call(
                "PUT", f"/crew/{member_id}/skills",
                json={"skills": [{"skill_id": k, "level": v} for k, v in skills.items()]},
            )
        )

    def set_availability(
        self, member_id: str, blocks: Iterable[tuple[datetime, datetime, str | None]]
    ) -> s.CrewProfileOut:
        return s.CrewProfileOut.model_validate(
            self._call(
                "PUT", f"/crew/{member_id}/availability",
                json={
                    "unavailability": [
                        {"start": a.isoformat(), "end": b.isoformat(), "reason": why}
                        for a, b, why in blocks
                    ]
                },
            )
        )

    # -- missions

    def missions(self) -> list[s.MissionSummary]:
        return [
            s.MissionSummary.model_validate(m) for m in self._call("GET", "/missions")
        ]

    def mission(self, mission_id: str) -> s.MissionDetail | s.MissionForCrew:
        return _mission(self._call("GET", f"/missions/{mission_id}"))

    def create_mission(
        self, title: str, plan: s.PlanIn, **metadata: Any
    ) -> s.MissionDetail | s.MissionForCrew:
        body = {"title": title, "plan": plan.model_dump(mode="json"), **metadata}
        return _mission(self._call("POST", "/missions", json=body))

    def update_plan(
        self, mission_id: str, plan: s.PlanIn
    ) -> s.MissionDetail | s.MissionForCrew:
        return _mission(
            self._call(
                "PUT", f"/missions/{mission_id}/plan", json=plan.model_dump(mode="json")
            )
        )

    def update_metadata(
        self, mission_id: str, **fields: Any
    ) -> s.MissionDetail | s.MissionForCrew:
        return _mission(self._call("PATCH", f"/missions/{mission_id}", json=fields))

    def fire(
        self, mission_id: str, event: str, reason: str | None = None
    ) -> s.MissionDetail | s.MissionForCrew:
        return _mission(
            self._call(
                "POST", f"/missions/{mission_id}/events/{event}", json={"reason": reason}
            )
        )

    def submit(self, mission_id: str):
        return self.fire(mission_id, "submit")

    def reopen(self, mission_id: str):
        return self.fire(mission_id, "reopen")

    def approve(self, mission_id: str):
        return self.fire(mission_id, "approve")

    def request_changes(self, mission_id: str, reason: str):
        return self.fire(mission_id, "request_changes", reason)

    def activate(self, mission_id: str):
        return self.fire(mission_id, "activate")

    def complete(self, mission_id: str, reason: str | None = None):
        return self.fire(mission_id, "complete", reason)

    def cancel(self, mission_id: str, reason: str):
        return self.fire(mission_id, "cancel", reason)

    def abort(self, mission_id: str, reason: str):
        return self.fire(mission_id, "abort", reason)

    def roster(self, mission_id: str) -> list[s.AssignmentOut]:
        return [
            s.AssignmentOut.model_validate(a)
            for a in self._call("GET", f"/missions/{mission_id}/roster")
        ]

    def allocation_report(self, mission_id: str) -> s.AllocationReportOut:
        return s.AllocationReportOut.model_validate(
            self._call("GET", f"/missions/{mission_id}/allocation-report")
        )

    # -- matching

    def run_matcher(self, mission_id: str) -> s.MatchProposalOut:
        return s.MatchProposalOut.model_validate(
            self._call("POST", f"/missions/{mission_id}/match")
        )

    def offer(self, mission_id: str, rows: Sequence[s.OfferRow]) -> list[s.AssignmentOut]:
        return [
            s.AssignmentOut.model_validate(a)
            for a in self._call(
                "POST", f"/missions/{mission_id}/offers",
                json={"offers": [r.model_dump(mode="json") for r in rows]},
            )
        ]

    def offer_proposal(
        self,
        mission_id: str,
        proposal: s.MatchProposalOut,
        *,
        slots: Iterable[tuple[str, int]] | None = None,
    ) -> list[s.AssignmentOut]:
        """Turn proposal rows into offers — the Lead's act, not the matcher's.

        Pinned rows are skipped: those people already accepted, and re-offering
        would discard the consent pinning exists to preserve (§6.7).
        """
        wanted = set(slots) if slots is not None else None
        rows = [
            s.OfferRow(
                requirement_id=row.requirement_id,
                slot_index=row.slot_index,
                member_id=row.member_id,
            )
            for row in proposal.slots
            if not row.pinned
            and (wanted is None or (row.requirement_id, row.slot_index) in wanted)
        ]
        return self.offer(mission_id, rows) if rows else []

    # -- assignments

    def assignments(self, mission_id: str | None = None) -> list[s.AssignmentOut]:
        params = {"mission_id": mission_id} if mission_id else None
        return [
            s.AssignmentOut.model_validate(a)
            for a in self._call("GET", "/assignments", params=params)
        ]

    def accept(self, assignment_id: str) -> s.AssignmentOut:
        return s.AssignmentOut.model_validate(
            self._call("POST", f"/assignments/{assignment_id}/accept")
        )

    def decline(self, assignment_id: str) -> s.AssignmentOut:
        return s.AssignmentOut.model_validate(
            self._call("POST", f"/assignments/{assignment_id}/decline")
        )

    # -- system sweeps

    def expire_offers(self) -> list[s.AssignmentOut]:
        return [
            s.AssignmentOut.model_validate(a)
            for a in self._call("POST", "/system/expire-offers")
        ]

    def expire_approvals(self) -> list[s.MissionSummary]:
        return [
            s.MissionSummary.model_validate(m)
            for m in self._call("POST", "/system/expire-approvals")
        ]


def _mission(payload: Mapping[str, Any]) -> s.MissionDetail | s.MissionForCrew:
    """Pick the right model for what came back.

    Crew receive the redacted shape (§5.3), so the client cannot simply assume
    MissionDetail — and shouldn't pretend the missing fields are ``None``.
    """
    if "available_events" in payload:
        return s.MissionDetail.model_validate(payload)
    return s.MissionForCrew.model_validate(payload)
