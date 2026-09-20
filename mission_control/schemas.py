"""The wire contract, shared by the server and the client.

One definition of every request and response body, imported by :mod:`api` to
serialise and by :mod:`client` to parse. A field can't drift between the two
because there is only one of it.

**Redaction lives here** (§5.3). Crew get :class:`MissionForCrew`, a *different
model* rather than a filtered dict — so a field added to :class:`MissionDetail`
later is not exposed to them by default. That is the direction the design wants
to fail in.

Levels cross the wire as names (``"MASTER"``) rather than integers. The number
is an implementation detail of ``Proficiency`` being an ``IntEnum`` so that
``level >= required`` reads naturally (§4.2); a client should never have to know
that Expert is 4.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .domain import (
    Assignment,
    CrewProfile,
    Member,
    Mission,
    MissionEvent,
    MissionPlan,
    Proficiency,
    Requirement,
    Skill,
    TimeWindow,
)
from .lifecycle import AllocationReport
from .matching import MatchProposal


class Request(BaseModel):
    """Requests reject unknown fields, so a typo in a body is a 422 rather than
    a silently ignored intent."""

    model_config = ConfigDict(extra="forbid")


def parse_level(value: str | int | Proficiency) -> Proficiency:
    if isinstance(value, Proficiency):
        return value
    if isinstance(value, int):
        return Proficiency(value)
    try:
        return Proficiency[str(value).strip().upper()]
    except KeyError:
        raise ValueError(
            f"unknown proficiency {value!r}; expected one of "
            + ", ".join(p.name for p in Proficiency)
        ) from None


# ------------------------------------------------------------------------ auth


class BootstrapRequest(Request):
    tenant_id: str
    name: str
    director_name: str


class TokenRequest(Request):
    member_id: str


class TokenOut(BaseModel):
    token: str
    tenant_id: str
    member_id: str
    name: str
    role: str


class MeOut(BaseModel):
    """What the server derived from your credentials — the Actor, made visible.

    Returning the permission set makes the two-layer model (§5.1) inspectable:
    this is layer one, and a 403 with a policy code is layer two.
    """

    tenant_id: str
    tenant_name: str
    member_id: str
    name: str
    role: str
    permissions: list[str]


# -------------------------------------------------------------- org and people


class MemberCreate(Request):
    name: str
    role: str


class DeactivateRequest(Request):
    reason: str


class MemberOut(BaseModel):
    id: str
    name: str
    role: str
    status: str

    @classmethod
    def of(cls, member: Member) -> MemberOut:
        return cls(
            id=member.id, name=member.name,
            role=member.role.value, status=member.status.value,
        )


class SkillCreate(Request):
    name: str
    id: str | None = None


class SkillOut(BaseModel):
    id: str
    name: str

    @classmethod
    def of(cls, skill: Skill) -> SkillOut:
        return cls(id=skill.id, name=skill.name)


class CrewSkillIn(Request):
    skill_id: str
    level: str


class SkillsRequest(Request):
    skills: list[CrewSkillIn]


class WindowIn(Request):
    start: datetime
    end: datetime


class UnavailabilityIn(Request):
    start: datetime
    end: datetime
    reason: str | None = None


class AvailabilityRequest(Request):
    unavailability: list[UnavailabilityIn]


class WindowOut(BaseModel):
    start: datetime
    end: datetime

    @classmethod
    def of(cls, window: TimeWindow) -> WindowOut:
        return cls(start=window.start, end=window.end)


class CrewSkillOut(BaseModel):
    skill_id: str
    level: str


class UnavailabilityOut(BaseModel):
    start: datetime
    end: datetime
    reason: str | None = None


class CrewProfileOut(BaseModel):
    member_id: str
    name: str
    skills: list[CrewSkillOut]
    unavailability: list[UnavailabilityOut]
    version: int

    @classmethod
    def of(cls, profile: CrewProfile, member: Member) -> CrewProfileOut:
        return cls(
            member_id=profile.member_id,
            name=member.name,
            skills=[
                CrewSkillOut(skill_id=s.skill_id, level=s.level.name)
                for s in profile.skills
            ],
            unavailability=[
                UnavailabilityOut(
                    start=b.window.start, end=b.window.end, reason=b.reason
                )
                for b in profile.unavailability
            ],
            version=profile.version,
        )


# -------------------------------------------------------------------- missions


class SkillRequirementIn(Request):
    skill_id: str
    min_level: str


class RequirementIn(Request):
    label: str
    skills: list[SkillRequirementIn]
    count: int = 1
    mandatory: bool = True
    id: str | None = None


class PlanIn(Request):
    requirements: list[RequirementIn]
    window: WindowIn
    site: str


class MissionCreate(Request):
    title: str
    plan: PlanIn
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    reference_code: str | None = None


class MetadataPatch(Request):
    """Only the metadata half of §6.1.1. Plan fields go to PUT /missions/{id}/plan
    and reach the state machine; these do not move the mission."""

    title: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    reference_code: str | None = None

    def fields_set(self) -> dict[str, Any]:
        given = self.model_dump(exclude_unset=True)
        if "tags" in given and given["tags"] is not None:
            given["tags"] = tuple(given["tags"])
        return given


class EventRequest(Request):
    reason: str | None = None


class SkillRequirementOut(BaseModel):
    skill_id: str
    min_level: str


class RequirementOut(BaseModel):
    id: str
    label: str
    count: int
    mandatory: bool
    skills: list[SkillRequirementOut]

    @classmethod
    def of(cls, requirement: Requirement) -> RequirementOut:
        return cls(
            id=requirement.id, label=requirement.label,
            count=requirement.count, mandatory=requirement.mandatory,
            skills=[
                SkillRequirementOut(skill_id=s.skill_id, min_level=s.min_level.name)
                for s in requirement.skills
            ],
        )


class PlanOut(BaseModel):
    requirements: list[RequirementOut]
    window: WindowOut
    site: str
    version: int

    @classmethod
    def of(cls, plan: MissionPlan) -> PlanOut:
        return cls(
            requirements=[RequirementOut.of(r) for r in plan.requirements],
            window=WindowOut.of(plan.window),
            site=plan.site.name,
            version=plan.version,
        )


class AssignmentOut(BaseModel):
    id: str
    mission_id: str
    member_id: str
    member_name: str | None = None
    requirement_id: str
    slot_index: int
    state: str
    offered_at: datetime
    offer_expires_at: datetime
    responded_at: datetime | None = None
    release_reason: str | None = None

    @classmethod
    def of(cls, a: Assignment, member_name: str | None = None) -> AssignmentOut:
        return cls(
            id=a.id, mission_id=a.mission_id, member_id=a.member_id,
            member_name=member_name, requirement_id=a.requirement_id,
            slot_index=a.slot_index, state=a.state.value,
            offered_at=a.offered_at, offer_expires_at=a.offer_expires_at,
            responded_at=a.responded_at, release_reason=a.release_reason,
        )


class MissionSummary(BaseModel):
    id: str
    title: str
    state: str
    window: WindowOut
    site: str
    crewing_status: str

    @classmethod
    def of(cls, mission: Mission, crewing: str) -> MissionSummary:
        return cls(
            id=mission.id, title=mission.title, state=mission.state.value,
            window=WindowOut.of(mission.plan.window),
            site=mission.plan.site.name, crewing_status=crewing,
        )


class MissionDetail(MissionSummary):
    """The full view, for anyone holding MISSION_VIEW_ALL."""

    plan: PlanOut
    description: str
    tags: list[str]
    notes: str
    reference_code: str | None
    created_by: str
    submitted_by: str | None
    approved_by: str | None
    approved_plan_version: int | None
    pending_expires_at: datetime | None
    closed_reason: str | None
    version: int
    roster: list[AssignmentOut]
    available_events: list[str]
    blocked_events: dict[str, str]
    """What this actor may do but cannot do *yet*, and why — enough for a client
    to grey a button out and explain it (§6.4)."""


class MissionForCrew(MissionSummary):
    """The redacted view (§5.3).

    Requirements and window, not other members' details: no owner, no approver,
    no roster beyond the reader's own assignment. A separate model rather than a
    filtered dict, so a field added to MissionDetail is withheld by default.
    """

    plan: PlanOut
    my_assignment: AssignmentOut | None = None


# -------------------------------------------------------------------- matching


class SkillEvidenceOut(BaseModel):
    skill_id: str
    required: str
    held: str


class SlotProposalOut(BaseModel):
    requirement_id: str
    slot_index: int
    label: str
    member_id: str
    member_name: str
    rank: int
    recent_load: int
    pinned: bool
    skills: list[SkillEvidenceOut]
    alternates: list[str]


class UnfilledSlotOut(BaseModel):
    requirement_id: str
    slot_index: int
    label: str
    mandatory: bool
    reason: str


class NearMissOut(BaseModel):
    member_id: str
    member_name: str
    requirement_id: str
    filter: str
    detail: str


class MatchProposalOut(BaseModel):
    """No score anywhere in here, deliberately (§7.4). Each slot carries the
    facts behind it — the member's level against each required skill, and their
    recent load — because the engine has no opinion beyond "clears the bar"."""

    mission_id: str
    generated_at: datetime
    is_complete: bool
    slots: list[SlotProposalOut]
    unfilled: list[UnfilledSlotOut]
    near_misses: list[NearMissOut]
    team_warnings: list[str]

    @classmethod
    def of(cls, proposal: MatchProposal) -> MatchProposalOut:
        return cls(
            mission_id=proposal.mission_id,
            generated_at=proposal.generated_at,
            is_complete=proposal.is_complete,
            slots=[
                SlotProposalOut(
                    requirement_id=s.requirement_id, slot_index=s.slot_index,
                    label=s.label, member_id=s.member_id, member_name=s.member_name,
                    rank=s.rank, recent_load=s.recent_load, pinned=s.pinned,
                    skills=[
                        SkillEvidenceOut(
                            skill_id=e.skill_id,
                            required=e.required.name,
                            held=e.held.name,
                        )
                        for e in s.skills
                    ],
                    alternates=list(s.alternates),
                )
                for s in proposal.slots
            ],
            unfilled=[
                UnfilledSlotOut(
                    requirement_id=u.requirement_id, slot_index=u.slot_index,
                    label=u.label, mandatory=u.mandatory, reason=u.reason,
                )
                for u in proposal.unfilled
            ],
            near_misses=[
                NearMissOut(
                    member_id=n.member_id, member_name=n.member_name,
                    requirement_id=n.requirement_id, filter=n.filter, detail=n.detail,
                )
                for n in proposal.near_misses
            ],
            team_warnings=list(proposal.team_warnings),
        )


class OfferRow(Request):
    requirement_id: str
    slot_index: int
    member_id: str


class OfferRequest(Request):
    """The Lead's decision, stated explicitly.

    Not a proposal echoed back: re-running the matcher server-side could offer
    people the Lead never saw, and trusting a client-supplied proposal means
    trusting client-supplied rankings. The matcher proposes; a human picks
    (§7.9).
    """

    offers: list[OfferRow]


# --------------------------------------------------------------- reports, audit


class StaleSlotOut(BaseModel):
    requirement_id: str
    slot_index: int
    member_id: str
    mandatory: bool
    reasons: list[str]


class AllocationReportOut(BaseModel):
    valid_slots: int
    stale: list[StaleSlotOut]
    blocking: bool

    @classmethod
    def of(cls, report: AllocationReport) -> AllocationReportOut:
        return cls(
            valid_slots=len(report.valid),
            stale=[
                StaleSlotOut(
                    requirement_id=s.slot[0], slot_index=s.slot[1],
                    member_id=s.member_id, mandatory=s.mandatory,
                    reasons=list(s.reasons),
                )
                for s in report.stale
            ],
            blocking=bool(report.blocking),
        )


class EventOut(BaseModel):
    at: datetime
    actor_id: str
    actor_role: str
    event: str
    subject_kind: str
    subject_id: str
    from_state: str | None
    to_state: str
    reason: str | None
    plan_version: int | None
    metadata: dict[str, Any]

    @classmethod
    def of(cls, event: MissionEvent) -> EventOut:
        return cls(
            at=event.at, actor_id=event.actor_id, actor_role=event.actor_role.value,
            event=event.event, subject_kind=event.subject_kind,
            subject_id=event.subject_id, from_state=event.from_state,
            to_state=event.to_state, reason=event.reason,
            plan_version=event.plan_version, metadata=dict(event.metadata),
        )


class ErrorOut(BaseModel):
    """Every failure comes back in this shape.

    ``error`` is a stable machine-readable code — ``SELF_APPROVAL`` rather than
    just "403" — because §6.4's whole point is that "you lack the permission"
    and "you hold it but this case is denied" are different answers. ``detail``
    carries whatever the specific error knows, including the ``available``
    events on an illegal transition.
    """

    model_config = ConfigDict(extra="allow")

    error: str
    message: str
