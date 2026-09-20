"""Domain types: ids, enums, value objects, errors, and the eligibility predicate.

Every aggregate here is a frozen dataclass. Mutations produce a new object with
``version + 1`` and the store swaps the dict entry (§2). Three things fall out:
optimistic concurrency via version compare-and-swap, reads that cannot mutate
platform state, and cheap snapshot reads (§3.3).

Nothing in this module imports from the rest of the package.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, IntEnum
from typing import Any, NewType, Protocol

# --------------------------------------------------------------------------- ids
#
# Free at runtime, and they make get_mission(crew_id) a type error. Ids are
# unique *within* a tenant, not globally — which is why the test fixture seeds
# two tenants with deliberately colliding ids (§10).

TenantId = NewType("TenantId", str)
MemberId = NewType("MemberId", str)
SkillId = NewType("SkillId", str)
MissionId = NewType("MissionId", str)
AssignmentId = NewType("AssignmentId", str)
RequirementId = NewType("RequirementId", str)

UTC = timezone.utc
"""**Every datetime in this system is timezone-aware UTC.**

Stated because the alternative — naive datetimes meaning "UTC by convention" —
works right up until one aware value enters from outside and every comparison
raises ``can't compare offset-naive and offset-aware``. An HTTP body is exactly
where that happens, so the boundary coerces rather than trusting callers, and
``TimeWindow`` coerces again in case anything slips past.
"""

EPOCH = datetime.min.replace(tzinfo=UTC)
"""Sorts before every real timestamp. Used for "never flown" (§7.4)."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def as_utc(moment: datetime) -> datetime:
    """Coerce to aware UTC, reading a naive value as UTC rather than local time.

    Naive-means-local is the other classic bug in this area: it makes behaviour
    depend on the server's timezone.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


SlotKey = tuple[RequirementId, int]
"""Which slot: (requirement, index within that requirement's ``count``).

Slot identity is what lets a re-match pin an accepted member to the *same* slot
rather than merely keeping them on the mission (§6.7).
"""


# ------------------------------------------------------------------------- enums


class Role(str, Enum):
    """One role per member. Multi-role would let one person hold both submit and
    approve and quietly defeat separation of duties (§4.1)."""

    DIRECTOR = "director"
    MISSION_LEAD = "mission_lead"
    CREW = "crew"


class MemberStatus(str, Enum):
    ACTIVE = "active"
    DEACTIVATED = "deactivated"


class Proficiency(IntEnum):
    """IntEnum so ``level >= required`` reads naturally; an enum rather than a
    bare int so a level is self-describing and cannot be set to 7 (§4.2)."""

    NOVICE = 1
    COMPETENT = 2
    PROFICIENT = 3
    EXPERT = 4
    MASTER = 5


class MissionState(str, Enum):
    """Authorisation states. Crew responses live on the assignment (§8)."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


TERMINAL_MISSION_STATES = frozenset(
    {MissionState.COMPLETED, MissionState.CANCELLED, MissionState.ABORTED}
)
PRE_ACTIVE_MISSION_STATES = frozenset(
    {MissionState.DRAFT, MissionState.PENDING_APPROVAL, MissionState.APPROVED}
)


class AssignmentState(str, Enum):
    OFFERED = "offered"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    RELEASED = "released"
    COMPLETED = "completed"
    PARTIAL = "partial"


LIVE_ASSIGNMENT_STATES = frozenset({AssignmentState.OFFERED, AssignmentState.ACCEPTED})
HISTORY_ASSIGNMENT_STATES = frozenset({AssignmentState.COMPLETED, AssignmentState.PARTIAL})


class CrewingStatus(str, Enum):
    """A projection over assignments, derived and never stored (§8.3). It gates
    ``submit`` and ``activate``; MissionState stays about authorisation."""

    UNDER_CREWED = "under_crewed"
    AWAITING_RESPONSES = "awaiting_responses"
    FULLY_CREWED = "fully_crewed"


# ------------------------------------------------------------------------ errors


class DomainError(Exception):
    """Base for everything this package raises deliberately."""


class NotFound(DomainError):
    """An id that does not exist *in this tenant*.

    Raised for foreign ids too, deliberately: Forbidden would confirm the record
    exists somewhere, which is an enumeration oracle (§3.2).
    """

    def __init__(self, kind: str, id_: str) -> None:
        super().__init__(f"{kind} {id_!r} not found")
        self.kind = kind
        self.id = id_


class Denied(DomainError):
    """The actor may not do this.

    ``code`` distinguishes "you lack the permission" from "you hold it but this
    case is denied", so the UI can say *"you cannot approve a mission you
    submitted — ask another Director"* rather than a generic 403 (§6.4).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class GuardFailed(DomainError):
    """The actor may do this, but a domain precondition is unmet.

    Separate from Denied because the remedy is different: change the data, not
    the caller.
    """

    def __init__(self, event: str, reason: str) -> None:
        super().__init__(f"cannot {event}: {reason}")
        self.event = event
        self.reason = reason


class IllegalTransition(DomainError):
    """No transition exists from this state for this event.

    Carries the events that *would* be legal for this actor, which costs nothing
    because ``available_events`` already exists (§6.4).
    """

    def __init__(self, kind: str, state: str, event: str, available: Sequence[str]) -> None:
        options = ", ".join(available) if available else "nothing"
        super().__init__(
            f"cannot {event} a {state} {kind}; you can: {options}"
        )
        self.kind = kind
        self.state = state
        self.event = event
        self.available = tuple(available)


class StaleVersion(DomainError):
    """Compare-and-swap failed: someone else wrote first (§3.3)."""

    def __init__(self, kind: str, id_: str, expected: int, found: int) -> None:
        super().__init__(
            f"{kind} {id_!r} changed underneath you (expected v{expected}, found v{found})"
        )
        self.kind = kind
        self.id = id_
        self.expected = expected
        self.found = found


# ----------------------------------------------------------------- time & skills


@dataclass(frozen=True, order=True)
class TimeWindow:
    """A half-open interval ``[start, end)``.

    Half-open so back-to-back windows do not overlap: a mission ending 09:00
    does not conflict with one starting 09:00.

    ``overlaps`` is the only operation needed. Because availability is a boolean
    rather than a fraction (§4.3), there is no interval subtraction, no duration
    measurement, and no ratio to keep in range.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", as_utc(self.start))
        object.__setattr__(self, "end", as_utc(self.end))
        if self.end <= self.start:
            raise ValueError(f"window end {self.end!r} is not after start {self.start!r}")

    def overlaps(self, other: TimeWindow) -> bool:
        return self.start < other.end and other.start < self.end

    def __str__(self) -> str:
        return f"{self.start:%Y-%m-%d}..{self.end:%Y-%m-%d}"


@dataclass(frozen=True)
class UnavailabilityBlock:
    """Crew-declared. Commitments cannot express leave, training, medical
    grounding or part-time patterns — someone on holiday is on no mission at
    all (§4.3)."""

    window: TimeWindow
    reason: str | None = None
    """Display only; never interpreted. An engine that branched on the reason
    would be making a judgement it has no basis for."""


@dataclass(frozen=True)
class Skill:
    """A tenant taxonomy entry. A space agency and a research lab share no
    vocabulary, so each tenant defines its own (§4.2)."""

    id: SkillId
    name: str


@dataclass(frozen=True)
class CrewSkill:
    """A crew member's claim to a skill. No hierarchy, no certification expiry,
    no recency decay — which is what makes skill matching a plain integer
    comparison with no temporal input (§4.2)."""

    skill_id: SkillId
    level: Proficiency


# ----------------------------------------------------------------------- members


@dataclass(frozen=True)
class Member:
    id: MemberId
    name: str
    role: Role
    status: MemberStatus = MemberStatus.ACTIVE
    version: int = 1


@dataclass(frozen=True)
class CrewProfile:
    """The extra data the Crew role carries.

    Keyed by MemberId rather than an id of its own: a crew member *is* a member
    holding the Crew role, which avoids the "member 47 and crew 12 are the same
    person" bug class (§2).
    """

    member_id: MemberId
    skills: tuple[CrewSkill, ...] = ()
    unavailability: tuple[UnavailabilityBlock, ...] = ()
    version: int = 1

    def level(self, skill_id: SkillId) -> Proficiency | None:
        for skill in self.skills:
            if skill.skill_id == skill_id:
                return skill.level
        return None


# ---------------------------------------------------------------------- missions


@dataclass(frozen=True)
class Location:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class SkillRequirement:
    skill_id: SkillId
    min_level: Proficiency


@dataclass(frozen=True)
class Requirement:
    """One specification, and how many people are wanted at it.

    Three orthogonal kinds of multiplicity (§7.1), which is why ``skills`` is a
    tuple:

    * different roles on a mission → several ``Requirement`` objects
    * several people of the same role → ``count``
    * one person holding several skills → the ``skills`` tuple

    A Flight Surgeon needing ``Medicine >= Expert`` *and*
    ``Physiology >= Proficient`` is one person with both. Split it into two
    requirements and you have asked for two people.

    Differing levels mean differing requirements: three Surgeons at Expert and
    two at Competent is *two* requirements, because ``mandatory`` is
    per-specification too.
    """

    id: RequirementId
    label: str
    count: int
    skills: tuple[SkillRequirement, ...]
    mandatory: bool = True
    """Unfilled mandatory slots block submission and activation. An unfilled
    optional slot is dropped and noted (§6.8)."""

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"requirement {self.id!r} needs count >= 1, got {self.count}")
        if not self.skills:
            raise ValueError(f"requirement {self.id!r} specifies no skills")

    def slots(self) -> Iterator[SlotKey]:
        for index in range(self.count):
            yield (self.id, index)


@dataclass(frozen=True)
class MissionPlan:
    """What a Director approves. Replacing this object is what emits
    ``plan_edited`` (§6.1.1).

    Enforced structurally: plan fields live in here, so the only way to change
    one is to replace the plan — which is the operation that emits the event. A
    field added to MissionPlan later is covered automatically; one added to
    Mission is metadata by default.

    **The roster is not a field here**, though §6.5 calls it part of the plan.
    An Assignment changes state when crew respond, so holding Assignment objects
    would make every acceptance a plan edit and bounce the mission back to DRAFT
    — contradicting §8.3 and making crewing impossible. The roster is derived
    from ``tenant.assignments``; ``approve`` records the approved members on the
    event. §6.5's guarantee is unaffected, because it is actually delivered by
    three other mechanisms: a requirements or window change releases everyone
    (§6.7), acceptance is final (§8.2), and the approval gate re-validates
    (§6.8).
    """

    requirements: tuple[Requirement, ...]
    window: TimeWindow
    site: Location
    version: int = 1

    def requirement(self, requirement_id: RequirementId) -> Requirement:
        for requirement in self.requirements:
            if requirement.id == requirement_id:
                return requirement
        raise NotFound("requirement", requirement_id)

    def matching_signature(self) -> tuple[Any, ...]:
        """The part of the plan the matcher actually consults (§6.7).

        Two plans with the same signature produce the same slots and the same
        eligible sets, so somebody who accepted one is equally valid on the
        other. Everything outside it — requirement labels, the site, and every
        field on the Mission wrapper — is presentation.

        **This is what decides whether an edit releases the crew**, and it is
        *derived* rather than a hand-kept list of "material fields" that a new
        field could be forgotten from. A field added to Requirement is
        cosmetic by default and only becomes material by being named here,
        which is the safe direction to fail: the cost of a wrong guess is
        re-offering a slot, not crewing a mission against a stale spec.

        Sorted by requirement id, so reordering the tuple is not a change.
        """
        return (
            self.window,
            tuple(
                (r.id, r.count, r.mandatory, r.skills)
                for r in sorted(self.requirements, key=lambda r: r.id)
            ),
        )

    def slots(self) -> Iterator[SlotKey]:
        for requirement in self.requirements:
            yield from requirement.slots()

    def mandatory_slots(self) -> frozenset[SlotKey]:
        return frozenset(
            slot for r in self.requirements if r.mandatory for slot in r.slots()
        )

    def slot_count(self) -> int:
        return sum(r.count for r in self.requirements)


@dataclass(frozen=True)
class Mission:
    """Plan fields live on ``plan``; everything else on this wrapper is metadata,
    editable in any non-terminal state without disturbing an approval (§6.1.1)."""

    id: MissionId
    title: str
    plan: MissionPlan
    created_by: MemberId
    state: MissionState = MissionState.DRAFT

    # metadata — editing these does not revert the mission
    description: str = ""
    tags: tuple[str, ...] = ()
    notes: str = ""
    reference_code: str | None = None

    # lifecycle bookkeeping
    submitted_by: MemberId | None = None
    submitted_at: datetime | None = None
    pending_expires_at: datetime | None = None
    approved_by: MemberId | None = None
    approved_at: datetime | None = None
    approved_plan_version: int | None = None
    closed_reason: str | None = None

    version: int = 1

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_MISSION_STATES


METADATA_FIELDS = frozenset({"title", "description", "tags", "notes", "reference_code"})
"""The right-hand column of §6.1.1's table. Anything in MissionPlan is the left."""


# ------------------------------------------------------------------- assignments


@dataclass(frozen=True)
class Assignment:
    """The join between crew and mission, and the only record of a commitment.

    There is no separate "active missions" structure — that would be a
    denormalisation of ``tenant.assignments`` that could drift (§4.3).
    """

    id: AssignmentId
    mission_id: MissionId
    member_id: MemberId
    requirement_id: RequirementId
    slot_index: int
    state: AssignmentState
    offered_at: datetime
    offer_expires_at: datetime
    responded_at: datetime | None = None
    release_reason: str | None = None
    version: int = 1

    @property
    def slot(self) -> SlotKey:
        return (self.requirement_id, self.slot_index)


# ------------------------------------------------------------- audit & messaging


@dataclass(frozen=True)
class MissionEvent:
    """The audit truth; aggregate state is a projection of the log (§6.3).

    Answers "who approved this, when, against which version of the plan, and had
    anyone objected first". For an approval workflow that is not a nice-to-have,
    it is the reason the workflow exists — and it is why no state exists merely
    to record that something happened.
    """

    at: datetime
    actor_id: MemberId
    actor_role: Role
    event: str
    subject_kind: str  # "mission" | "assignment"
    subject_id: str
    from_state: str | None
    to_state: str
    reason: str | None = None
    plan_version: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Notification:
    """Recorded rather than sent. Keeps effects testable and the core I/O-free."""

    at: datetime
    to: MemberId
    subject: str
    body: str


# -------------------------------------------------------------------- the caller


@dataclass(frozen=True)
class Actor:
    """Who is making this call. Not a fourth role — the role is a field on it.

    Separate from Member because it comes from the auth token rather than a
    request argument, it is immutable for the request, and it carries no domain
    data: passing a Member into authorisation invites policies to branch on
    profile fields (§3.2).
    """

    tenant_id: TenantId
    member_id: MemberId
    role: Role


@dataclass(frozen=True)
class TenantSettings:
    lookback: timedelta = timedelta(days=90)
    """Fairness window for the load term (§7.8)."""

    offer_ttl: timedelta = timedelta(hours=72)
    """How long a crew member has to answer an offer (§8.2)."""

    pending_ttl: timedelta = timedelta(days=7)
    """How long a fully-crewed mission may wait on a Director while holding real
    people (§6.7)."""

    alternates_depth: int = 3
    """How many next-best candidates to report per slot (§7.9)."""

    near_miss_depth: int = 10
    """How many near misses to report per requirement (§7.9).

    A display cap, not a matching parameter. "Excluded by exactly one filter"
    describes most of the roster once a tenant has thousands of crew, and the
    report is for a human — so each requirement keeps its most actionable few.
    """


# ------------------------------------------------------------------ availability


class SnapshotLike(Protocol):
    """The read surface availability needs. A Protocol so this module stays
    import-free; ``store.TenantSnapshot`` satisfies it structurally."""

    members: Mapping[MemberId, Member]
    crew: Mapping[MemberId, CrewProfile]
    missions: Mapping[MissionId, Mission]

    def assignments_for(self, member_id: MemberId) -> Sequence[Assignment]: ...


def blocking_windows(
    profile: CrewProfile,
    snap: SnapshotLike,
    *,
    ignoring: Iterable[AssignmentId] = (),
) -> Iterator[TimeWindow]:
    """Every window in which this crew member is unavailable.

    Two sources (§4.3)::

        unavailable = committed_mission_windows  u  declared_unavailability

    "Committed" is the **pair** of states, not MissionState alone: an ACCEPTED
    assignment on any non-terminal mission blocks. Reading it as "ACTIVE
    missions only" is a double-booking bug — two APPROVED missions with
    overlapping windows would each see the member as free.

    ``ignoring`` excludes an assignment from its own conflict check, which is
    what ``accept`` needs (§8.2).
    """
    skip = set(ignoring)

    for block in profile.unavailability:
        yield block.window

    for assignment in snap.assignments_for(profile.member_id):
        if assignment.id in skip:
            continue
        if assignment.state is not AssignmentState.ACCEPTED:
            continue  # OFFERED takes no hold; DECLINED/RELEASED never did
        mission = snap.missions.get(assignment.mission_id)
        if mission is None or mission.is_terminal:
            continue  # history, not a commitment
        yield mission.plan.window


def is_available(
    profile: CrewProfile,
    window: TimeWindow,
    snap: SnapshotLike,
    *,
    ignoring: Iterable[AssignmentId] = (),
) -> bool:
    """Availability is a boolean, not a fraction (§4.3).

    A slot is one person for the whole window, so accepting someone who covers
    85% of it leaves the slot nominally filled and actually empty for three
    days, with nowhere to record who covers the gap. The genuine need — someone
    for the first fortnight, someone else for the last week — is two
    requirements with their own windows.
    """
    return not any(
        blocked.overlaps(window)
        for blocked in blocking_windows(profile, snap, ignoring=ignoring)
    )


def conflicting_commitments(
    profile: CrewProfile,
    window: TimeWindow,
    snap: SnapshotLike,
    *,
    ignoring: Iterable[AssignmentId] = (),
) -> list[Assignment]:
    """The ACCEPTED assignments that overlap ``window``, for error messages.

    "You have already accepted Kepler (2026-03-01..2026-03-20)" is actionable in
    a way that "unavailable" is not.
    """
    skip = set(ignoring)
    out = []
    for assignment in snap.assignments_for(profile.member_id):
        if assignment.id in skip or assignment.state is not AssignmentState.ACCEPTED:
            continue
        mission = snap.missions.get(assignment.mission_id)
        if mission is None or mission.is_terminal:
            continue
        if mission.plan.window.overlaps(window):
            out.append(assignment)
    return out


# ------------------------------------------------------------------- eligibility

FILTER_ACTIVE_MEMBER = "active_member"
FILTER_SKILLS = "skills"
FILTER_AVAILABILITY = "availability"

FILTER_DECLINED = "declined"
"""Already said no to *this* mission.

Mission-scoped, so it is applied by the matcher during candidate generation
rather than by ``eligibility_failures`` — which is also used to re-validate an
existing roster (§6.8), where a member who declined one slot and later accepted
another must not be marked stale.

§7.3 does not list this filter. It is a gap rather than a decision: without it a
decline is information the engine discards, so re-running the matcher proposes
the same person for the same slot and the loop never terminates. A Lead can
still offer to someone who declined — they may have spoken to them — but the
engine should not *suggest* it.
"""

FILTERS = (FILTER_ACTIVE_MEMBER, FILTER_SKILLS, FILTER_AVAILABILITY, FILTER_DECLINED)


def eligibility_failures(
    requirement: Requirement,
    member: Member,
    profile: CrewProfile,
    window: TimeWindow,
    snap: SnapshotLike,
    *,
    ignoring: Iterable[AssignmentId] = (),
    available: bool | None = None,
) -> tuple[str, ...]:
    """Which of §7.3's filters reject this ``(crew, requirement)`` pair.

    Empty tuple means eligible. Lives here rather than in ``matching`` because
    it is the *definition* of eligible, and two callers need exactly the same
    definition: candidate generation (§7.3) and the approval gate that
    re-validates an existing roster (§6.8). One copy, so they cannot diverge.

    Returns **all** failures rather than short-circuiting: a near-miss is
    "excluded by exactly one filter" (§7.9), and you cannot count to one if you
    stopped at the first.

    ``available`` lets the caller pass a precomputed answer, since availability
    is mission-level and worth computing once per member rather than once per
    ``(member, requirement)`` pair (§7.3).
    """
    failed: list[str] = []

    if member.status is not MemberStatus.ACTIVE or member.role is not Role.CREW:
        failed.append(FILTER_ACTIVE_MEMBER)

    for wanted in requirement.skills:
        held = profile.level(wanted.skill_id)
        if held is None or held < wanted.min_level:
            failed.append(FILTER_SKILLS)
            break

    if available is None:
        available = is_available(profile, window, snap, ignoring=ignoring)
    if not available:
        failed.append(FILTER_AVAILABILITY)

    return tuple(failed)


def skill_shortfall(
    requirement: Requirement, profile: CrewProfile
) -> list[tuple[SkillId, Proficiency, Proficiency | None]]:
    """``(skill, required, held)`` for each unmet skill — for explanations."""
    return [
        (wanted.skill_id, wanted.min_level, profile.level(wanted.skill_id))
        for wanted in requirement.skills
        if (profile.level(wanted.skill_id) or 0) < wanted.min_level
    ]
