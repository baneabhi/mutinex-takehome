"""The two state machines: mission (authorisation) and assignment (crew response).

Transitions are **data**, not control flow (§6.2). One structure drives
``available_events()``, the UI's buttons, the API's responses and the exhaustive
test sweep — derived from one table, they cannot disagree.

The machines couple in one direction only. Mission events drive assignment
events; **crew responses never move the mission** (§8.3). When the last crew
member accepts, the mission becomes *eligible* for submission and a Lead submits
it. So "awaiting responses" and "fully crewed" are not mission states but
``crewing_status``, a projection.

This module talks to storage through the ``Store`` Protocol rather than
importing it, so ``lifecycle`` depends on ``domain`` and ``authz`` only (§9).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Protocol

from .authz import (
    MissionOwnershipPolicy,
    OwnAssignmentPolicy,
    OwnershipPolicy,
    Permission,
    Policy,
    SeparationOfDuties,
    enforce,
    evaluate,
    has,
    require,
)
from .domain import (
    LIVE_ASSIGNMENT_STATES,
    Actor,
    Assignment,
    AssignmentState,
    CrewingStatus,
    GuardFailed,
    IllegalTransition,
    MemberId,
    Mission,
    MissionEvent,
    MissionId,
    MissionPlan,
    MissionState,
    SlotKey,
    SnapshotLike,
    TenantSettings,
    conflicting_commitments,
    eligibility_failures,
    is_available,
    skill_shortfall,
)

P = Permission
MS = MissionState
AS = AssignmentState


# ----------------------------------------------------------------- the seam


class Store(Protocol):
    """The slice of storage guards and effects need.

    A Protocol rather than an import: it keeps the dependency arrow pointing at
    ``domain`` and lets tests drive the machines with a dict-backed fake.
    """

    settings: TenantSettings

    def snapshot(self) -> SnapshotLike: ...
    def assignments_for_mission(self, mission_id: MissionId) -> list[Assignment]: ...
    def save_mission(self, mission: Mission) -> None: ...
    def save_assignment(self, assignment: Assignment) -> None: ...
    def record_event(self, event: MissionEvent) -> None: ...
    def notify(
        self, member_id: MemberId, subject: str, body: str, *, at: datetime
    ) -> None: ...


@dataclass(frozen=True)
class Context:
    """Everything a transition is allowed to see.

    ``now`` is passed in rather than read from the clock so a run is
    reproducible and so time-dependent guards are testable without sleeping.
    """

    store: Store
    actor: Actor
    now: datetime
    reason: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    mission: Mission | None = None
    """Set for assignment transitions, which need the mission's window and
    owner. Assignments are never authorised in isolation."""

    @property
    def settings(self) -> TenantSettings:
        return self.store.settings

    def assignments_for_mission(self, mission_id: MissionId) -> Sequence[Assignment]:
        return self.store.assignments_for_mission(mission_id)

    def notify(self, member_id: MemberId, subject: str, body: str) -> None:
        """Stamps every notification with the transition's ``now``, so one
        operation's messages all share a timestamp."""
        self.store.notify(member_id, subject, body, at=self.now)


Guard = Callable[[Any, Context], "str | None"]
"""Returns a failure reason, or None to pass. A string rather than a bool so the
error can say *why*."""

Effect = Callable[[Any, Context], Any]
"""Returns the updated aggregate. May also write assignments and notifications
through ``ctx.store``."""


@dataclass(frozen=True)
class Transition:
    source: Any  # MissionState | AssignmentState | None (None = creation)
    event: str
    target: Any
    permission: Permission | None
    """None means system-triggered. Those are excluded from
    ``available_events`` — they are not buttons."""
    policies: tuple[Policy, ...] = ()
    guards: tuple[Guard, ...] = ()
    effects: tuple[Effect, ...] = ()


def _index(transitions: Sequence[Transition]) -> dict[tuple[Any, str], Transition]:
    table: dict[tuple[Any, str], Transition] = {}
    for transition in transitions:
        key = (transition.source, transition.event)
        if key in table:
            raise ValueError(f"duplicate transition {key}")
        table[key] = transition
    return table


# ---------------------------------------------------------- derived projections


def crewing_status(mission: Mission, assignments: Sequence[Assignment]) -> CrewingStatus:
    """Derived on read, never stored (§8.3).

    There is no cached projection to invalidate, which removes the largest class
    of concurrency bug by not having the state that causes it (§3.3).
    """
    accepted = {a.slot for a in assignments if a.state is AS.ACCEPTED}
    offered = {a.slot for a in assignments if a.state is AS.OFFERED}

    outstanding = mission.plan.mandatory_slots() - accepted
    if not outstanding:
        return CrewingStatus.FULLY_CREWED
    if outstanding <= offered:
        return CrewingStatus.AWAITING_RESPONSES
    return CrewingStatus.UNDER_CREWED


def filled_slots(assignments: Sequence[Assignment]) -> set[SlotKey]:
    """Slots with a live assignment. OFFERED counts as filled for the purpose of
    not double-offering the same slot, even though it takes no availability hold
    (§6.7)."""
    return {a.slot for a in assignments if a.state in LIVE_ASSIGNMENT_STATES}


def pinned_slots(assignments: Sequence[Assignment]) -> dict[SlotKey, MemberId]:
    """Accepted crew, by slot. A re-match pins these rather than re-asking
    (§6.7)."""
    return {a.slot: a.member_id for a in assignments if a.state is AS.ACCEPTED}


# --------------------------------------------------------- the §6.8 approval gate


@dataclass(frozen=True)
class StaleSlot:
    slot: SlotKey
    member_id: MemberId
    mandatory: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AllocationReport:
    """Computed on read, never stored, so the report cannot drift from the
    assignments it describes — and it is less code than a flag plus every write
    path that must update it (§6.8)."""

    valid: tuple[SlotKey, ...]
    stale: tuple[StaleSlot, ...]

    @property
    def blocking(self) -> tuple[StaleSlot, ...]:
        """Only unfillable **mandatory** slots bounce a mission. A stale optional
        slot is dropped and noted, because blocking a Director's judgement about
        whether a mission should happen over an optional slot enforces crewing
        completeness in the wrong place (§6.8)."""
        return tuple(s for s in self.stale if s.mandatory)

    def summary(self) -> str:
        return "; ".join(
            f"{s.member_id} no longer eligible for {s.slot[0]}#{s.slot[1]} ({', '.join(s.reasons)})"
            for s in self.stale
        )


def validate_allocation(
    mission: Mission, snap: SnapshotLike, assignments: Sequence[Assignment]
) -> AllocationReport:
    """Re-run §7.3's filters over the accepted roster.

    Acceptance-before-approval shrinks what can go wrong to two triggers (§6.8):
    between ``submit`` and ``approve`` an accepted crew member can be
    **deactivated**, or have their **skill levels revised down** below a
    requirement. Crew cannot withdraw an acceptance, and an availability edit
    that conflicts with an accepted assignment is rejected at the write.

    Filters evaluate against the **mission's** window, not ``now``: a mission
    approved today for a window six months out must be checked for conflicts
    *in that window*.
    """
    valid: list[SlotKey] = []
    stale: list[StaleSlot] = []

    for assignment in assignments:
        if assignment.state is not AS.ACCEPTED:
            continue
        try:
            requirement = mission.plan.requirement(assignment.requirement_id)
        except Exception:
            # The requirement was removed from the plan; the slot no longer
            # exists, so there is nothing to validate. plan_edited already
            # released this assignment.
            continue

        member = snap.members.get(assignment.member_id)
        profile = snap.crew.get(assignment.member_id)
        if member is None or profile is None:
            stale.append(
                StaleSlot(assignment.slot, assignment.member_id, requirement.mandatory,
                          ("member no longer exists",))
            )
            continue

        failures = eligibility_failures(
            requirement,
            member,
            profile,
            mission.plan.window,
            snap,
            ignoring=(assignment.id,),  # their own acceptance is not a conflict
        )
        if not failures:
            valid.append(assignment.slot)
            continue

        reasons = []
        for name in failures:
            if name == "skills":
                short = skill_shortfall(requirement, profile)
                reasons.append(
                    "skills below requirement: "
                    + ", ".join(f"{sid} needs {req.name}, holds {held.name if held else 'none'}"
                                for sid, req, held in short)
                )
            elif name == "active_member":
                reasons.append("member deactivated or no longer crew")
            else:
                clashes = conflicting_commitments(
                    profile, mission.plan.window, snap, ignoring=(assignment.id,)
                )
                if clashes:
                    reasons.append(
                        "conflicts with "
                        + ", ".join(snap.missions[c.mission_id].title for c in clashes)
                    )
                else:
                    reasons.append("declared unavailable for this window")
        stale.append(
            StaleSlot(assignment.slot, assignment.member_id, requirement.mandatory, tuple(reasons))
        )

    return AllocationReport(tuple(valid), tuple(stale))


# ---------------------------------------------------------------- mission guards


def has_requirements(mission: Mission, ctx: Context) -> str | None:
    if not mission.plan.requirements:
        return "the plan has no requirements"
    return None


def window_is_future(mission: Mission, ctx: Context) -> str | None:
    if mission.plan.window.start <= ctx.now:
        return f"the mission window ({mission.plan.window}) does not start in the future"
    return None


def fully_crewed(mission: Mission, ctx: Context) -> str | None:
    """``submit`` asks "is this staffed?", not "does this look staffable?" (§6.6).

    Also guards ``activate``, where **nobody can waive a missing mandatory
    slot** — activating short of mandatory crew runs something different from
    what was reviewed, and since Directors hold no authoring permissions there
    is not even a role that could own such an override (§6.8).
    """
    status = crewing_status(mission, ctx.assignments_for_mission(mission.id))
    if status is CrewingStatus.FULLY_CREWED:
        return None
    return (
        f"mission is {status.value}; every mandatory slot must be ACCEPTED first"
    )


def roster_still_valid(mission: Mission, ctx: Context) -> str | None:
    """Defensive backstop on ``approve``, and the real check on ``activate``.

    ``MissionService.approve`` evaluates the gate itself and fires
    ``invalidated`` instead of ``approve`` when it fails, because §6.8 wants a
    broken roster to *return the mission to DRAFT* rather than raise. This guard
    exists so a caller reaching ``apply_mission_event`` directly cannot skip it.
    """
    report = validate_allocation(
        mission, ctx.store.snapshot(), ctx.assignments_for_mission(mission.id)
    )
    if report.blocking:
        return f"roster is no longer valid — {report.summary()}"
    return None


def reason_required(mission: Mission, ctx: Context) -> str | None:
    if not (ctx.reason or "").strip():
        return "a reason is required"
    return None


def approval_window_expired(mission: Mission, ctx: Context) -> str | None:
    if mission.pending_expires_at is None:
        return "no approval deadline is set"
    if ctx.now <= mission.pending_expires_at:
        return f"the approval window runs until {mission.pending_expires_at}"
    return None


def window_ended_or_reason_given(mission: Mission, ctx: Context) -> str | None:
    if ctx.now < mission.plan.window.end and not (ctx.reason or "").strip():
        return (
            f"the mission window runs until {mission.plan.window.end}; "
            "completing early needs a reason"
        )
    return None


# --------------------------------------------------------------- mission effects

PLAN_FIELDS = ("requirements", "window", "site")

def plan_changes(old: MissionPlan, new: MissionPlan) -> frozenset[str]:
    return frozenset(f for f in PLAN_FIELDS if getattr(old, f) != getattr(new, f))


def install_new_plan(mission: Mission, ctx: Context) -> Mission:
    """The plan is replaced *by* the event, which is what makes §6.1.1
    structural: there is no other route to a plan field."""
    new_plan: MissionPlan = ctx.params["new_plan"]
    return replace(mission, plan=replace(new_plan, version=mission.plan.version + 1))


def release_for_plan_change(mission: Mission, ctx: Context) -> Mission:
    """Release the crew when, and only when, the **matching signature** changed.

    An edit that alters what the matcher consults — requirement skills, counts,
    the mandatory flag, or the mission window — invalidates the roster: those
    people agreed to a different specification, and the matcher must run again.
    An edit that only alters presentation does not. Fixing "Flight Sugeon" to
    "Flight Surgeon", or correcting the spelling of the site, should not cost a
    Lead their crew and make five people answer the same question twice.

    Offers and acceptances are treated alike here. An open offer is a pending
    question; if the specification behind it is unchanged, the question is still
    the right one.

    The rule needs no maintained list of "material fields" — it is a comparison
    of ``MissionPlan.matching_signature()``, which is derived. And because the
    release happens at the instant of the change, the invariant *the live roster
    was built against the current signature* holds continuously, so nothing has
    to remember which version the crew said yes to.
    """
    if "previous_plan" not in ctx.params:
        # Belongs on the plan_edited row and nowhere else. Attached to a row
        # that does not supply the old plan — run_matcher is the tempting one —
        # it would have nothing to compare against and silently release the
        # whole roster every time. Loud beats subtle.
        raise ValueError(
            "release_for_plan_change needs params['previous_plan']; it is only "
            "valid on a transition that replaces the plan"
        )
    previous: MissionPlan = ctx.params["previous_plan"]
    if previous.matching_signature() == mission.plan.matching_signature():
        return mission  # presentation only; the roster still stands

    for assignment in ctx.assignments_for_mission(mission.id):
        if assignment.state in LIVE_ASSIGNMENT_STATES:
            _fire_assignment(
                assignment, "release", ctx, mission,
                reason="the mission requirements or window changed; re-crewing",
            )
    return mission


def record_submission(mission: Mission, ctx: Context) -> Mission:
    return replace(
        mission,
        submitted_by=ctx.actor.member_id,
        submitted_at=ctx.now,
        pending_expires_at=ctx.now + ctx.settings.pending_ttl,
    )


def clear_submission(mission: Mission, ctx: Context) -> Mission:
    return replace(mission, submitted_by=None, submitted_at=None, pending_expires_at=None)


def record_approval(mission: Mission, ctx: Context) -> Mission:
    """Binds the approval to a specific plan version. Any later edit reverts the
    mission to DRAFT, so a recorded version can never describe a plan the
    Director did not see (§6.1.1)."""
    return replace(
        mission,
        approved_by=ctx.actor.member_id,
        approved_at=ctx.now,
        approved_plan_version=mission.plan.version,
    )


def clear_approval(mission: Mission, ctx: Context) -> Mission:
    return replace(mission, approved_by=None, approved_at=None, approved_plan_version=None)


def record_closed_reason(mission: Mission, ctx: Context) -> Mission:
    return replace(mission, closed_reason=ctx.reason)


def notify_crew(subject: str) -> Effect:
    def effect(mission: Mission, ctx: Context) -> Mission:
        for assignment in ctx.assignments_for_mission(mission.id):
            if assignment.state in LIVE_ASSIGNMENT_STATES:
                ctx.notify(
                    assignment.member_id,
                    subject,
                    f"Mission {mission.title!r} ({mission.plan.window}): {subject}."
                    + (f" Reason: {ctx.reason}" if ctx.reason else ""),
                )
        return mission

    effect.__name__ = f"notify_crew[{subject}]"
    return effect


def notify_lead(subject: str) -> Effect:
    def effect(mission: Mission, ctx: Context) -> Mission:
        ctx.notify(
            mission.created_by,
            subject,
            f"Mission {mission.title!r}: {subject}."
            + (f" Reason: {ctx.reason}" if ctx.reason else ""),
        )
        return mission

    effect.__name__ = f"notify_lead[{subject}]"
    return effect


def release_all_assignments(mission: Mission, ctx: Context) -> Mission:
    """Without this a cancelled mission would hold its crew's dates forever
    (§8.1)."""
    for assignment in ctx.assignments_for_mission(mission.id):
        if assignment.state in LIVE_ASSIGNMENT_STATES:
            _fire_assignment(
                assignment, "release", ctx, mission,
                reason=ctx.reason or f"mission {mission.state.value}",
            )
    return mission


def release_stale_assignments(mission: Mission, ctx: Context) -> Mission:
    """Used by ``invalidated``: release only the members who failed the gate,
    leaving the rest of the roster intact so re-crewing is a gap-fill rather
    than a restart."""
    stale = {s.slot for s in ctx.params.get("report", AllocationReport((), ())).stale}
    for assignment in ctx.assignments_for_mission(mission.id):
        if assignment.slot in stale and assignment.state in LIVE_ASSIGNMENT_STATES:
            _fire_assignment(
                assignment, "release", ctx, mission,
                reason="no longer eligible at approval time",
            )
    return mission


def complete_assignments(mission: Mission, ctx: Context) -> Mission:
    for assignment in ctx.assignments_for_mission(mission.id):
        if assignment.state is AS.ACCEPTED:
            _fire_assignment(assignment, "complete", ctx, mission)
        elif assignment.state is AS.OFFERED:
            _fire_assignment(
                assignment, "release", ctx, mission, reason="mission completed unanswered"
            )
    return mission


def partial_assignments(mission: Mission, ctx: Context) -> Mission:
    for assignment in ctx.assignments_for_mission(mission.id):
        if assignment.state is AS.ACCEPTED:
            _fire_assignment(assignment, "abort", ctx, mission, reason=ctx.reason)
        elif assignment.state is AS.OFFERED:
            _fire_assignment(
                assignment, "release", ctx, mission, reason="mission aborted"
            )
    return mission


# ---------------------------------------------------------- the mission table
#
# §6.2. Every row is somebody's decision except the two marked system-triggered.

MISSION_TRANSITIONS = _index(
    [
        # --- DRAFT: the one editable state, and the only one the matcher runs in
        Transition(
            MS.DRAFT, "run_matcher", MS.DRAFT,
            P.MATCHER_RUN,
            policies=(OwnershipPolicy(),),
            guards=(has_requirements,),
            # No effects: the matcher never creates assignments. It returns a
            # MatchProposal and a Lead accepts it slot by slot (§7.9). The row
            # exists so available_events() reports it and so the run is audited.
        ),
        Transition(
            # The **only** row that can change a plan. DRAFT is the one editable
            # state; every other state reaches it through `reopen`, which is a
            # decision somebody makes rather than a side effect of typing
            # (§6.1.1). That is also why install_new_plan and
            # release_for_plan_change appear exactly once in this table.
            MS.DRAFT, "plan_edited", MS.DRAFT,
            P.MISSION_UPDATE,
            policies=(OwnershipPolicy(),),
            effects=(install_new_plan, release_for_plan_change),
        ),
        Transition(
            MS.DRAFT, "submit", MS.PENDING_APPROVAL,
            P.MISSION_SUBMIT,
            policies=(OwnershipPolicy(),),
            guards=(has_requirements, window_is_future, fully_crewed),
            effects=(record_submission, notify_crew("submitted for approval")),
        ),
        Transition(
            MS.DRAFT, "cancel", MS.CANCELLED,
            P.MISSION_CANCEL,
            policies=(OwnershipPolicy(),),
            guards=(reason_required,),
            effects=(release_all_assignments, record_closed_reason, notify_crew("cancelled")),
        ),
        # --- PENDING_APPROVAL: authorisation, and four ways back to DRAFT
        Transition(
            MS.PENDING_APPROVAL, "approve", MS.APPROVED,
            P.MISSION_APPROVE,
            policies=(SeparationOfDuties(),),
            # fully_crewed as well as roster validity: a member deactivated
            # between submit and approve has their assignment released, which
            # leaves a hole rather than a stale slot. Checking only the roster
            # would let an under-crewed mission be approved (§6.8).
            guards=(fully_crewed, roster_still_valid),
            effects=(record_approval, notify_crew("approved"), notify_lead("approved")),
        ),
        Transition(
            MS.PENDING_APPROVAL, "request_changes", MS.DRAFT,
            P.MISSION_REJECT,
            policies=(SeparationOfDuties(),),
            guards=(reason_required,),
            effects=(clear_submission, notify_lead("changes requested"),
                     notify_crew("returned to draft")),
        ),
        Transition(
            # The plan is frozen while a Director is reading it. To change
            # anything the Lead reopens first, which is an explicit act rather
            # than a side effect of editing (§6.1.1).
            MS.PENDING_APPROVAL, "reopen", MS.DRAFT,
            P.MISSION_UPDATE,
            policies=(OwnershipPolicy(),),
            effects=(clear_submission, notify_crew("returned to draft by the lead")),
        ),
        Transition(
            # System-triggered. Bounds how long a fully-crewed mission can wait
            # on a Director while holding real people (§6.7).
            MS.PENDING_APPROVAL, "expire", MS.DRAFT,
            None,
            guards=(approval_window_expired,),
            effects=(clear_submission, notify_lead("approval window expired"),
                     notify_crew("still awaiting approval")),
        ),
        Transition(
            # System-triggered, fired by MissionService.approve when the §6.8
            # gate fails: the mission returns to DRAFT rather than the Director
            # getting an error, because there is exactly one editable state and
            # this is the same destination as every other backwards transition.
            MS.PENDING_APPROVAL, "invalidated", MS.DRAFT,
            None,
            effects=(clear_submission, release_stale_assignments,
                     notify_lead("roster invalid at approval — returned to draft")),
        ),
        Transition(
            MS.PENDING_APPROVAL, "cancel", MS.CANCELLED,
            P.MISSION_CANCEL,
            policies=(OwnershipPolicy(),),
            guards=(reason_required,),
            effects=(release_all_assignments, record_closed_reason, notify_crew("cancelled")),
        ),
        # --- APPROVED: frozen too. Reopening costs the approval.
        Transition(
            MS.APPROVED, "activate", MS.ACTIVE,
            P.MISSION_ACTIVATE,
            policies=(OwnershipPolicy(),),
            guards=(fully_crewed, roster_still_valid),
            # No effects: the crew are already ACCEPTED (§6.6), which is the
            # whole point of crewing before approval.
        ),
        Transition(
            MS.APPROVED, "reopen", MS.DRAFT,
            P.MISSION_UPDATE,
            policies=(OwnershipPolicy(),),
            effects=(clear_submission, clear_approval,
                     notify_crew("reopened for changes; re-approval needed"),
                     notify_lead("reopened; the approval has been discarded")),
        ),
        Transition(
            MS.APPROVED, "cancel", MS.CANCELLED,
            P.MISSION_CANCEL,
            policies=(OwnershipPolicy(),),
            guards=(reason_required,),
            effects=(release_all_assignments, record_closed_reason, notify_crew("cancelled")),
        ),
        # --- ACTIVE: no plan edits, two ways out
        Transition(
            MS.ACTIVE, "complete", MS.COMPLETED,
            P.MISSION_COMPLETE,
            policies=(OwnershipPolicy(),),
            guards=(window_ended_or_reason_given,),
            effects=(complete_assignments, record_closed_reason, notify_crew("completed")),
        ),
        Transition(
            # Distinct from cancel: stopping a mission people are currently
            # flying has different consequences, and needs assignment history to
            # record partial participation (§6.1).
            #
            # §6.2's table leaves the policy column blank here. Read literally
            # that lets any Lead abort any other Lead's active mission, which
            # cancel — the same act before launch — does not allow. Treating it
            # as a slip and scoping it like cancel.
            MS.ACTIVE, "abort", MS.ABORTED,
            P.MISSION_CANCEL,
            policies=(OwnershipPolicy(),),
            guards=(reason_required,),
            effects=(partial_assignments, record_closed_reason, notify_crew("aborted")),
        ),
    ]
)


# ------------------------------------------------------------- assignment guards


def mission_is_draft(assignment: Assignment, ctx: Context) -> str | None:
    mission = _require_mission(ctx)
    if mission.state is not MS.DRAFT:
        return f"mission is {mission.state.value}; offers are only made in draft (§6.7)"
    return None


def slot_is_unfilled(assignment: Assignment, ctx: Context) -> str | None:
    mission = _require_mission(ctx)
    existing = ctx.assignments_for_mission(mission.id)
    if assignment.slot in filled_slots([a for a in existing if a.id != assignment.id]):
        return f"slot {assignment.requirement_id}#{assignment.slot_index} already has a live assignment"
    try:
        requirement = mission.plan.requirement(assignment.requirement_id)
    except Exception:
        return f"requirement {assignment.requirement_id} is not in this plan"
    if not 0 <= assignment.slot_index < requirement.count:
        return f"slot index {assignment.slot_index} is out of range for count {requirement.count}"
    return None


def candidate_is_eligible(assignment: Assignment, ctx: Context) -> str | None:
    """The matcher's proposal may be slightly stale — it ran lock-free on a
    snapshot (§3.3). This is where that is caught: the worst case of a stale
    proposal is a row that turns out unofferable.
    """
    mission = _require_mission(ctx)
    snap = ctx.store.snapshot()
    member = snap.members.get(assignment.member_id)
    profile = snap.crew.get(assignment.member_id)
    if member is None or profile is None:
        return f"{assignment.member_id} has no crew profile"
    requirement = mission.plan.requirement(assignment.requirement_id)
    failures = eligibility_failures(requirement, member, profile, mission.plan.window, snap)
    if failures:
        return f"{member.name} is no longer eligible ({', '.join(failures)})"
    return None


def offer_not_expired(assignment: Assignment, ctx: Context) -> str | None:
    if ctx.now > assignment.offer_expires_at:
        return f"this offer expired at {assignment.offer_expires_at}"
    return None


def offer_has_expired(assignment: Assignment, ctx: Context) -> str | None:
    if ctx.now <= assignment.offer_expires_at:
        return f"this offer runs until {assignment.offer_expires_at}"
    return None


def no_conflicting_acceptance(assignment: Assignment, ctx: Context) -> str | None:
    """**The one race that needs a lock** (§3.3).

    OFFERED applies no hold, so two Leads may court the same person for
    overlapping windows — by design, and the first acceptance wins. That makes
    conflict-check-and-write the one operation that must be atomic per crew
    member. Resolved at commitment rather than prevented by locking the
    candidate set, because no lock can span the human gap between running the
    matcher and someone answering an offer.
    """
    mission = _require_mission(ctx)
    snap = ctx.store.snapshot()
    profile = snap.crew.get(assignment.member_id)
    if profile is None:
        return f"{assignment.member_id} has no crew profile"

    clashes = conflicting_commitments(
        profile, mission.plan.window, snap, ignoring=(assignment.id,)
    )
    if clashes:
        names = ", ".join(
            f"{snap.missions[c.mission_id].title!r} ({snap.missions[c.mission_id].plan.window})"
            for c in clashes
        )
        return f"you have already accepted {names}, which overlaps this window"

    if not is_available(profile, mission.plan.window, snap, ignoring=(assignment.id,)):
        return "you have declared yourself unavailable for this window"
    return None


# ------------------------------------------------------------ assignment effects


def stamp_response(assignment: Assignment, ctx: Context) -> Assignment:
    return replace(assignment, responded_at=ctx.now)


def stamp_release(assignment: Assignment, ctx: Context) -> Assignment:
    return replace(assignment, responded_at=assignment.responded_at or ctx.now,
                   release_reason=ctx.reason)


def notify_offer(assignment: Assignment, ctx: Context) -> Assignment:
    mission = _require_mission(ctx)
    ctx.notify(
        assignment.member_id,
        "You have a mission offer",
        # Honest about the mission being unapproved: crewing precedes approval,
        # so people can be asked about work a Director later refuses (§6.6).
        f"{mission.title!r} ({mission.plan.window}) at {mission.plan.site}, as "
        f"{mission.plan.requirement(assignment.requirement_id).label}. "
        f"This mission is not yet approved. Respond by {assignment.offer_expires_at}.",
    )
    return assignment


def notify_lead_of_response(verb: str) -> Effect:
    def effect(assignment: Assignment, ctx: Context) -> Assignment:
        mission = _require_mission(ctx)
        ctx.notify(
            mission.created_by,
            f"Offer {verb}",
            f"{assignment.member_id} {verb} {mission.title!r} "
            f"({assignment.requirement_id}#{assignment.slot_index}).",
        )
        return assignment

    effect.__name__ = f"notify_lead_of_response[{verb}]"
    return effect


def notify_crew_of_release(assignment: Assignment, ctx: Context) -> Assignment:
    ctx.notify(
        assignment.member_id,
        "Assignment released",
        f"Your assignment on {assignment.mission_id} has been released."
        + (f" Reason: {ctx.reason}" if ctx.reason else ""),
    )
    return assignment


# ------------------------------------------------------- the assignment table
#
# §8.2. Everything except offer/accept/decline is system-triggered — driven by a
# mission event, or by the clock.

ASSIGNMENT_TRANSITIONS = _index(
    [
        Transition(
            None, "offer", AS.OFFERED,
            P.ASSIGNMENT_OFFER,
            policies=(MissionOwnershipPolicy(),),
            guards=(mission_is_draft, slot_is_unfilled, candidate_is_eligible),
            effects=(notify_offer,),
        ),
        Transition(
            AS.OFFERED, "accept", AS.ACCEPTED,
            P.ASSIGNMENT_RESPOND_OWN,
            policies=(OwnAssignmentPolicy(),),
            guards=(offer_not_expired, no_conflicting_acceptance),
            effects=(stamp_response, notify_lead_of_response("accepted")),
        ),
        Transition(
            AS.OFFERED, "decline", AS.DECLINED,
            P.ASSIGNMENT_RESPOND_OWN,
            policies=(OwnAssignmentPolicy(),),
            effects=(stamp_response, notify_lead_of_response("declined")),
        ),
        Transition(
            # Expiry *is* a decline: it stops one unresponsive person holding a
            # mission indefinitely, and harmlessly, because it stalls a draft
            # rather than an approved mission (§6.7).
            AS.OFFERED, "offer_expired", AS.DECLINED,
            None,
            guards=(offer_has_expired,),
            effects=(stamp_response, notify_lead_of_response("let the offer lapse")),
        ),
        Transition(
            AS.OFFERED, "release", AS.RELEASED,
            None,
            effects=(stamp_release, notify_crew_of_release),
        ),
        Transition(
            AS.ACCEPTED, "release", AS.RELEASED,
            None,
            effects=(stamp_release, notify_crew_of_release),
        ),
        Transition(AS.ACCEPTED, "complete", AS.COMPLETED, None, effects=(stamp_release,)),
        Transition(AS.ACCEPTED, "abort", AS.PARTIAL, None, effects=(stamp_release,)),
        # There is deliberately no ACCEPTED -> DECLINED edge. Acceptance is
        # final (§8.2). In a real system this is insufficient — people get sick
        # — and the honest model needs a crew-initiated release with a re-crew
        # path (§11 trade-off 3).
    ]
)


# ------------------------------------------------------------------- the engine


def _require_mission(ctx: Context) -> Mission:
    if ctx.mission is None:
        raise ValueError("assignment transitions need ctx.mission")
    return ctx.mission


def _apply(
    table: Mapping[tuple[Any, str], Transition],
    kind: str,
    aggregate: Any,
    source: Any,
    event: str,
    ctx: Context,
    *,
    available: Callable[[], Sequence[str]],
) -> Any:
    transition = table.get((source, event))
    if transition is None:
        # Computed only on the failure path: the whole point of the message is
        # that it costs nothing when nothing went wrong (§6.4).
        raise IllegalTransition(
            kind, source.value if hasattr(source, "value") else str(source), event, available()
        )

    # Layer one, then layer two, then domain preconditions (§5.1). The order
    # matters: a caller who may not do this at all should not learn from the
    # error message whether the preconditions hold.
    if transition.permission is not None:
        require(ctx.actor, transition.permission)
    enforce(transition.policies, ctx.actor, aggregate, ctx)
    for guard in transition.guards:
        failure = guard(aggregate, ctx)
        if failure:
            raise GuardFailed(event, failure)

    updated = replace(aggregate, state=transition.target, version=aggregate.version + 1)
    for effect in transition.effects:
        updated = effect(updated, ctx)
    return updated


def apply_mission_event(mission: Mission, event: str, ctx: Context) -> Mission:
    """Fire ``event`` against ``mission``, persisting the result.

    Callers should hold the tenant lock. The critical section stays small
    because nothing expensive happens in here — the matcher runs outside it, on
    a snapshot (§3.3).
    """
    updated = _apply(
        MISSION_TRANSITIONS, "mission", mission, mission.state, event,
        replace(ctx, mission=mission),
        available=lambda: available_events(mission, ctx),
    )
    ctx.store.save_mission(updated)
    ctx.store.record_event(
        MissionEvent(
            at=ctx.now,
            actor_id=ctx.actor.member_id,
            actor_role=ctx.actor.role,
            event=event,
            subject_kind="mission",
            subject_id=mission.id,
            from_state=mission.state.value,
            to_state=updated.state.value,
            reason=ctx.reason,
            plan_version=updated.plan.version,
            metadata=dict(ctx.params.get("metadata", {})),
        )
    )
    return updated


def apply_assignment_event(assignment: Assignment, event: str, ctx: Context) -> Assignment:
    updated = _apply(
        ASSIGNMENT_TRANSITIONS, "assignment", assignment, assignment.state, event, ctx,
        available=lambda: available_assignment_events(assignment, ctx),
    )
    ctx.store.save_assignment(updated)
    ctx.store.record_event(
        MissionEvent(
            at=ctx.now,
            actor_id=ctx.actor.member_id,
            actor_role=ctx.actor.role,
            event=event,
            subject_kind="assignment",
            subject_id=assignment.id,
            from_state=assignment.state.value,
            to_state=updated.state.value,
            reason=ctx.reason,
            plan_version=ctx.mission.plan.version if ctx.mission else None,
            metadata={"mission_id": assignment.mission_id, "member_id": assignment.member_id},
        )
    )
    return updated


def offer_assignment(candidate: Assignment, ctx: Context) -> Assignment:
    """The ``offer`` transition, whose source state is None because the
    assignment does not exist yet.

    Handled separately rather than squeezed into ``_apply``: there is no prior
    object to read a version off, and ``candidate`` arrives already shaped as
    the assignment we intend to create so the guards can inspect it.
    """
    transition = ASSIGNMENT_TRANSITIONS[(None, "offer")]
    assert transition.permission is not None
    require(ctx.actor, transition.permission)
    enforce(transition.policies, ctx.actor, candidate, ctx)
    for guard in transition.guards:
        failure = guard(candidate, ctx)
        if failure:
            raise GuardFailed("offer", failure)

    created = replace(candidate, state=AS.OFFERED, version=1)
    for effect in transition.effects:
        created = effect(created, ctx)

    ctx.store.save_assignment(created)
    ctx.store.record_event(
        MissionEvent(
            at=ctx.now,
            actor_id=ctx.actor.member_id,
            actor_role=ctx.actor.role,
            event="offer",
            subject_kind="assignment",
            subject_id=created.id,
            from_state=None,
            to_state=created.state.value,
            reason=ctx.reason,
            plan_version=ctx.mission.plan.version if ctx.mission else None,
            metadata={
                "mission_id": created.mission_id,
                "member_id": created.member_id,
                "slot": f"{created.requirement_id}#{created.slot_index}",
            },
        )
    )
    return created


def _fire_assignment(
    assignment: Assignment,
    event: str,
    ctx: Context,
    mission: Mission,
    *,
    reason: str | None = None,
) -> Assignment:
    """Mission effects drive assignment events through here (§8.3).

    Declared once in the mission table rather than re-implemented at each call
    site, which is what makes "cancelling a mission releases its crew" a
    property of the design rather than of whoever wrote the handler.
    """
    return apply_assignment_event(
        assignment, event, replace(ctx, mission=mission, reason=reason)
    )


def available_events(mission: Mission, ctx: Context) -> tuple[str, ...]:
    """What this actor may do to this mission next — the single source of truth
    behind UI buttons, API responses and the error message in §6.4.

    Filters by source state, permission and policy but **not** by guards: a
    guard failure is a precondition to report ("you cannot submit yet, two
    mandatory slots are unfilled"), not a button to hide. System-triggered
    transitions are excluded because they are nobody's button.
    """
    out = []
    for (source, event), transition in MISSION_TRANSITIONS.items():
        if source is not mission.state or transition.permission is None:
            continue
        if not has(ctx.actor, transition.permission):
            continue
        if evaluate(transition.policies, ctx.actor, mission, ctx) is not None:
            continue
        out.append(event)
    return tuple(sorted(out))


def available_assignment_events(assignment: Assignment, ctx: Context) -> tuple[str, ...]:
    out = []
    for (source, event), transition in ASSIGNMENT_TRANSITIONS.items():
        if source is not assignment.state or transition.permission is None:
            continue
        if not has(ctx.actor, transition.permission):
            continue
        if evaluate(transition.policies, ctx.actor, assignment, ctx) is not None:
            continue
        out.append(event)
    return tuple(sorted(out))


def blocked_events(mission: Mission, ctx: Context) -> dict[str, str]:
    """``available_events`` minus the guards, with the reason each is not yet
    possible. What a UI needs to grey a button out *and* explain it."""
    out: dict[str, str] = {}
    for event in available_events(mission, ctx):
        transition = MISSION_TRANSITIONS[(mission.state, event)]
        for guard in transition.guards:
            failure = guard(mission, replace(ctx, mission=mission))
            if failure:
                out[event] = failure
                break
    return out
