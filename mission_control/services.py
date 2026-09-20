"""The operations a caller actually invokes.

Each one follows the same shape (§3.3): authorise, compute on an immutable
snapshot holding no lock, then commit in a tiny critical section where the
authoritative check happens. No method takes a ``tenant_id`` — scope came from
the Actor when the workspace was opened (§3.2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime

from . import matching
from .authz import Permission as P
from .authz import (
    OwnershipPolicy,
    SeparationOfDuties,
    enforce,
    has,
    require,
    visible_missions,
)
from .domain import (
    LIVE_ASSIGNMENT_STATES,
    METADATA_FIELDS,
    Assignment,
    AssignmentId,
    AssignmentState,
    CrewProfile,
    CrewSkill,
    CrewingStatus,
    Denied,
    GuardFailed,
    Member,
    MemberId,
    MemberStatus,
    Mission,
    MissionEvent,
    MissionId,
    MissionPlan,
    MissionState,
    NotFound,
    Role,
    Skill,
    SkillId,
    TenantSettings,
    TimeWindow,
    UnavailabilityBlock,
    as_utc,
    utc_now,
    conflicting_commitments,
)
from .lifecycle import (
    AllocationReport,
    Context,
    apply_assignment_event,
    available_events,
    blocked_events,
    crewing_status,
    offer_assignment,
    plan_changes,
    validate_allocation,
)
from .store import TenantWorkspace

Clock = Callable[[], datetime]





class _Service:
    """Shared plumbing: the workspace, a clock, and context construction."""

    def __init__(self, workspace: TenantWorkspace, clock: Clock | None = None) -> None:
        self.ws = workspace
        self.clock = clock or utc_now

    @property
    def actor(self):  # noqa: ANN201
        return self.ws.actor

    def _ctx(
        self,
        *,
        now: datetime | None = None,
        reason: str | None = None,
        params: Mapping[str, object] | None = None,
        mission: Mission | None = None,
    ) -> Context:
        return Context(
            store=self.ws,
            actor=self.ws.actor,
            now=as_utc(now) if now is not None else as_utc(self.clock()),
            reason=reason,
            params=params or {},
            mission=mission,
        )


# ---------------------------------------------------------------------- missions


class MissionService(_Service):
    def create(
        self,
        title: str,
        plan: MissionPlan,
        *,
        now: datetime | None = None,
        **metadata: object,
    ) -> Mission:
        """Only a Lead can do this. A Director holds no authoring permissions,
        which is what makes self-approval unrepresentable (§5.2)."""
        require(self.actor, P.MISSION_CREATE)
        unknown = set(metadata) - METADATA_FIELDS
        if unknown:
            raise ValueError(f"not metadata fields: {sorted(unknown)}")

        at = now or self.clock()
        with self.ws.transaction():
            mission = Mission(
                id=MissionId(self.ws.new_id("mis")),
                title=title,
                plan=plan,
                created_by=self.actor.member_id,
                state=MissionState.DRAFT,
                **metadata,  # type: ignore[arg-type]
            )
            self.ws.save_mission(mission)
            self.ws.record_event(
                MissionEvent(
                    at=at, actor_id=self.actor.member_id, actor_role=self.actor.role,
                    event="create", subject_kind="mission", subject_id=mission.id,
                    from_state=None, to_state=mission.state.value,
                    plan_version=mission.plan.version,
                )
            )
            return mission

    def update_plan(
        self, mission_id: MissionId, new_plan: MissionPlan, *, now: datetime | None = None
    ) -> Mission:
        """The only route to a plan field, and therefore the only thing that can
        emit ``plan_edited`` (§6.1.1).

        **The plan is editable in DRAFT and nowhere else.** Calling this on a
        submitted or approved mission raises ``IllegalTransition`` naming
        ``reopen`` as the way forward — enforced by the absence of a row in the
        transition table, not by a guard written here that a future call site
        could forget.

        "You cannot approve something that changed under you" therefore holds
        for the strongest available reason: while a Director is reading a plan,
        it cannot change at all.
        """
        with self.ws.transaction():
            mission = self.ws.mission(mission_id)
            changed = plan_changes(mission.plan, new_plan)
            if not changed:
                return mission  # nothing to edit, so nothing to check
            return self._fire(
                mission, "plan_edited", now=now,
                params={
                    "new_plan": new_plan,
                    "changed": changed,
                    # The effect compares signatures rather than field names, so
                    # it needs the plan as it stood, not a list of what differs.
                    "previous_plan": mission.plan,
                    "metadata": {
                        "changed": sorted(changed),
                        "rematch_needed": (
                            mission.plan.matching_signature()
                            != new_plan.matching_signature()
                        ),
                    },
                },
            )

    def update_metadata(
        self, mission_id: MissionId, *, now: datetime | None = None, **fields: object
    ) -> Mission:
        """Metadata edits do **not** revert the mission.

        Read literally, "any edit reverts to DRAFT" would un-approve a mission
        for a typo fix, which trains people not to document anything (§6.1.1).
        The split is structural: plan fields live on MissionPlan, so a field
        added to Mission is metadata by default — and this method refuses
        anything not on that list rather than trusting the caller.
        """
        require(self.actor, P.MISSION_UPDATE)
        unknown = set(fields) - METADATA_FIELDS
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} are plan fields or unknown; use update_plan()"
            )

        at = now or self.clock()
        with self.ws.transaction():
            mission = self.ws.mission(mission_id)
            if mission.is_terminal:
                raise GuardFailed("update_metadata", f"mission is {mission.state.value}")
            enforce((OwnershipPolicy(),), self.actor, mission, self._ctx(now=at))
            updated = replace(mission, version=mission.version + 1, **fields)  # type: ignore[arg-type]
            self.ws.save_mission(updated)
            self.ws.record_event(
                MissionEvent(
                    at=at, actor_id=self.actor.member_id, actor_role=self.actor.role,
                    event="metadata_edited", subject_kind="mission", subject_id=mission.id,
                    from_state=mission.state.value, to_state=updated.state.value,
                    plan_version=updated.plan.version,
                    metadata={"fields": sorted(fields)},
                )
            )
            return updated

    # -- lifecycle events

    def _fire(
        self,
        mission: Mission,
        event: str,
        *,
        now: datetime | None = None,
        reason: str | None = None,
        params: Mapping[str, object] | None = None,
    ) -> Mission:
        from .lifecycle import apply_mission_event

        return apply_mission_event(
            mission, event, self._ctx(now=now, reason=reason, params=params)
        )

    def fire(
        self,
        mission_id: MissionId,
        event: str,
        *,
        now: datetime | None = None,
        reason: str | None = None,
    ) -> Mission:
        """Generic entry point. ``approve`` is routed through :meth:`approve`
        because its gate can choose a different event (§6.8).

        **System-triggered events are not reachable from here.** They carry no
        permission — because nobody decided them — so exposing them on a request
        path would let any caller fire them: a Crew member could bounce someone
        else's PENDING_APPROVAL mission with ``invalidated``. They live on
        SystemService and inside :meth:`approve` instead.
        """
        if event == "approve":
            return self.approve(mission_id, now=now)
        from .lifecycle import MISSION_TRANSITIONS

        with self.ws.transaction():
            mission = self.ws.mission(mission_id)
            transition = MISSION_TRANSITIONS.get((mission.state, event))
            if transition is not None and transition.permission is None:
                raise Denied(
                    "SYSTEM_EVENT", f"{event!r} is triggered by the system, not by a caller"
                )
            return self._fire(mission, event, now=now, reason=reason)

    def submit(self, mission_id: MissionId, *, now: datetime | None = None) -> Mission:
        return self.fire(mission_id, "submit", now=now)

    def reopen(self, mission_id: MissionId, *, now: datetime | None = None) -> Mission:
        """Pull a submitted or approved mission back to DRAFT so it can be
        edited.

        The plan is frozen in every state but DRAFT, so this is the way to
        change one. Reopening an APPROVED mission discards the approval — the
        Director authorised a specific plan, and the Lead is about to stop that
        being the plan (§6.1.1).
        """
        return self.fire(mission_id, "reopen", now=now)

    def request_changes(
        self, mission_id: MissionId, reason: str, *, now: datetime | None = None
    ) -> Mission:
        return self.fire(mission_id, "request_changes", reason=reason, now=now)

    def activate(self, mission_id: MissionId, *, now: datetime | None = None) -> Mission:
        """If the gate is broken the Lead has exactly two moves: ``cancel``, or
        edit the plan to return to DRAFT and re-crew. Nobody can waive a missing
        mandatory slot — and since Directors hold no authoring permissions there
        is not even a role that could own such an override (§6.8)."""
        return self.fire(mission_id, "activate", now=now)

    def complete(
        self, mission_id: MissionId, *, reason: str | None = None, now: datetime | None = None
    ) -> Mission:
        return self.fire(mission_id, "complete", reason=reason, now=now)

    def abort(
        self, mission_id: MissionId, reason: str, *, now: datetime | None = None
    ) -> Mission:
        return self.fire(mission_id, "abort", reason=reason, now=now)

    def cancel(
        self, mission_id: MissionId, reason: str, *, now: datetime | None = None
    ) -> Mission:
        return self.fire(mission_id, "cancel", reason=reason, now=now)

    def approve(self, mission_id: MissionId, *, now: datetime | None = None) -> Mission:
        """The §6.8 gate, then the transition.

        Authorisation is checked *first*, so a caller who could never approve
        cannot use this to bounce someone else's mission back to DRAFT. Then the
        roster is re-validated: if a mandatory slot no longer holds, or crewing
        is no longer complete, the mission **returns to DRAFT** rather than the
        Director getting an error — the same destination as every other
        backwards transition, since there is exactly one editable state.
        """
        at = now or self.clock()
        with self.ws.transaction():
            mission = self.ws.mission(mission_id)
            require(self.actor, P.MISSION_APPROVE)
            enforce((SeparationOfDuties(),), self.actor, mission, self._ctx(now=at))
            if mission.state is not MissionState.PENDING_APPROVAL:
                return self._fire(mission, "approve", now=at)  # raises IllegalTransition

            assignments = self.ws.assignments_for_mission(mission_id)
            report = validate_allocation(mission, self.ws.snapshot(), assignments)
            status = crewing_status(mission, assignments)

            if report.blocking or status is not CrewingStatus.FULLY_CREWED:
                why = report.summary() or f"mission is {status.value}"
                return self._fire(
                    mission, "invalidated", now=at, reason=why,
                    params={"report": report, "metadata": {
                        "stale_slots": [f"{s.slot[0]}#{s.slot[1]}" for s in report.stale],
                        "crewing_status": status.value,
                    }},
                )
            return self._fire(
                mission, "approve", now=at,
                params={"metadata": {
                    # Who was approved, recorded on the event rather than frozen
                    # into the plan: it is audit truth, and the plan holding
                    # Assignment objects would make every acceptance a plan edit.
                    "approved_roster": sorted(
                        a.member_id for a in assignments
                        if a.state is AssignmentState.ACCEPTED
                    ),
                }},
            )

    # -- reads

    def get(self, mission_id: MissionId) -> Mission:
        """Direct access to a non-visible mission raises NotFound, not
        Forbidden (§5.3)."""
        mission = self.ws.mission(mission_id)
        if not has(self.actor, P.MISSION_VIEW_ALL):
            mine = any(
                a.member_id == self.actor.member_id
                for a in self.ws.assignments_for_mission(mission_id)
            )
            if not mine:
                raise NotFound("mission", mission_id)
        return mission

    def list(self) -> list[Mission]:
        """Listing *filters* rather than rejecting (§5.3)."""
        return list(visible_missions(self.actor, self._ctx(), self.ws.missions()))

    def available_events(self, mission_id: MissionId) -> tuple[str, ...]:
        return available_events(self.get(mission_id), self._ctx())

    def blocked_events(self, mission_id: MissionId) -> dict[str, str]:
        """What this actor may do but cannot do *yet*, and why — the reason a UI
        can grey a button out and explain it."""
        return blocked_events(self.get(mission_id), self._ctx())

    def crewing_status(self, mission_id: MissionId) -> CrewingStatus:
        return crewing_status(
            self.get(mission_id), self.ws.assignments_for_mission(mission_id)
        )

    def allocation_report(self, mission_id: MissionId) -> AllocationReport:
        return validate_allocation(
            self.get(mission_id),
            self.ws.snapshot(),
            self.ws.assignments_for_mission(mission_id),
        )

    def roster(self, mission_id: MissionId) -> list[Assignment]:
        """The roster, derived rather than stored (§6.5 note in domain.py)."""
        return [
            a
            for a in self.ws.assignments_for_mission(mission_id)
            if a.state in LIVE_ASSIGNMENT_STATES
        ]


# ---------------------------------------------------------------------- matching


class MatchingService(_Service):
    def run_matcher(
        self,
        mission_id: MissionId,
        *,
        team_constraints: Sequence[matching.TeamConstraint] = (),
        now: datetime | None = None,
    ) -> matching.MatchProposal:
        """Authorise and audit under the lock; **match outside it** (§3.3).

        The matcher is not a mutation — it returns a proposal, and nothing is
        assigned until a Lead acts. The lock would have to span a human
        decision anyway (the Lead reviews, then offers, minutes or hours later),
        so optimistic validation is not merely preferable, it is the only thing
        that works. A matcher reading slightly stale availability is therefore
        **accepted**; the worst case is a proposal row that turns out
        unofferable, and ``offer`` refuses it.
        """
        at = now or self.clock()
        with self.ws.transaction():
            mission = self.ws.mission(mission_id)
            # The run_matcher row in the mission table carries the permission,
            # the ownership policy and the DRAFT-only source state. It has no
            # effects: the matcher never creates assignments (§7.9).
            self._fire_matcher_event(mission, at)
            snap = self.ws.snapshot()

        return matching.match(
            mission, snap, self.ws.settings,
            now=at, team_constraints=team_constraints,
        )

    def _fire_matcher_event(self, mission: Mission, at: datetime) -> Mission:
        from .lifecycle import apply_mission_event

        return apply_mission_event(mission, "run_matcher", self._ctx(now=at))

    def offer_proposal(
        self,
        mission_id: MissionId,
        proposal: matching.MatchProposal,
        *,
        slots: Sequence[tuple] | None = None,
        now: datetime | None = None,
    ) -> list[Assignment]:
        """Turn proposal rows into offers — the Lead's act, not the matcher's.

        Accepts the proposal whole or slot by slot. Each row is re-validated at
        this point, so a stale proposal costs a refused row rather than a bad
        assignment. Already-pinned rows are skipped: they are people who have
        already accepted.
        """
        wanted = set(slots) if slots is not None else None
        assignments = AssignmentService(self.ws, self.clock)
        created: list[Assignment] = []
        for row in proposal.slots:
            if row.pinned:
                continue
            if wanted is not None and row.slot not in wanted:
                continue
            created.append(
                assignments.offer(
                    mission_id, row.requirement_id, row.slot_index, row.member_id, now=now
                )
            )
        return created


# ------------------------------------------------------------------- assignments


class AssignmentService(_Service):
    def offer(
        self,
        mission_id: MissionId,
        requirement_id,
        slot_index: int,
        member_id: MemberId,
        *,
        now: datetime | None = None,
    ) -> Assignment:
        at = now or self.clock()
        with self.ws.transaction():
            mission = self.ws.mission(mission_id)
            # Never past mission start: an offer that could be accepted after
            # the mission began is meaningless (§8.2).
            expires = min(at + self.ws.settings.offer_ttl, mission.plan.window.start)
            candidate = Assignment(
                id=AssignmentId(self.ws.new_id("asg")),
                mission_id=mission_id,
                member_id=member_id,
                requirement_id=requirement_id,
                slot_index=slot_index,
                state=AssignmentState.OFFERED,
                offered_at=at,
                offer_expires_at=expires,
            )
            return offer_assignment(candidate, self._ctx(now=at, mission=mission))

    def accept(self, assignment_id: AssignmentId, *, now: datetime | None = None) -> Assignment:
        """The one operation where conflict-check and write must be atomic per
        crew member (§3.3). Everything about it happens inside the lock."""
        return self._respond(assignment_id, "accept", now=now)

    def decline(self, assignment_id: AssignmentId, *, now: datetime | None = None) -> Assignment:
        return self._respond(assignment_id, "decline", now=now)

    def _respond(
        self, assignment_id: AssignmentId, event: str, *, now: datetime | None = None
    ) -> Assignment:
        with self.ws.transaction():
            assignment = self.ws.assignment(assignment_id)
            if assignment.member_id != self.actor.member_id and not has(
                self.actor, P.MISSION_VIEW_ALL
            ):
                # Someone else's assignment is not theirs to see, let alone
                # answer. NotFound rather than Denied, for the same reason
                # foreign tenant ids are.
                raise NotFound("assignment", assignment_id)
            mission = self.ws.mission(assignment.mission_id)
            return apply_assignment_event(
                assignment, event, self._ctx(now=now, mission=mission)
            )

    def mine(self) -> list[Assignment]:
        return self.ws.assignments_for_member(self.actor.member_id)

    def for_mission(self, mission_id: MissionId) -> list[Assignment]:
        MissionService(self.ws, self.clock).get(mission_id)  # visibility check
        return self.ws.assignments_for_mission(mission_id)


# ------------------------------------------------------------------------- crew


class CrewService(_Service):
    def set_skills(
        self,
        member_id: MemberId,
        skills: Sequence[CrewSkill],
        *,
        now: datetime | None = None,
    ) -> CrewProfile:
        """Crew write their own; a Director may correct anyone's.

        A downward revision is one of the two things the approval gate exists to
        catch (§6.8) — it is deliberately *not* blocked here, because an
        out-of-date record is worse than a bounced mission.
        """
        self._authorise_profile_write(member_id, P.CREW_PROFILE_WRITE_OWN)
        at = now or self.clock()
        with self.ws.transaction():
            profile = self._profile_or_new(member_id)
            updated = replace(profile, skills=tuple(skills), version=profile.version + 1)
            self.ws.save_crew_profile(updated)
            self._record(member_id, "skills_updated", at, {"count": len(skills)})
            return updated

    def set_unavailability(
        self,
        member_id: MemberId,
        blocks: Sequence[UnavailabilityBlock],
        *,
        now: datetime | None = None,
    ) -> CrewProfile:
        """**Rejected outright if it conflicts with an ACCEPTED assignment**
        (§3.3, §8.2).

        Acceptance is final in this design, so the alternative — letting the
        declaration win — would silently strand a mission whose Director has
        already approved that crew. The write fails naming the mission, and the
        crew member's route out is to talk to the Lead.
        """
        self._authorise_profile_write(member_id, P.CREW_AVAILABILITY_WRITE_OWN)
        at = now or self.clock()
        with self.ws.transaction():
            profile = self._profile_or_new(member_id)
            snap = self.ws.snapshot()
            for block in blocks:
                clashes = conflicting_commitments(profile, block.window, snap)
                if clashes:
                    names = ", ".join(
                        f"{snap.missions[c.mission_id].title!r} "
                        f"({snap.missions[c.mission_id].plan.window})"
                        for c in clashes
                    )
                    raise GuardFailed(
                        "set_unavailability",
                        f"{block.window} conflicts with accepted assignment(s) on {names}",
                    )
            updated = replace(
                profile, unavailability=tuple(blocks), version=profile.version + 1
            )
            self.ws.save_crew_profile(updated)
            self._record(member_id, "availability_updated", at, {"blocks": len(blocks)})
            return updated

    def profile(self, member_id: MemberId) -> CrewProfile:
        if member_id != self.actor.member_id and not has(self.actor, P.CREW_PROFILE_READ_ALL):
            raise NotFound("crew profile", member_id)
        return self.ws.crew_profile(member_id)

    def _authorise_profile_write(self, member_id: MemberId, own_permission) -> None:
        if member_id == self.actor.member_id:
            require(self.actor, own_permission)
            return
        if has(self.actor, P.MEMBER_MANAGE):
            return
        raise Denied(
            "NOT_YOUR_PROFILE", "you can only change your own profile and availability"
        )

    def _profile_or_new(self, member_id: MemberId) -> CrewProfile:
        member = self.ws.member(member_id)
        if member.role is not Role.CREW:
            raise GuardFailed(
                "profile_write", f"{member.name} holds the {member.role.value} role, not crew"
            )
        try:
            return self.ws.crew_profile(member_id)
        except NotFound:
            return CrewProfile(member_id=member_id, version=0)

    def _record(self, member_id: MemberId, event: str, at: datetime, meta: dict) -> None:
        self.ws.record_event(
            MissionEvent(
                at=at, actor_id=self.actor.member_id, actor_role=self.actor.role,
                event=event, subject_kind="crew_profile", subject_id=member_id,
                from_state=None, to_state="updated", metadata=meta,
            )
        )


# ----------------------------------------------------------------- organisation


class OrgService(_Service):
    """Director-only. Governance, not authorship (§5.2)."""

    def add_member(
        self, name: str, role: Role, *, now: datetime | None = None
    ) -> Member:
        require(self.actor, P.MEMBER_MANAGE)
        at = now or self.clock()
        with self.ws.transaction():
            member = Member(id=MemberId(self.ws.new_id("mem")), name=name, role=role)
            self.ws.save_member(member)
            if role is Role.CREW:
                self.ws.save_crew_profile(CrewProfile(member_id=member.id))
            self.ws.record_event(
                MissionEvent(
                    at=at, actor_id=self.actor.member_id, actor_role=self.actor.role,
                    event="member_added", subject_kind="member", subject_id=member.id,
                    from_state=None, to_state=MemberStatus.ACTIVE.value,
                    metadata={"role": role.value},
                )
            )
            return member

    def deactivate_member(
        self, member_id: MemberId, reason: str, *, now: datetime | None = None
    ) -> Member:
        """Releases their live assignments.

        §8.1 lists deactivation among the things ``ACCEPTED -> RELEASED`` exists
        for, while §6.8 describes a deactivated member being caught by the
        approval gate. Both cannot be literally true. Releasing is the better
        reading: the person is gone, so holding the slot is a lie and blocking
        their availability is worse. The gate still catches the case, via
        crewing status rather than a stale slot — which is why
        :meth:`MissionService.approve` checks both.
        """
        require(self.actor, P.MEMBER_MANAGE)
        at = now or self.clock()
        with self.ws.transaction():
            member = self.ws.member(member_id)
            updated = replace(
                member, status=MemberStatus.DEACTIVATED, version=member.version + 1
            )
            self.ws.save_member(updated)
            for assignment in self.ws.assignments_for_member(member_id):
                if assignment.state not in LIVE_ASSIGNMENT_STATES:
                    continue
                mission = self.ws.mission(assignment.mission_id)
                if mission.is_terminal:
                    continue
                apply_assignment_event(
                    assignment, "release",
                    self._ctx(now=at, mission=mission,
                              reason=f"{member.name} deactivated: {reason}"),
                )
            self.ws.record_event(
                MissionEvent(
                    at=at, actor_id=self.actor.member_id, actor_role=self.actor.role,
                    event="member_deactivated", subject_kind="member", subject_id=member_id,
                    from_state=MemberStatus.ACTIVE.value,
                    to_state=MemberStatus.DEACTIVATED.value, reason=reason,
                )
            )
            return updated

    def define_skill(self, name: str, *, skill_id: SkillId | None = None) -> Skill:
        require(self.actor, P.ORG_SETTINGS_MANAGE)
        with self.ws.transaction():
            skill = Skill(id=skill_id or SkillId(self.ws.new_id("skl")), name=name)
            self.ws.save_skill(skill)
            return skill

    def update_settings(self, settings: TenantSettings) -> TenantSettings:
        require(self.actor, P.ORG_SETTINGS_MANAGE)
        self.ws.save_settings(settings)
        return settings

    def audit(self) -> list[MissionEvent]:
        require(self.actor, P.AUDIT_READ)
        return self.ws.audit()


# ----------------------------------------------------------------- system sweeps


class SystemService(_Service):
    """The two clock-driven transitions.

    In production these run as a scheduled job with a synthetic actor rather
    than on a request. Gated on MISSION_VIEW_ALL here so a Crew member cannot
    trigger a tenant-wide sweep — the transitions themselves carry no
    permission, because nobody *decided* them (§6.2).
    """

    def expire_offers(self, *, now: datetime | None = None) -> list[Assignment]:
        """An unanswered offer becomes a decline. Harmless, because it stalls a
        draft rather than an approved mission (§6.7)."""
        require(self.actor, P.MISSION_VIEW_ALL)
        at = now or self.clock()
        expired: list[Assignment] = []
        with self.ws.transaction():
            for assignment in list(self.ws.snapshot().assignments.values()):
                if assignment.state is not AssignmentState.OFFERED:
                    continue
                if at <= assignment.offer_expires_at:
                    continue
                mission = self.ws.mission(assignment.mission_id)
                expired.append(
                    apply_assignment_event(
                        assignment, "offer_expired",
                        self._ctx(now=at, mission=mission, reason="offer expired unanswered"),
                    )
                )
        return expired

    def expire_pending_approvals(self, *, now: datetime | None = None) -> list[Mission]:
        """Bounds how long a fully-crewed mission can wait on a Director while
        holding real people (§6.7)."""
        require(self.actor, P.MISSION_VIEW_ALL)
        from .lifecycle import apply_mission_event

        at = now or self.clock()
        expired: list[Mission] = []
        with self.ws.transaction():
            for mission in self.ws.missions():
                if mission.state is not MissionState.PENDING_APPROVAL:
                    continue
                if mission.pending_expires_at is None or at <= mission.pending_expires_at:
                    continue
                expired.append(
                    apply_mission_event(
                        mission, "expire",
                        self._ctx(now=at, reason="approval window expired"),
                    )
                )
        return expired

    def long_held_drafts(
        self, older_than: TimeWindow | None = None
    ) -> list[tuple[Mission, int]]:
        """**Deferred feature, stubbed to show the shape** (§6.7).

        Nothing bounds the pre-submission window: a Lead can crew a mission in
        DRAFT and leave it, dates hard-blocked, no clock running. The intended
        mechanism is a tenant-configured ``crew_hold_review_after``, a Lead
        notification with one-click release, and this Director-visible report.

        **Notification, not auto-release**: the crew *accepted*, so silently
        cancelling their commitment because a Lead was slow is worse than the
        hoarding.
        """
        require(self.actor, P.MISSION_VIEW_ALL)
        out = []
        for mission in self.ws.missions():
            if mission.state is not MissionState.DRAFT:
                continue
            held = sum(
                1
                for a in self.ws.assignments_for_mission(mission.id)
                if a.state is AssignmentState.ACCEPTED
            )
            if held:
                out.append((mission, held))
        return out


# --------------------------------------------------------------------- facade


class Scope:
    """Every operation available to one authenticated Actor, over one tenant.

    **Request-scoped, not a session.** Nothing here is retained between calls:
    an Actor is derived from the caller's credentials, a Scope is built over it,
    the work happens, and the Scope is discarded. The only state that outlives a
    request is the Platform itself (§3.1), which every Scope reaches through its
    own TenantWorkspace.

    Named Scope rather than Session precisely because an HTTP layer sits above
    this one, where "session" would mean server-side state keyed by a cookie —
    which this design deliberately does not have.
    """

    def __init__(self, workspace: TenantWorkspace, clock: Clock | None = None) -> None:
        self.workspace = workspace
        self.actor = workspace.actor
        self.missions = MissionService(workspace, clock)
        self.matching = MatchingService(workspace, clock)
        self.assignments = AssignmentService(workspace, clock)
        self.crew = CrewService(workspace, clock)
        self.org = OrgService(workspace, clock)
        self.system = SystemService(workspace, clock)
