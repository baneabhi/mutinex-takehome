"""The matching engine (§7).

The task is not "rank crew for this mission". It is: **fill the slots across
several requirements, from one shared pool, where each person occupies at most
one slot** — an assignment problem, not a ranking problem (§7.1).

Pipeline (§7.2)::

    1. Candidate generation   hard filters, per requirement   -> eligible sets
    2. Ranking                (load, last flew, id)           -> dense rank per set
    3. Team assembly          global optimal assignment       -> slot allocation
    4. Team constraints       mission-level rules, repair     -> feasible team
    5. Explanation            reasons + near-misses           -> MatchProposal

Pure functions over domain objects and a snapshot. Creates nothing: the matcher
returns a proposal and a Lead acts on it (§7.9).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from . import solver
from .domain import (
    FILTER_ACTIVE_MEMBER,
    FILTER_AVAILABILITY,
    FILTER_DECLINED,
    FILTER_SKILLS,
    HISTORY_ASSIGNMENT_STATES,
    Assignment,
    AssignmentId,
    AssignmentState,
    CrewProfile,
    Member,
    MemberId,
    Mission,
    MissionId,
    MissionPlan,
    Proficiency,
    Requirement,
    RequirementId,
    Role,
    SkillId,
    SlotKey,
    EPOCH,
    TenantSettings,
    conflicting_commitments,
    eligibility_failures,
    is_available,
    skill_shortfall,
)


class MatchSnapshot(Protocol):
    """An immutable read of one tenant. The matcher holds no lock: it computes
    on this and the authoritative check happens later, at ``offer`` (§3.3)."""

    members: Mapping[MemberId, Member]
    crew: Mapping[MemberId, CrewProfile]
    missions: Mapping[MissionId, Mission]

    def assignments_for(self, member_id: MemberId) -> Sequence[Assignment]: ...
    def assignments_for_mission(self, mission_id: MissionId) -> Sequence[Assignment]: ...


# ------------------------------------------------------------------------ slots


@dataclass(frozen=True)
class Slot:
    """One seat. ``count: 3`` becomes three slots (§7.5)."""

    requirement: Requirement
    index: int

    @property
    def key(self) -> SlotKey:
        return (self.requirement.id, self.index)

    @property
    def label(self) -> str:
        return self.requirement.label


def expand_slots(plan: MissionPlan) -> tuple[Slot, ...]:
    return tuple(
        Slot(requirement, index)
        for requirement in plan.requirements
        for index in range(requirement.count)
    )


# ---------------------------------------------------------------------- ranking


def load(
    member_id: MemberId, snap: MatchSnapshot, *, now: datetime, lookback: timedelta
) -> int:
    """How much work this person has recently had or is already promised (§7.8).

    **It counts commitments, not just history.** Counting only COMPLETED means
    someone who accepted five missions for next month still looks idle and gets
    offered a sixth. Work already promised is opportunity already received —
    and unlike the history term it needs no window, because an outstanding
    commitment is current by definition.

    **It anchors on ``now``, not the mission window** — the one place in this
    design that does not use mission-relative time, deliberately. A window
    anchored on a mission six months out would look back to dates three months
    from now, miss the twenty missions someone flew this year, and rank them as
    idle. Fairness is about a person's recent past.

    DECLINED, RELEASED and OFFERED do not count: being *asked* is not being
    given work.
    """
    done = 0
    committed = 0
    for assignment in snap.assignments_for(member_id):
        if assignment.state in HISTORY_ASSIGNMENT_STATES:
            mission = snap.missions.get(assignment.mission_id)
            if mission is not None and mission.plan.window.end >= now - lookback:
                done += 1
        elif assignment.state is AssignmentState.ACCEPTED:
            committed += 1
    return done + committed


def last_mission_end(member_id: MemberId, snap: MatchSnapshot) -> datetime | None:
    ends = [
        snap.missions[a.mission_id].plan.window.end
        for a in snap.assignments_for(member_id)
        if a.state in HISTORY_ASSIGNMENT_STATES and a.mission_id in snap.missions
    ]
    return max(ends) if ends else None


def rank_key(
    profile: CrewProfile, snap: MatchSnapshot, *, now: datetime, lookback: timedelta
) -> tuple[int, datetime, str]:
    """The whole ranking model — no scorers, no weights, no ``[0, 1]`` floats.

    **There is no fit score, deliberately** (§7.4). The requirement *is* the
    statement of suitability: if a tenant wants an Expert they write
    ``min_level: Expert``. Ranking by surplus above the bar treats "more than
    requested" as "better", which is a soft preference, and soft preferences
    need weights nobody can source. A number you cannot source is not
    configuration, it is a placeholder pretending to be a decision. So
    overqualification is neither rewarded nor penalised — it is not consulted.

    **Why load rather than level.** Preferring the least-qualified eligible
    candidate to conserve talent is the obvious alternative and the wrong
    mechanism: within a mission the solver already avoids spending an Expert
    needed elsewhere (§7.6); across missions there is no lookahead to the harder
    mission being saved for; and it gives the most skilled people the least work,
    eroding the bench it was protecting.

    **Key 2 replaces arbitrary tie-breaking.** Load-and-id alone resolves ties
    alphabetically, a systematic bias toward whoever sorts first. Breaking on
    when someone last flew uses the mission dates already present.
    ``EPOCH`` for "never flown" sorts new crew first.
    """
    return (
        load(profile.member_id, snap, now=now, lookback=lookback),
        last_mission_end(profile.member_id, snap) or EPOCH,
        profile.member_id,
    )


def rank_keys(
    profiles: Iterable[CrewProfile],
    snap: MatchSnapshot,
    *,
    now: datetime,
    lookback: timedelta,
) -> dict[MemberId, tuple[int, datetime, str]]:
    """Every member's rank key, computed **once**.

    ``rank_key`` depends only on the member — not on the requirement — so
    computing it inside the per-requirement loop recomputed it once per
    ``(member, requirement)`` pair, and each call walks that member's assignment
    index twice (``load`` and ``last_mission_end``). At a hundred requirements
    that was a hundredfold waste and it dominated the run.
    """
    return {
        profile.member_id: rank_key(profile, snap, now=now, lookback=lookback)
        for profile in profiles
    }


def costs_for(
    eligible: Iterable[CrewProfile],
    keys: Mapping[MemberId, tuple[int, datetime, str]],
) -> dict[MemberId, int]:
    """Dense integer rank, which *is* the solver's cost (§7.5).

    Rank-as-cost is what lets ``rank_key``'s tiebreak chain be an ordinary sort
    key. Packing the keys into one integer with magnitude multipliers would
    require every term's range to be bounded and clamped, and breaks the moment
    a key is added.

    Takes precomputed keys from :func:`rank_keys` rather than computing them,
    because it is called once per requirement over overlapping candidate sets.
    """
    ordered = sorted(eligible, key=lambda p: keys[p.member_id])
    return {profile.member_id: rank for rank, profile in enumerate(ordered)}


# ------------------------------------------------------------ team constraints


class TeamConstraint(Protocol):
    """A property of the *team* rather than of any one person.

    These break linear assignment, because the cost of assigning someone
    depends on who else was assigned (§7.7).
    """

    @property
    def label(self) -> str: ...

    def violation(
        self, team: Mapping[SlotKey, MemberId], snap: MatchSnapshot
    ) -> str | None: ...

    def satisfiers(self, snap: MatchSnapshot) -> set[MemberId]: ...


@dataclass(frozen=True)
class AtLeastAtLevel:
    """"At least one member at Proficient or above in Emergency Medicine"."""

    skill_id: SkillId
    min_level: Proficiency
    count: int = 1

    @property
    def label(self) -> str:
        return (
            f"at least {self.count} team member(s) at {self.min_level.name} "
            f"or above in {self.skill_id}"
        )

    def _holds(self, member_id: MemberId, snap: MatchSnapshot) -> bool:
        profile = snap.crew.get(member_id)
        if profile is None:
            return False
        level = profile.level(self.skill_id)
        return level is not None and level >= self.min_level

    def violation(
        self, team: Mapping[SlotKey, MemberId], snap: MatchSnapshot
    ) -> str | None:
        found = sum(1 for member_id in team.values() if self._holds(member_id, snap))
        if found >= self.count:
            return None
        return f"{self.label} — team has {found}"

    def satisfiers(self, snap: MatchSnapshot) -> set[MemberId]:
        return {member_id for member_id in snap.crew if self._holds(member_id, snap)}


# ------------------------------------------------------------------- the output


@dataclass(frozen=True)
class SkillEvidence:
    """Per-slot reasons, not a score. The engine treats everyone who clears the
    bar as equally qualified, so the levels are exactly what a Lead needs to
    apply the judgement it deliberately does not (§7.9)."""

    skill_id: SkillId
    required: Proficiency
    held: Proficiency


@dataclass(frozen=True)
class SlotProposal:
    requirement_id: RequirementId
    slot_index: int
    label: str
    member_id: MemberId
    member_name: str
    rank: int
    recent_load: int
    skills: tuple[SkillEvidence, ...]
    pinned: bool
    """True when this person had already accepted this slot and a re-match kept
    them there rather than re-asking (§6.7)."""
    alternates: tuple[MemberId, ...] = ()
    """Next-best eligible crew, ranked. Advisory only, never part of the plan —
    they let a Lead re-offer immediately when someone declines during DRAFT."""

    @property
    def slot(self) -> SlotKey:
        return (self.requirement_id, self.slot_index)


@dataclass(frozen=True)
class UnfilledSlot:
    requirement_id: RequirementId
    slot_index: int
    label: str
    mandatory: bool
    reason: str

    @property
    def slot(self) -> SlotKey:
        return (self.requirement_id, self.slot_index)


@dataclass(frozen=True)
class NearMiss:
    """Crew excluded by *exactly one* filter, naming it (§7.9).

    "Three people qualify but are already committed to Kepler that week" turns
    a matching result into a scheduling decision, and makes over-constrained
    requirements self-diagnosing.
    """

    member_id: MemberId
    member_name: str
    requirement_id: RequirementId
    filter: str
    detail: str


@dataclass(frozen=True)
class MatchProposal:
    """**The matcher never creates assignments.**

    Auto-assignment would be a small convenience and a large mistake — this
    allocates work to people, and a human should own that. It also gives the
    Lead somewhere to apply context the engine does not have (§7.9).
    """

    mission_id: MissionId
    generated_at: datetime
    slots: tuple[SlotProposal, ...]
    unfilled: tuple[UnfilledSlot, ...]
    near_misses: tuple[NearMiss, ...]
    team_warnings: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not any(slot.mandatory for slot in self.unfilled)


# ---------------------------------------------------------------- the pipeline


def match(
    mission: Mission,
    snap: MatchSnapshot,
    settings: TenantSettings,
    *,
    now: datetime,
    team_constraints: Sequence[TeamConstraint] = (),
    max_repair_rounds: int = 4,
) -> MatchProposal:
    """Propose a team for ``mission``.

    Deterministic: integer costs plus ``member_id`` as the final tiebreak mean
    identical inputs give identical teams, with no floating-point associativity
    to make the total depend on summation order. ``now`` is captured once by the
    caller and threaded through (§7.5).
    """
    plan = mission.plan
    window = plan.window
    slots = expand_slots(plan)

    own_assignments = list(snap.assignments_for_mission(mission.id))
    own_ids: set[AssignmentId] = {a.id for a in own_assignments}
    pinned: dict[SlotKey, MemberId] = {
        a.slot: a.member_id for a in own_assignments if a.state is AssignmentState.ACCEPTED
    }

    # --- 1. candidate generation (§7.3)
    #
    # Availability is mission-level and computed once per crew member;
    # eligibility is per (crew, requirement), because one mission-level set
    # would collapse a Pilot and a Flight Surgeon together and lose the scarcity
    # information §7.6 depends on. A member's own commitment to *this* mission
    # must not make them unavailable for it.
    pool: dict[MemberId, tuple[Member, CrewProfile]] = {
        member_id: (member, snap.crew[member_id])
        for member_id, member in snap.members.items()
        if member.role is Role.CREW and member_id in snap.crew
    }
    availability = {
        member_id: is_available(profile, window, snap, ignoring=own_ids)
        for member_id, (_, profile) in pool.items()
    }

    # Declining is information; proposing the same person again discards it and
    # never terminates. Mission-scoped, so it is applied here rather than in
    # eligibility_failures (see FILTER_DECLINED).
    declined: set[MemberId] = {
        a.member_id for a in own_assignments if a.state is AssignmentState.DECLINED
    }

    failures: dict[tuple[RequirementId, MemberId], tuple[str, ...]] = {}
    eligible: dict[RequirementId, list[CrewProfile]] = {}
    for requirement in plan.requirements:
        ok: list[CrewProfile] = []
        for member_id, (member, profile) in pool.items():
            failed = eligibility_failures(
                requirement, member, profile, window, snap,
                available=availability[member_id],
            )
            if member_id in declined:
                failed = failed + (FILTER_DECLINED,)
            failures[(requirement.id, member_id)] = failed
            if not failed:
                ok.append(profile)
        eligible[requirement.id] = ok

    # --- 2. ranking (§7.4)
    #
    # One key per member, then a dense rank per requirement over it. The key is
    # a property of the person, not of the slot they are being considered for.
    keys = rank_keys(
        (profile for _, profile in pool.values()),
        snap, now=now, lookback=settings.lookback,
    )
    ranks: dict[RequirementId, dict[MemberId, int]] = {
        requirement.id: costs_for(eligible[requirement.id], keys)
        for requirement in plan.requirements
    }

    # --- 3. team assembly (§7.5)
    #
    # Pinned slots and their members leave the problem entirely: re-asking
    # someone who already agreed discards their consent, and the solver being
    # global means an unpinned re-run could legitimately move them to a
    # *different* slot — someone who agreed to fly as Navigator finding
    # themselves proposed as Pilot (§6.7).
    free = [slot for slot in slots if slot.key not in pinned]
    already_placed = set(pinned.values())
    candidates = sorted(
        {
            member_id
            for slot in free
            for member_id in ranks[slot.requirement.id]
            if member_id not in already_placed
        }
    )

    # Each requirement's eligible candidates, as index/cost arrays over
    # `candidates`, built once. Several slots share a requirement, so a slot's
    # edges are a copy of its requirement's rather than a fresh lookup per
    # candidate — and an ineligible pair is never visited at all.
    candidate_index = {member_id: i for i, member_id in enumerate(candidates)}
    requirement_edges: dict[RequirementId, tuple[np.ndarray, np.ndarray]] = {}
    for requirement in plan.requirements:
        pairs = sorted(
            (
                (rank, candidate_index[member_id])
                for member_id, rank in ranks[requirement.id].items()
                if member_id in candidate_index
            )
        )
        requirement_edges[requirement.id] = (
            np.array([c for _, c in pairs], dtype=np.int64),
            np.array([r for r, _ in pairs], dtype=np.int64),
        )

    def solve(forced: Mapping[SlotKey, MemberId]) -> dict[SlotKey, MemberId]:
        rows = [slot for slot in free if slot.key not in forced]
        if not rows or not candidates:
            return dict(forced)

        # Excluded candidates simply get no edges. The solver never matches a
        # column it cannot reach, so there is nothing to renumber.
        taken = already_placed | set(forced.values())
        open_column = np.ones(len(candidates), dtype=bool)
        for member_id in taken:
            index = candidate_index.get(member_id)
            if index is not None:
                open_column[index] = False

        # With R slots to fill, a slot never needs more than its R cheapest
        # candidates: if an optimal assignment gave it one outside that set, at
        # most R-1 of its R cheapest are taken by other slots, so one is free
        # and no more expensive — swapping there cannot make the team worse.
        # The edges are already in rank order, so this is a slice rather than a
        # sort, and it is what keeps the graph from growing with the roster.
        budget = len(rows)

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        cost_parts: list[np.ndarray] = []
        for row_position, slot in enumerate(rows):
            columns, costs = requirement_edges[slot.requirement.id]
            keep = open_column[columns]
            columns, costs = columns[keep][:budget], costs[keep][:budget]
            row_parts.append(np.full(columns.size, row_position, dtype=np.int64))
            col_parts.append(columns)
            cost_parts.append(costs)

        result = solver.assign_pairs(
            np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64),
            np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int64),
            np.concatenate(cost_parts) if cost_parts else np.empty(0, dtype=np.int64),
            row_count=len(rows),
            column_count=len(candidates),
        )

        out = dict(forced)
        for row_position, col_position in enumerate(result):
            if col_position is not None:
                out[rows[row_position].key] = candidates[col_position]
        return out

    # --- 4. team constraints, by constraint generation (§7.7)
    forced: dict[SlotKey, MemberId] = {}
    team = solve(forced)
    warnings: tuple[str, ...] = ()

    for _ in range(max_repair_rounds):
        full = {**pinned, **team}
        outstanding = [
            violation
            for constraint in team_constraints
            if (violation := constraint.violation(full, snap)) is not None
        ]
        if not outstanding:
            break
        pin = _repair_pin(team_constraints, full, forced, free, ranks, candidates, snap)
        if pin is None:
            # Return the best infeasible team with the violated constraints
            # named, rather than an empty result (§7.7).
            warnings = tuple(outstanding)
            break
        forced[pin[0]] = pin[1]
        team = solve(forced)
    else:
        warnings = tuple(
            violation
            for constraint in team_constraints
            if (violation := constraint.violation({**pinned, **team}, snap)) is not None
        )

    # --- 5. explanation (§7.9), built in rather than bolted on
    assigned = {**pinned, **team}
    return _explain(
        mission, snap, settings, slots, assigned, pinned, eligible, ranks, failures, pool,
        warnings, now=now,
    )


def _repair_pin(
    constraints: Sequence[TeamConstraint],
    team: Mapping[SlotKey, MemberId],
    forced: Mapping[SlotKey, MemberId],
    free: Sequence[Slot],
    ranks: Mapping[RequirementId, Mapping[MemberId, int]],
    candidates: Sequence[MemberId],
    snap: MatchSnapshot,
) -> tuple[SlotKey, MemberId] | None:
    """Pick one ``(slot, member)`` to force, to repair a violated constraint.

    A heuristic, and honest about it: this is constraint generation, not a
    guarantee of optimality under team constraints. **The principled solution is
    a CP-SAT model** (``ortools``), which handles team constraints and pairwise
    cohesion natively; not reached for because it is a heavy dependency and
    makes explanation much harder, and explanation is a hard requirement here.
    If a tenant needs *pairwise* terms ("these two work well together"), the
    repair loop is the wrong tool and CP-SAT is the upgrade path (§7.7).
    """
    on_team = set(team.values())
    for constraint in constraints:
        if constraint.violation(team, snap) is None:
            continue
        for member_id in sorted(constraint.satisfiers(snap)):
            if member_id in on_team or member_id not in candidates:
                continue
            for slot in free:
                if slot.key in forced:
                    continue
                if ranks[slot.requirement.id].get(member_id) is None:
                    continue
                return (slot.key, member_id)
    return None


def _explain(
    mission: Mission,
    snap: MatchSnapshot,
    settings: TenantSettings,
    slots: Sequence[Slot],
    assigned: Mapping[SlotKey, MemberId],
    pinned: Mapping[SlotKey, MemberId],
    eligible: Mapping[RequirementId, Sequence[CrewProfile]],
    ranks: Mapping[RequirementId, Mapping[MemberId, int]],
    failures: Mapping[tuple[RequirementId, MemberId], tuple[str, ...]],
    pool: Mapping[MemberId, tuple[Member, CrewProfile]],
    warnings: tuple[str, ...],
    *,
    now: datetime,
) -> MatchProposal:
    proposals: list[SlotProposal] = []
    unfilled: list[UnfilledSlot] = []
    placed = set(assigned.values())

    # Each requirement's candidates in rank order, computed **once per
    # requirement** rather than once per slot. Alternates are per-slot in the
    # output but the ordering behind them is not, and re-sorting the eligible
    # list for every slot of a `count: 20` requirement was twenty identical
    # sorts. Also cached per member: `load` walks the assignment index, so
    # recomputing it inside the slot loop repeated that walk too.
    ranked_spares: dict[RequirementId, tuple[MemberId, ...]] = {
        requirement_id: tuple(
            profile.member_id
            for profile in sorted(
                candidates, key=lambda p: ranks[requirement_id][p.member_id]
            )
        )
        for requirement_id, candidates in eligible.items()
    }
    recent_load: dict[MemberId, int] = {}

    for slot in slots:
        requirement = slot.requirement
        member_id = assigned.get(slot.key)
        if member_id is None:
            unfilled.append(
                UnfilledSlot(
                    requirement.id, slot.index, slot.label, requirement.mandatory,
                    _why_unfilled(slot, eligible[requirement.id], failures, placed, pool),
                )
            )
            continue

        profile = snap.crew[member_id]
        alternates = tuple(
            candidate
            for candidate in ranked_spares[requirement.id]
            if candidate not in placed
        )[: settings.alternates_depth]

        proposals.append(
            SlotProposal(
                requirement_id=requirement.id,
                slot_index=slot.index,
                label=slot.label,
                member_id=member_id,
                member_name=snap.members[member_id].name,
                rank=ranks[requirement.id][member_id],
                recent_load=recent_load.setdefault(
                    member_id, load(member_id, snap, now=now, lookback=settings.lookback)
                ),
                skills=tuple(
                    SkillEvidence(want.skill_id, want.min_level, profile.level(want.skill_id))  # type: ignore[arg-type]
                    for want in requirement.skills
                ),
                pinned=slot.key in pinned,
                alternates=alternates,
            )
        )

    near_misses = _near_misses(
        mission, snap, failures, placed, pool, depth=settings.near_miss_depth
    )

    return MatchProposal(
        mission_id=mission.id,
        generated_at=now,
        slots=tuple(proposals),
        unfilled=tuple(unfilled),
        near_misses=near_misses,
        team_warnings=warnings,
    )


NEAR_MISS_PRIORITY = {
    # How actionable the exclusion is. Someone who would qualify but is booked
    # is a scheduling decision a Lead can take; someone three levels short of
    # the bar is not.
    FILTER_AVAILABILITY: 0,
    FILTER_DECLINED: 1,
    FILTER_SKILLS: 2,
    FILTER_ACTIVE_MEMBER: 3,
}


def _near_misses(
    mission: Mission,
    snap: MatchSnapshot,
    failures: Mapping[tuple[RequirementId, MemberId], tuple[str, ...]],
    placed: set[MemberId],
    pool: Mapping[MemberId, tuple[Member, CrewProfile]],
    *,
    depth: int,
) -> tuple[NearMiss, ...]:
    """Crew excluded by *exactly one* filter, naming it (§7.9).

    Exclusions are recorded with their reason rather than discarded — that
    record is the difference between "no candidates" and "four people qualify
    but are already committed to Kepler that fortnight", which turns a matching
    result into a scheduling decision and makes an over-constrained requirement
    self-diagnosing (§7.3).

    **Capped per requirement.** §7.9 does not say to cap it, and at twenty crew
    it does not matter — but "excluded by exactly one filter" describes most of
    the roster once there are thousands of them, and a report with a hundred
    thousand rows is not a diagnostic. So each requirement keeps its most
    informative few, and the formatted detail — which is the expensive part — is
    computed only for those.

    Within a skills exclusion the *closest to the bar* come first, so "Fen is
    one level short" surfaces ahead of someone who holds none of the skills.
    That ordering is what makes the cap safe: the rows it drops are the ones
    nobody would have read.
    """
    ranked: dict[RequirementId, list[tuple[tuple[int, int, str], MemberId, str]]] = {}
    # MissionPlan.requirement() is a linear scan, and this loop runs once per
    # (requirement, candidate) pair — so at a hundred requirements the scan was
    # the dominant cost of building the report.
    by_id = {r.id: r for r in mission.plan.requirements}

    for (requirement_id, member_id), failed in failures.items():
        if len(failed) != 1 or member_id in placed:
            continue
        reason = failed[0]
        shortfall = 0
        if reason == FILTER_SKILLS:
            requirement = by_id[requirement_id]
            shortfall = sum(
                required - (held or 0)
                for _, required, held in skill_shortfall(requirement, pool[member_id][1])
            )
        sort_key = (NEAR_MISS_PRIORITY.get(reason, 9), shortfall, member_id)
        ranked.setdefault(requirement_id, []).append((sort_key, member_id, reason))

    return tuple(
        NearMiss(
            member_id,
            pool[member_id][0].name,
            requirement_id,
            reason,
            _filter_detail(reason, mission, snap, requirement_id, member_id, pool),
        )
        for requirement_id in sorted(ranked)
        for _, member_id, reason in sorted(ranked[requirement_id])[:depth]
    )


def _why_unfilled(
    slot: Slot,
    eligible: Sequence[CrewProfile],
    failures: Mapping[tuple[RequirementId, MemberId], tuple[str, ...]],
    placed: set[MemberId],
    pool: Mapping[MemberId, tuple[Member, CrewProfile]],
) -> str:
    """A partial team is actionable; an exception is not (§7.5). So say which
    kind of empty this is."""
    if not eligible:
        breakdown: dict[str, int] = {}
        for (requirement_id, _), failed in failures.items():
            if requirement_id != slot.requirement.id:
                continue
            for name in failed:
                breakdown[name] = breakdown.get(name, 0) + 1
        detail = ", ".join(f"{count} by {name}" for name, count in sorted(breakdown.items()))
        return (
            f"no eligible candidates out of {len(pool)} crew"
            + (f" — excluded {detail}" if detail else "")
        )
    spare = [p for p in eligible if p.member_id not in placed]
    if not spare:
        return (
            f"all {len(eligible)} eligible candidates are assigned to other slots "
            f"on this mission"
        )
    return f"{len(spare)} eligible candidates remain but the solver could not place one"


def _filter_detail(
    name: str,
    mission: Mission,
    snap: MatchSnapshot,
    requirement_id: RequirementId,
    member_id: MemberId,
    pool: Mapping[MemberId, tuple[Member, CrewProfile]],
) -> str:
    profile = pool[member_id][1]
    if name == FILTER_SKILLS:
        requirement = mission.plan.requirement(requirement_id)
        return "; ".join(
            f"{skill_id} needs {required.name}, holds {held.name if held else 'nothing'}"
            for skill_id, required, held in skill_shortfall(requirement, profile)
        )
    if name == "active_member":
        return "not an active crew member"
    if name == FILTER_DECLINED:
        return "already declined this mission"
    clashes = conflicting_commitments(profile, mission.plan.window, snap)
    if clashes:
        return "committed to " + ", ".join(
            f"{snap.missions[c.mission_id].title!r} ({snap.missions[c.mission_id].plan.window})"
            for c in clashes
        )
    return "declared unavailable for this window"
