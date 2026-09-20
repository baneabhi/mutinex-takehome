"""The matching engine against real crew (§7, §10).

The fixture roster, from conftest:

======  ==========================================
Anya    Piloting MASTER, Medicine EXPERT
Boris   Piloting EXPERT, Medicine NOVICE
Chen    Piloting NOVICE, Medicine NOVICE
Dara    Medicine EXPERT, Physiology PROFICIENT
Eli     Medicine PROFICIENT, Engineering COMPETENT
Fen     Piloting PROFICIENT, Engineering EXPERT
======  ==========================================
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from mission_control.domain import (
    AssignmentState,
    CrewSkill,
    MissionState,
    Proficiency,
    UnavailabilityBlock,
)
from mission_control.matching import AtLeastAtLevel, match

from .conftest import CREW, ENG, MED, PHYS, PILOT, T0, req

Pr = Proficiency


def proposal_for(world, mission_id):
    return world.lead().matching.run_matcher(mission_id)


def placed(proposal):
    return {slot.slot: slot.member_id for slot in proposal.slots}


# --------------------------------------------------------- global beats greedy


def test_global_assignment_fills_both_slots_where_greedy_would_strand_one(world):
    """§7.6, end to end through the real engine.

    Restricted to Anya, Boris and Chen so the matrix is exactly the doc's.
    Filling Pilot greedily takes Anya — the only eligible Flight Surgeon — and
    the mission cannot be activated. Greedy did not merely rank worse; it
    produced a mission that could not fly.
    """
    director = world.director()
    for index in (4, 5, 6):
        director.org.deactivate_member(CREW[index], "not in this scenario")

    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Flight Surgeon", {MED: Pr.PROFICIENT}),
    )
    assignment = placed(proposal_for(world, mission.id))

    assert assignment == {("r-pilot", 0): CREW[2], ("r-flight-surgeon", 0): CREW[1]}
    assert proposal_for(world, mission.id).unfilled == ()


def test_scarcity_is_a_property_of_the_whole_matrix(world):
    """The symmetric counterexample: no row ordering heuristic saves greedy."""
    director = world.director()
    for index in (2, 3, 5, 6):
        director.org.deactivate_member(CREW[index], "not in this scenario")

    # Anya: pilot + medicine. Dara: medicine only. So Dara must take the
    # Surgeon slot and Anya the Pilot slot — the mirror of the case above.
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Flight Surgeon", {MED: Pr.PROFICIENT}),
    )
    assignment = placed(proposal_for(world, mission.id))
    assert assignment == {("r-pilot", 0): CREW[1], ("r-flight-surgeon", 0): CREW[4]}


# ------------------------------------------------------------------ eligibility


def test_a_multi_skill_requirement_needs_one_person_holding_all_of_them(world):
    """A Flight Surgeon needing ``Medicine >= Expert`` **and**
    ``Physiology >= Proficient`` is one person with both (§7.1).

    Only Dara holds both. Anya has Medicine EXPERT but no Physiology.
    """
    mission = world.draft(
        req("Flight Surgeon", {MED: Pr.EXPERT, PHYS: Pr.PROFICIENT}, rid="r-surgeon")
    )
    assignment = placed(proposal_for(world, mission.id))
    assert assignment == {("r-surgeon", 0): CREW[4]}


def test_differing_levels_are_differing_requirements(world):
    """Three at Expert and two at Competent is *two* requirements (§7.1), and
    the Experts must land in the Expert slots."""
    mission = world.draft(
        req("Senior Medic", {MED: Pr.EXPERT}, count=2, rid="r-senior"),
        req("Medic", {MED: Pr.PROFICIENT}, count=1, rid="r-junior"),
    )
    assignment = placed(proposal_for(world, mission.id))

    seniors = {m for (r, _), m in assignment.items() if r == "r-senior"}
    juniors = {m for (r, _), m in assignment.items() if r == "r-junior"}

    assert seniors == {CREW[1], CREW[4]}, "Anya and Dara are the only Experts"
    assert juniors == {CREW[5]}, "Eli is Proficient, so he takes the junior slot"


def test_count_produces_that_many_slots(world):
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}, count=3))
    proposal = proposal_for(world, mission.id)

    assert len(proposal.slots) == 3
    assert {s.slot_index for s in proposal.slots} == {0, 1, 2}
    assert len({s.member_id for s in proposal.slots}) == 3, "one person, one slot"


def test_nobody_holds_two_slots_on_one_mission(world):
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Engineer", {ENG: Pr.COMPETENT}),
    )
    proposal = proposal_for(world, mission.id)
    members = [s.member_id for s in proposal.slots]
    assert len(members) == len(set(members))


def test_an_impossible_requirement_is_reported_not_raised(world):
    """A partial team is actionable; an exception is not (§7.5)."""
    mission = world.draft(req("Xenolinguist", {PHYS: Pr.MASTER}, rid="r-xeno"))
    proposal = proposal_for(world, mission.id)

    assert proposal.slots == ()
    assert len(proposal.unfilled) == 1
    assert "no eligible candidates" in proposal.unfilled[0].reason
    assert not proposal.is_complete


def test_more_slots_than_qualified_people_fills_what_it_can(world):
    mission = world.draft(req("Master Pilot", {PILOT: Pr.MASTER}, count=3, rid="r-mp"))
    proposal = proposal_for(world, mission.id)

    assert [s.member_id for s in proposal.slots] == [CREW[1]]
    assert len(proposal.unfilled) == 2
    assert "assigned to other slots" in proposal.unfilled[0].reason


# --------------------------------------------------------------------- ranking


def test_load_outranks_qualification(world):
    """Anya is MASTER, Fen is PROFICIENT, and the bar is PROFICIENT. Giving Anya
    a commitment makes Fen the pick — under any "prefer the best" model Anya
    would win regardless (§7.4)."""
    world.director().org.deactivate_member(CREW[2], "not in this scenario")

    other = world.draft(req("Master Pilot", {PILOT: Pr.MASTER}, rid="r-mp"), title="Prior")
    world.crew_up(other.id)
    assert world.accepted(other.id)[0].member_id == CREW[1]

    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}), title="Artemis", start_days=90)
    proposal = proposal_for(world, mission.id)

    assert proposal.slots[0].member_id == CREW[6], "Fen, the less qualified but idle one"
    assert proposal.slots[0].rank == 0


def test_surplus_qualification_is_not_consulted_even_at_equal_load(world):
    """The sharper version: with load tied, the *lower*-qualified candidate wins
    if they sort first.

    Ranking by surplus above the bar treats "more than requested" as "better",
    which is a soft preference needing a weight nobody can source (§7.4). So
    the level is not consulted at all — not even as a tiebreak. Skills are
    swapped here so the alphabetically-first member is the weaker one, which is
    what makes the assertion mean something.
    """
    director = world.director()
    for index in (3, 4, 5, 6):
        director.org.deactivate_member(CREW[index], "not in this scenario")
    director.crew.set_skills(CREW[1], [CrewSkill(PILOT, Pr.PROFICIENT)])
    director.crew.set_skills(CREW[2], [CrewSkill(PILOT, Pr.MASTER)])

    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    slot = proposal_for(world, mission.id).slots[0]

    assert slot.member_id == CREW[1], "the barely-qualified one, because ids tie-break"
    assert slot.skills[0].held is Pr.PROFICIENT


def test_load_counts_commitments_not_only_completed_missions(world):
    """Counting only COMPLETED means someone who accepted five missions for next
    month still looks idle and gets offered a sixth (§7.8)."""
    lead = world.lead()
    world.director().org.deactivate_member(CREW[2], "not in this scenario")

    for index in range(4):
        prior = world.draft(
            req("Master Pilot", {PILOT: Pr.MASTER}, rid="r-mp"),
            title=f"Prior {index}",
            start_days=200 + index * 20,
        )
        world.crew_up(prior.id)

    snap = lead.workspace.snapshot()
    from mission_control.matching import load

    assert load(CREW[1], snap, now=T0, lookback=timedelta(days=90)) == 4, (
        "four future commitments, nothing completed — she is not idle"
    )
    assert load(CREW[6], snap, now=T0, lookback=timedelta(days=90)) == 0

    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30)
    assert proposal_for(world, mission.id).slots[0].member_id == CREW[6]


def test_being_asked_is_not_being_given_work(world):
    """DECLINED, RELEASED and OFFERED do not count toward load (§7.8)."""
    from mission_control.matching import load

    mission = world.draft(req("Master Pilot", {PILOT: Pr.MASTER}, rid="r-mp"))
    lead = world.lead()
    proposal = lead.matching.run_matcher(mission.id)
    offers = lead.matching.offer_proposal(mission.id, proposal)

    snap = lead.workspace.snapshot()
    assert load(CREW[1], snap, now=T0, lookback=timedelta(days=90)) == 0

    world.crew(1).assignments.decline(offers[0].id)
    snap = lead.workspace.snapshot()
    assert load(CREW[1], snap, now=T0, lookback=timedelta(days=90)) == 0


def test_equal_load_breaks_on_who_flew_longest_ago(world):
    """Under id-only tie-breaking this passes half the time by luck (§10).

    Boris and Fen are both eligible Pilots, each with one completed mission
    inside the lookback, so load ties at 1. Boris flew three weeks ago and Fen
    eleven, so Fen should be picked — which is the *opposite* of the
    alphabetical order of their ids, and therefore a real discrimination
    between the two rules.
    """
    world.director().org.deactivate_member(CREW[1], "not in this scenario")
    world.seed_history(CREW[2], ended_days_ago=21, title="Boris recent")
    world.seed_history(CREW[6], ended_days_ago=77, title="Fen, a while back")

    from mission_control.matching import last_mission_end, load

    snap = world.lead().workspace.snapshot()
    assert load(CREW[2], snap, now=T0, lookback=timedelta(days=90)) == 1
    assert load(CREW[6], snap, now=T0, lookback=timedelta(days=90)) == 1
    assert last_mission_end(CREW[6], snap) < last_mission_end(CREW[2], snap)

    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30)
    assert proposal_for(world, mission.id).slots[0].member_id == CREW[6]


def test_someone_who_never_flew_sorts_ahead_of_someone_who_has(world):
    """``datetime.min`` for "never flown" sorts new crew first (§7.4)."""
    world.director().org.deactivate_member(CREW[1], "not in this scenario")
    world.seed_history(CREW[2], ended_days_ago=200, title="Long ago")

    from mission_control.matching import last_mission_end, load

    snap = world.lead().workspace.snapshot()
    assert load(CREW[2], snap, now=T0, lookback=timedelta(days=90)) == 0, "outside the lookback"
    assert last_mission_end(CREW[6], snap) is None

    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30)
    assert proposal_for(world, mission.id).slots[0].member_id == CREW[6]


def test_the_lookback_window_forgets_old_work(world):
    """The history term is bounded; the commitment term is not, because an
    outstanding commitment is current by definition (§7.8)."""
    from mission_control.matching import load

    world.seed_history(CREW[1], ended_days_ago=30, title="Inside")
    world.seed_history(CREW[2], ended_days_ago=200, title="Outside")

    snap = world.lead().workspace.snapshot()
    assert load(CREW[1], snap, now=T0, lookback=timedelta(days=90)) == 1
    assert load(CREW[2], snap, now=T0, lookback=timedelta(days=90)) == 0


def test_load_anchors_on_now_not_on_the_mission_window(world):
    """A window anchored on a mission six months out would look back to dates
    three months from *now*, miss the missions someone flew this year, and rank
    them as idle (§7.8)."""
    from mission_control.matching import load

    world.seed_history(CREW[1], ended_days_ago=30, title="Recent")
    snap = world.lead().workspace.snapshot()

    # Counted, even though the mission being crewed is 180 days out.
    assert load(CREW[1], snap, now=T0, lookback=timedelta(days=90)) == 1

    # Anchored on the far-future mission instead, the lookback would start 90
    # days *after* now and miss it entirely — which is the bug this pins.
    far_future = T0 + timedelta(days=180)
    assert load(CREW[1], snap, now=far_future, lookback=timedelta(days=90)) == 0


def test_ranking_is_a_dense_integer_sequence(world):
    """Rank-as-cost: the solver's cost *is* the position in the sort (§7.5)."""
    mission = world.draft(req("Medic", {MED: Pr.PROFICIENT}, count=3, rid="r-med"))
    proposal = proposal_for(world, mission.id)
    assert sorted(s.rank for s in proposal.slots) == [0, 1, 2]


# ---------------------------------------------------------------- explanation


def test_each_slot_carries_the_facts_behind_it_and_no_score(world):
    """There is no score to show because §7.4 computes none. The levels are
    exactly what a Lead needs to apply the judgement the engine does not (§7.9)."""
    mission = world.draft(req("Flight Surgeon", {MED: Pr.EXPERT, PHYS: Pr.PROFICIENT}, rid="r-fs"))
    slot = proposal_for(world, mission.id).slots[0]

    assert slot.member_name == "Dara"
    assert {(e.skill_id, e.required, e.held) for e in slot.skills} == {
        (MED, Pr.EXPERT, Pr.EXPERT),
        (PHYS, Pr.PROFICIENT, Pr.PROFICIENT),
    }
    assert slot.recent_load == 0
    assert not hasattr(slot, "score")


def test_near_misses_name_the_single_filter_that_excluded_someone(world):
    """"Three people qualify but are already committed to Kepler that week"
    converts a matching result into a scheduling decision (§7.9)."""
    prior = world.draft(
        req("Master Pilot", {PILOT: Pr.MASTER}, rid="r-mp"), title="Kepler", start_days=30
    )
    world.crew_up(prior.id)

    mission = world.draft(req("Pilot", {PILOT: Pr.EXPERT}), title="Artemis", start_days=32)
    proposal = proposal_for(world, mission.id)

    by_member = {n.member_id: n for n in proposal.near_misses}
    assert by_member[CREW[1]].filter == "availability"
    assert "Kepler" in by_member[CREW[1]].detail

    assert by_member[CREW[3]].filter == "skills"
    assert "needs EXPERT, holds NOVICE" in by_member[CREW[3]].detail


def test_someone_excluded_by_two_filters_is_not_a_near_miss(world):
    """You cannot count to one if you stopped at the first failure (§7.9) — and
    two failures is not a near miss."""
    world.crew(3).crew.set_unavailability(
        CREW[3], [UnavailabilityBlock(world.draft().plan.window, "leave")]
    )
    mission = world.draft(req("Pilot", {PILOT: Pr.EXPERT}), title="Artemis")
    proposal = proposal_for(world, mission.id)

    assert CREW[3] not in {n.member_id for n in proposal.near_misses}


def test_alternates_are_the_next_best_eligible_crew(world):
    """Advisory only, never part of the plan — they let a Lead re-offer
    immediately when someone declines during DRAFT (§7.9)."""
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    slot = proposal_for(world, mission.id).slots[0]

    assert slot.member_id == CREW[1]
    assert slot.alternates == (CREW[2], CREW[6])
    assert len(slot.alternates) <= world.lead().workspace.settings.alternates_depth


def test_an_unfilled_slot_says_which_kind_of_empty_it_is(world):
    contested = world.draft(req("Master Pilot", {PILOT: Pr.MASTER}, count=2, rid="r-mp"))
    reasons = [u.reason for u in proposal_for(world, contested.id).unfilled]
    assert "assigned to other slots" in reasons[0]

    impossible = world.draft(req("Xeno", {PHYS: Pr.MASTER}, rid="r-x"), title="Other")
    reasons = [u.reason for u in proposal_for(world, impossible.id).unfilled]
    assert "no eligible candidates" in reasons[0]
    assert "by skills" in reasons[0]


def test_an_optional_unfilled_slot_does_not_make_the_proposal_incomplete(world):
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Xeno", {PHYS: Pr.MASTER}, mandatory=False, rid="r-x"),
    )
    proposal = proposal_for(world, mission.id)
    assert len(proposal.unfilled) == 1
    assert proposal.is_complete


# ------------------------------------------------------------ team constraints


def test_a_team_constraint_is_repaired_by_forcing_a_qualifying_member(world):
    """Constraint generation (§7.7): solve, validate, re-solve with the
    constraint forced."""
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}, count=2))
    snap = world.lead().workspace.snapshot()
    settings = world.lead().workspace.settings

    unconstrained = match(
        world.lead().missions.get(mission.id), snap, settings, now=T0
    )
    assert {s.member_id for s in unconstrained.slots} == {CREW[1], CREW[2]}
    assert unconstrained.team_warnings == ()

    # Neither Anya nor Boris is an Engineering Expert; Fen is, and is also an
    # eligible Pilot. The repair loop should bring Fen in.
    constrained = match(
        world.lead().missions.get(mission.id), snap, settings, now=T0,
        team_constraints=[AtLeastAtLevel(ENG, Pr.EXPERT)],
    )
    assert CREW[6] in {s.member_id for s in constrained.slots}
    assert constrained.team_warnings == ()


def test_an_unsatisfiable_team_constraint_returns_the_best_team_with_a_warning(world):
    """Rather than an empty result (§7.7). A heuristic being honest about its
    limits."""
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    snap = world.lead().workspace.snapshot()

    proposal = match(
        world.lead().missions.get(mission.id), snap,
        world.lead().workspace.settings, now=T0,
        team_constraints=[AtLeastAtLevel(PHYS, Pr.MASTER)],
    )
    assert proposal.slots, "a best-effort team is still returned"
    assert len(proposal.team_warnings) == 1
    assert "PHYS" in proposal.team_warnings[0] or "phys" in proposal.team_warnings[0]


# --------------------------------------------------------------- determinism


def test_the_same_inputs_give_the_same_team(world):
    """Integer costs plus member_id as the final tiebreak, and ``now`` captured
    once per run (§7.5)."""
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}, count=2),
        req("Medic", {MED: Pr.PROFICIENT}, count=2, rid="r-med"),
    )
    snap = world.lead().workspace.snapshot()
    settings = world.lead().workspace.settings
    subject = world.lead().missions.get(mission.id)

    first = match(subject, snap, settings, now=T0)
    for _ in range(5):
        again = match(subject, snap, settings, now=T0)
        assert placed(again) == placed(first)
        assert [s.rank for s in again.slots] == [s.rank for s in first.slots]


# -------------------------------------------------------------------- pinning


def test_a_rematch_pins_accepted_crew_to_the_same_slot(world):
    """The slot-identity assertion is the point (§10).

    The solver is global, so re-running it with everyone free could legitimately
    move an already-accepted person to a *different* slot — someone who agreed
    to fly as Navigator finding themselves proposed as Pilot. Pinning stops
    consent quietly changing meaning (§6.7).
    """
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Medic", {MED: Pr.PROFICIENT}, rid="r-med"),
    )
    lead = world.lead()
    first = lead.matching.run_matcher(mission.id)
    offers = lead.matching.offer_proposal(mission.id, first)

    accepted, declined = offers[0], offers[1]
    accepted = world.session(accepted.member_id).assignments.accept(accepted.id)
    world.session(declined.member_id).assignments.decline(declined.id)

    second = lead.matching.run_matcher(mission.id)
    by_slot = {s.slot: s for s in second.slots}

    kept = by_slot[accepted.slot]
    assert kept.member_id == accepted.member_id
    assert kept.pinned is True

    refilled = by_slot[declined.slot]
    assert refilled.pinned is False
    assert refilled.member_id != accepted.member_id

    # Only the vacant slot is re-offered.
    new_offers = lead.matching.offer_proposal(mission.id, second)
    assert [o.slot for o in new_offers] == [declined.slot]
    # The pinned assignment was not rewritten: same state, same version.
    assert lead.workspace.assignment(accepted.id).state is AssignmentState.ACCEPTED
    assert lead.workspace.assignment(accepted.id).version == accepted.version
    assert lead.workspace.assignment(accepted.id).responded_at == accepted.responded_at


def test_a_pinned_member_is_not_offered_a_second_slot(world):
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}, count=2))
    lead = world.lead()
    offers = lead.matching.offer_proposal(mission.id, lead.matching.run_matcher(mission.id))
    world.session(offers[0].member_id).assignments.accept(offers[0].id)

    second = lead.matching.run_matcher(mission.id)
    members = [s.member_id for s in second.slots]
    assert len(members) == len(set(members))


def test_adding_worse_candidates_does_not_change_the_team(world):
    """Pins the candidate pruning in ``solve`` (§7.5).

    With R slots to fill, each slot's edges are cut to its R cheapest
    candidates — which is only sound if candidates beyond that could never be
    chosen. Growing the roster with people who all rank *worse* must therefore
    leave the chosen team identical, not merely equally good.
    """
    from mission_control.domain import CrewSkill, Role

    lead = world.lead()
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT}, count=2)

    first = world.draft(pilot, title="Before")
    baseline = [s.member_id for s in lead.matching.run_matcher(first.id).slots]
    assert len(baseline) == 2

    # Twenty more eligible pilots, every one of them already carrying load, so
    # each ranks below the whole original pool.
    director = world.director()
    for index in range(20):
        newcomer = director.org.add_member(f"Latecomer {index}", Role.CREW)
        director.crew.set_skills(newcomer.id, [CrewSkill(PILOT, Pr.MASTER)])
        busy = world.draft(
            req("Master Pilot", {PILOT: Pr.MASTER}, rid=f"r-busy-{index}"),
            title=f"Busywork {index}",
            start_days=400 + index * 30,
        )
        offer = lead.assignments.offer(busy.id, f"r-busy-{index}", 0, newcomer.id)
        world.session(newcomer.id).assignments.accept(offer.id)

    second = world.draft(pilot, title="After")
    after = [s.member_id for s in lead.matching.run_matcher(second.id).slots]

    assert after == baseline


def test_the_matcher_creates_nothing(world):
    """Auto-assignment would be a small convenience and a large mistake — this
    allocates work to people, and a human should own that (§7.9)."""
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    lead = world.lead()

    lead.matching.run_matcher(mission.id)
    lead.matching.run_matcher(mission.id)

    assert lead.workspace.assignments_for_mission(mission.id) == []
    assert lead.missions.get(mission.id).state is MissionState.DRAFT


def test_the_matcher_runs_only_in_draft(world):
    """Matching produces roster changes, roster changes are plan changes, plan
    changes need re-approval (§6.7)."""
    from mission_control.domain import IllegalTransition

    for state in (MissionState.PENDING_APPROVAL, MissionState.APPROVED, MissionState.ACTIVE):
        mission = world.mission_in(state, req("Pilot", {PILOT: Pr.PROFICIENT}))
        with pytest.raises(IllegalTransition):
            world.lead().matching.run_matcher(mission.id)


def test_running_the_matcher_is_audited(world):
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    world.lead().matching.run_matcher(mission.id)

    events = [e for e in world.director().org.audit() if e.event == "run_matcher"]
    assert len(events) == 1
    assert events[0].from_state == "draft" and events[0].to_state == "draft"
