"""Availability (§4.3, §10).

``unavailable = committed_mission_windows  u  declared_unavailability``

The test that matters most here is the double-booking one: reading "committed"
as "ACTIVE missions only" is a bug that two APPROVED missions expose and one
never will.
"""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from mission_control.domain import (
    AssignmentState,
    GuardFailed,
    MissionState,
    Proficiency,
    TimeWindow,
    UnavailabilityBlock,
    blocking_windows,
    is_available,
)

from .conftest import CREW, PILOT, T0, req

Pr = Proficiency


def window(start_day: int, end_day: int) -> TimeWindow:
    return TimeWindow(T0 + timedelta(days=start_day), T0 + timedelta(days=end_day))


# -------------------------------------------------------------- interval overlap


def test_a_window_must_end_after_it_starts():
    with pytest.raises(ValueError):
        TimeWindow(T0, T0)
    with pytest.raises(ValueError):
        TimeWindow(T0 + timedelta(days=1), T0)


def test_touching_windows_do_not_overlap():
    """Half-open ``[start, end)``: a mission ending 09:00 does not conflict with
    one starting 09:00 (§4.3)."""
    first = window(0, 10)
    second = window(10, 20)
    assert not first.overlaps(second)
    assert not second.overlaps(first)


def test_overlap_is_commutative_and_reflexive():
    random.seed(11)
    for _ in range(2000):
        a_start, a_len = random.randint(0, 40), random.randint(1, 20)
        b_start, b_len = random.randint(0, 40), random.randint(1, 20)
        a = window(a_start, a_start + a_len)
        b = window(b_start, b_start + b_len)

        assert a.overlaps(b) == b.overlaps(a), (a, b)
        assert a.overlaps(a)


def test_overlap_agrees_with_a_naive_day_by_day_check():
    random.seed(12)
    for _ in range(500):
        a_start, a_len = random.randint(0, 20), random.randint(1, 10)
        b_start, b_len = random.randint(0, 20), random.randint(1, 10)
        a = window(a_start, a_start + a_len)
        b = window(b_start, b_start + b_len)

        shared = set(range(a_start, a_start + a_len)) & set(range(b_start, b_start + b_len))
        assert a.overlaps(b) is bool(shared), (a, b)


def test_a_time_window_exposes_only_overlap():
    """Because availability is boolean rather than fractional, there is no
    interval subtraction, no duration and no ratio to keep in range (§4.3)."""
    public = {
        name for name in dir(TimeWindow)
        if not name.startswith("_") and callable(getattr(TimeWindow, name))
    }
    assert public == {"overlaps"}


# ------------------------------------------------------- the two sources, alone


def test_a_declared_block_alone_makes_someone_unavailable(world):
    anya = world.crew(1)
    anya.crew.set_unavailability(CREW[1], [UnavailabilityBlock(window(28, 40), "leave")])

    snap = world.lead().workspace.snapshot()
    profile = snap.crew[CREW[1]]
    assert not is_available(profile, window(30, 45), snap)
    assert is_available(profile, window(50, 60), snap)


def test_an_accepted_assignment_alone_makes_someone_unavailable(world):
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30)
    world.crew_up(mission.id)
    booked = world.accepted(mission.id)[0].member_id

    snap = world.lead().workspace.snapshot()
    profile = snap.crew[booked]
    assert not profile.unavailability, "no declared block; the commitment is the only source"
    assert not is_available(profile, window(30, 45), snap)
    assert is_available(profile, window(60, 70), snap)


def test_an_offer_takes_no_hold(world):
    """OFFERED is asked, not booked — so two Leads may court the same person and
    the first acceptance wins (§6.7)."""
    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    lead = world.lead()
    proposal = lead.matching.run_matcher(mission.id)
    offers = lead.matching.offer_proposal(mission.id, proposal)

    snap = lead.workspace.snapshot()
    profile = snap.crew[offers[0].member_id]
    assert is_available(profile, mission.plan.window, snap)


@pytest.mark.parametrize(
    "state,blocks",
    [
        (MissionState.DRAFT, True),
        (MissionState.PENDING_APPROVAL, True),
        (MissionState.APPROVED, True),
        (MissionState.ACTIVE, True),
        (MissionState.COMPLETED, False),
        (MissionState.CANCELLED, False),
        (MissionState.ABORTED, False),
    ],
)
def test_it_is_the_pair_of_states_that_decides(world, state, blocks):
    """§4.3's table, both halves.

    An ACCEPTED assignment on any non-terminal mission blocks; once the mission
    is terminal the assignment is history and the hold is released. Asserting
    the negative half is the point — that is where a double-booking bug lives.
    """
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT})
    if state in {MissionState.DRAFT, MissionState.CANCELLED}:
        # mission_in() reaches both of these without crewing, so crew by hand.
        # An acceptance during DRAFT must block, or two drafts could book the
        # same person (§6.6); cancelling must then free the dates (§8.1).
        mission = world.draft(pilot)
        world.crew_up(mission.id)
        if state is MissionState.CANCELLED:
            mission = world.lead().missions.cancel(mission.id, "scrubbed")
    else:
        mission = world.mission_in(state, pilot)

    snap = world.lead().workspace.snapshot()
    assignments = snap.assignments_for_mission(mission.id)
    assert assignments, "every case must have produced an assignment"
    assignment = assignments[0]
    profile = snap.crew[assignment.member_id]
    windows = list(blocking_windows(profile, snap))

    assert (mission.plan.window in windows) is blocks
    assert is_available(profile, mission.plan.window, snap) is (not blocks)

    if blocks:
        assert assignment.state is AssignmentState.ACCEPTED
    else:
        # The hold is released by a real state change on the assignment, not by
        # the mission state being consulted at read time (§8.1).
        assert assignment.state in {
            AssignmentState.COMPLETED,
            AssignmentState.PARTIAL,
            AssignmentState.RELEASED,
        }


def test_two_approved_missions_cannot_book_the_same_person(world):
    """Catches treating only ACTIVE as committed (§10).

    Two APPROVED missions with overlapping windows would each see the member as
    free, and the same person would be promised to both.
    """
    first = world.mission_in(
        MissionState.APPROVED, req("Pilot", {PILOT: Pr.MASTER}), start_days=30
    )
    booked = world.accepted(first.id)[0].member_id
    assert booked == CREW[1], "Anya is the only Master pilot, so the clash is forced"

    second = world.draft(req("Pilot", {PILOT: Pr.MASTER}), title="Europa", start_days=35)
    proposal = world.lead().matching.run_matcher(second.id)

    assert [s.member_id for s in proposal.slots] == []
    assert len(proposal.unfilled) == 1
    assert [n.member_id for n in proposal.near_misses if n.filter == "availability"] == [booked]


def test_a_one_hour_conflict_in_a_three_week_window_excludes(world):
    """No partial-credit path exists to get wrong (§10). A tolerance would hide
    an uncovered window rather than express one."""
    anya = world.crew(1)
    sliver = UnavailabilityBlock(
        TimeWindow(T0 + timedelta(days=35), T0 + timedelta(days=35, hours=1)), "medical"
    )
    anya.crew.set_unavailability(CREW[1], [sliver])

    snap = world.lead().workspace.snapshot()
    assert not is_available(snap.crew[CREW[1]], window(30, 51), snap)


# ----------------------------------------------- availability versus acceptance


def test_declaring_unavailability_over_an_accepted_assignment_is_refused(world):
    """Rejected outright (§3.3, §8.2).

    Acceptance is final here, so letting the declaration win would silently
    strand a mission whose Director has already approved that crew.
    """
    mission = world.mission_in(
        MissionState.APPROVED, req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30
    )
    booked = world.accepted(mission.id)[0].member_id

    with pytest.raises(GuardFailed) as caught:
        world.session(booked).crew.set_unavailability(
            booked, [UnavailabilityBlock(window(32, 36), "holiday")]
        )
    assert "Artemis VII" in str(caught.value)
    assert not world.lead().workspace.crew_profile(booked).unavailability


def test_declaring_unavailability_elsewhere_is_allowed(world):
    mission = world.mission_in(
        MissionState.APPROVED, req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30
    )
    booked = world.accepted(mission.id)[0].member_id

    profile = world.session(booked).crew.set_unavailability(
        booked, [UnavailabilityBlock(window(60, 70), "holiday")]
    )
    assert len(profile.unavailability) == 1


def test_a_cancelled_mission_frees_the_dates_immediately(world):
    mission = world.mission_in(
        MissionState.APPROVED, req("Pilot", {PILOT: Pr.PROFICIENT}), start_days=30
    )
    booked = world.accepted(mission.id)[0].member_id
    world.director().missions.cancel(mission.id, "programme descoped")

    snap = world.lead().workspace.snapshot()
    assert is_available(snap.crew[booked], mission.plan.window, snap)

    # And the declaration that was refused a moment ago now succeeds.
    world.session(booked).crew.set_unavailability(
        booked, [UnavailabilityBlock(window(32, 36), "holiday")]
    )


def test_a_members_own_assignment_does_not_block_their_own_mission(world):
    """``ignoring`` exists for exactly this: re-validating an acceptance must not
    treat that acceptance as a conflict with itself (§6.8)."""
    mission = world.mission_in(MissionState.PENDING_APPROVAL, req("Pilot", {PILOT: Pr.PROFICIENT}))
    report = world.director().missions.allocation_report(mission.id)
    assert report.stale == ()
    assert len(report.valid) == 1
