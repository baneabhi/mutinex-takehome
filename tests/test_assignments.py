"""The assignment state machine — crew response (§8, §10).

The two machines couple in one direction only. Mission events drive assignment
events; crew responses never move the mission. That asymmetry is what several of
these tests exist to pin, because a well-meaning auto-advance is the obvious
thing for someone to add later.
"""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest

from mission_control.domain import (
    AssignmentState,
    CrewingStatus,
    Denied,
    GuardFailed,
    IllegalTransition,
    Location,
    MissionState,
    NotFound,
    Proficiency,
)
from mission_control.lifecycle import ASSIGNMENT_TRANSITIONS

from .conftest import CREW, MED, PILOT, T0, req

AS = AssignmentState
MS = MissionState
Pr = Proficiency

ALL_EVENTS = sorted({event for _, event in ASSIGNMENT_TRANSITIONS})


def offer_one(world, *, requirement=None, member=None, start_days=30):
    """Put a single offer on the table and return ``(mission, assignment)``."""
    requirement = requirement or req("Pilot", {PILOT: Pr.PROFICIENT})
    mission = world.draft(requirement, start_days=start_days)
    lead = world.lead()
    target = member or CREW[1]
    assignment = lead.assignments.offer(mission.id, requirement.id, 0, target)
    return mission, assignment


# ------------------------------------------------------------------- the table


def test_there_is_no_route_out_of_accepted_back_to_declined():
    """Acceptance is final (§8.2).

    **In a real system this is insufficient** — people get sick — and the honest
    model needs a crew-initiated release with a re-crew path (§11 trade-off 3).
    This asserts the scope decision rather than defending it.
    """
    assert (AS.ACCEPTED, "decline") not in ASSIGNMENT_TRANSITIONS
    targets = {
        event: t.target for (source, event), t in ASSIGNMENT_TRANSITIONS.items()
        if source is AS.ACCEPTED
    }
    assert targets == {"release": AS.RELEASED, "complete": AS.COMPLETED, "abort": AS.PARTIAL}


def test_terminal_assignment_states_have_no_outgoing_transitions():
    sources = {source for source, _ in ASSIGNMENT_TRANSITIONS}
    assert sources == {None, AS.OFFERED, AS.ACCEPTED}


def test_only_accept_and_decline_are_crew_actions():
    from mission_control.authz import Permission

    crew_driven = {
        event for (_, event), t in ASSIGNMENT_TRANSITIONS.items()
        if t.permission is Permission.ASSIGNMENT_RESPOND_OWN
    }
    assert crew_driven == {"accept", "decline"}


SYSTEM_ASSIGNMENT_EVENTS = sorted(
    {
        event
        for (_, event), t in ASSIGNMENT_TRANSITIONS.items()
        if t.permission is None
    }
)


@pytest.mark.parametrize("event", SYSTEM_ASSIGNMENT_EVENTS)
def test_system_assignment_events_are_not_on_any_service(world, event):
    """``release``, ``complete``, ``abort`` and ``offer_expired`` are driven by a
    mission event or the clock. They are not callable at all — not by Crew, not
    by a Lead, not by a Director."""
    assert event in {"release", "complete", "abort", "offer_expired"}
    for session in (world.crew(1), world.lead(), world.director()):
        assert not hasattr(session.assignments, event)


def test_only_a_lead_can_make_an_offer(world):
    """``offer`` is a Lead action — the Lead accepting a matcher proposal — so it
    is on the service, but gated."""
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT})
    mission = world.draft(pilot)

    for session in (world.crew(1), world.director()):
        with pytest.raises(Denied) as caught:
            session.assignments.offer(mission.id, pilot.id, 0, CREW[1])
        assert caught.value.code == "MISSING_PERMISSION"

    with pytest.raises(Denied) as caught:
        world.other_lead().assignments.offer(mission.id, pilot.id, 0, CREW[1])
    assert caught.value.code == "NOT_OWNER"


# ------------------------------------------------------------- offer and accept


def test_an_offer_notifies_the_crew_member_and_says_it_is_unapproved(world):
    """Mitigated by honesty in the UI — the offer shows the mission as
    unapproved (§6.6)."""
    mission, assignment = offer_one(world)

    assert assignment.state is AS.OFFERED
    assert assignment.offer_expires_at == T0 + timedelta(hours=72)

    notes = world.notifications(CREW[1])
    assert len(notes) == 1
    assert "not yet approved" in notes[0].body
    assert "Pilot" in notes[0].body


def test_an_offer_expiry_is_never_past_the_mission_start(world):
    mission, assignment = offer_one(world, start_days=1)
    assert assignment.offer_expires_at == mission.plan.window.start
    assert assignment.offer_expires_at < T0 + timedelta(hours=72)


def test_accepting_books_the_dates(world):
    mission, assignment = offer_one(world)
    accepted = world.crew(1).assignments.accept(assignment.id)

    assert accepted.state is AS.ACCEPTED
    assert accepted.responded_at == T0

    lead_notes = [n.subject for n in world.notifications(mission.created_by)]
    assert "Offer accepted" in lead_notes


def test_declining_reopens_the_slot(world):
    mission, assignment = offer_one(world)
    declined = world.crew(1).assignments.decline(assignment.id)

    assert declined.state is AS.DECLINED
    assert world.lead().missions.crewing_status(mission.id) is CrewingStatus.UNDER_CREWED

    second = world.lead().matching.run_matcher(mission.id)
    assert second.slots[0].member_id != CREW[1], "the decline is respected on re-match"


def test_the_matcher_does_not_re_propose_someone_who_declined(world):
    """A decline is information; re-proposing the same person discards it and
    the loop never terminates. Mission-scoped, so it does not affect their
    standing on any other mission (see FILTER_DECLINED)."""
    mission, assignment = offer_one(world)
    world.crew(1).assignments.decline(assignment.id)

    proposal = world.lead().matching.run_matcher(mission.id)
    assert CREW[1] not in {s.member_id for s in proposal.slots}
    assert CREW[1] not in {a for s in proposal.slots for a in s.alternates}

    near = {n.member_id: n for n in proposal.near_misses}
    assert near[CREW[1]].filter == "declined"
    assert near[CREW[1]].detail == "already declined this mission"


def test_a_decline_on_one_mission_does_not_affect_another(world):
    mission, assignment = offer_one(world)
    world.crew(1).assignments.decline(assignment.id)

    other = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}), title="Europa", start_days=90)
    proposal = world.lead().matching.run_matcher(other.id)
    assert proposal.slots[0].member_id == CREW[1]


def test_a_lead_may_still_offer_to_someone_who_declined(world):
    """The engine should not *suggest* it; a Lead who has spoken to them can
    still do it. That split is the point of the matcher returning a proposal
    rather than making assignments (§7.9)."""
    mission, assignment = offer_one(world)
    world.crew(1).assignments.decline(assignment.id)

    again = world.lead().assignments.offer(mission.id, "r-pilot", 0, CREW[1])
    assert again.state is AS.OFFERED
    assert world.crew(1).assignments.accept(again.id).state is AS.ACCEPTED


def test_crew_can_only_answer_their_own_offer(world):
    mission, assignment = offer_one(world, member=CREW[1])

    with pytest.raises(NotFound):
        world.crew(2).assignments.accept(assignment.id)
    with pytest.raises(NotFound):
        world.crew(2).assignments.decline(assignment.id)

    assert world.lead().workspace.assignment(assignment.id).state is AS.OFFERED


def test_a_lead_cannot_accept_on_a_crew_members_behalf(world):
    mission, assignment = offer_one(world)
    with pytest.raises(Denied) as caught:
        world.lead().assignments.accept(assignment.id)
    assert caught.value.code in {"MISSING_PERMISSION", "NOT_YOUR_ASSIGNMENT"}


def test_accepting_twice_is_an_illegal_transition(world):
    mission, assignment = offer_one(world)
    world.crew(1).assignments.accept(assignment.id)
    with pytest.raises(IllegalTransition):
        world.crew(1).assignments.accept(assignment.id)


def test_declining_after_accepting_is_refused(world):
    mission, assignment = offer_one(world)
    world.crew(1).assignments.accept(assignment.id)

    with pytest.raises(IllegalTransition) as caught:
        world.crew(1).assignments.decline(assignment.id)
    assert caught.value.available == ()


# -------------------------------------------------------------------- offers only in draft


def test_offers_are_only_made_in_draft(world):
    """Matching produces roster changes, roster changes are plan changes (§6.7)."""
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT}, count=2)
    mission = world.mission_in(MS.PENDING_APPROVAL, pilot)

    with pytest.raises(GuardFailed, match="offers are only made in draft"):
        world.lead().assignments.offer(mission.id, pilot.id, 1, CREW[6])


def test_a_slot_cannot_be_double_offered(world):
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT})
    mission = world.draft(pilot)
    lead = world.lead()
    lead.assignments.offer(mission.id, pilot.id, 0, CREW[1])

    with pytest.raises(GuardFailed, match="already has a live assignment"):
        lead.assignments.offer(mission.id, pilot.id, 0, CREW[2])


def test_a_declined_slot_can_be_re_offered(world):
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT})
    mission = world.draft(pilot)
    lead = world.lead()
    first = lead.assignments.offer(mission.id, pilot.id, 0, CREW[1])
    world.crew(1).assignments.decline(first.id)

    second = lead.assignments.offer(mission.id, pilot.id, 0, CREW[2])
    assert second.state is AS.OFFERED


def test_an_ineligible_candidate_cannot_be_offered(world):
    """Where a stale proposal is caught: the matcher ran lock-free on a snapshot,
    and this is the authoritative check (§3.3)."""
    pilot = req("Pilot", {PILOT: Pr.EXPERT})
    mission = world.draft(pilot)

    with pytest.raises(GuardFailed, match="no longer eligible"):
        world.lead().assignments.offer(mission.id, pilot.id, 0, CREW[3])


def test_a_slot_index_out_of_range_is_refused(world):
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT}, count=2)
    mission = world.draft(pilot)
    with pytest.raises(GuardFailed, match="out of range"):
        world.lead().assignments.offer(mission.id, pilot.id, 2, CREW[1])


# ------------------------------------------------------------------ offer expiry


def test_an_unanswered_offer_expires_into_a_decline(world):
    """Expiry *is* a decline. It stops one unresponsive person holding a mission
    indefinitely (§8.2)."""
    mission, assignment = offer_one(world)
    world.advance(hours=73)

    expired = world.lead().system.expire_offers()
    assert [a.id for a in expired] == [assignment.id]
    assert expired[0].state is AS.DECLINED

    lead_notes = [n.subject for n in world.notifications(mission.created_by)]
    assert "Offer let the offer lapse" in lead_notes


def test_an_unexpired_offer_is_left_alone(world):
    mission, assignment = offer_one(world)
    world.advance(hours=71)
    assert world.lead().system.expire_offers() == []
    assert world.lead().workspace.assignment(assignment.id).state is AS.OFFERED


def test_accepting_after_expiry_is_refused(world):
    mission, assignment = offer_one(world)
    world.advance(hours=73)

    with pytest.raises(GuardFailed, match="expired"):
        world.crew(1).assignments.accept(assignment.id)


def test_an_expired_offer_no_longer_blocks_the_mission(world):
    """Harmless, because it stalls a draft rather than an approved mission
    (§6.7)."""
    pilot = req("Pilot", {PILOT: Pr.PROFICIENT})
    mission = world.draft(pilot)
    lead = world.lead()
    lead.assignments.offer(mission.id, pilot.id, 0, CREW[1])

    world.advance(hours=73)
    lead.system.expire_offers()

    assert lead.missions.crewing_status(mission.id) is CrewingStatus.UNDER_CREWED
    replacement = lead.assignments.offer(mission.id, pilot.id, 0, CREW[2])
    world.session(CREW[2]).assignments.accept(replacement.id)
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED


# ---------------------------------------------------- the one genuine race


def test_one_member_cannot_accept_two_overlapping_offers(world):
    """OFFERED applies no hold, so two Leads may court the same person — by
    design, and the first acceptance wins (§8.2)."""
    pilot = req("Pilot", {PILOT: Pr.MASTER}, rid="r-mp")
    first = world.draft(pilot, title="Artemis", start_days=30)
    second = world.draft(pilot, title="Europa", start_days=35)

    lead = world.lead()
    offer_a = lead.assignments.offer(first.id, pilot.id, 0, CREW[1])
    offer_b = lead.assignments.offer(second.id, pilot.id, 0, CREW[1])

    anya = world.crew(1)
    anya.assignments.accept(offer_a.id)

    with pytest.raises(GuardFailed) as caught:
        anya.assignments.accept(offer_b.id)
    assert "Artemis" in str(caught.value) and "overlaps" in str(caught.value)


def test_non_overlapping_offers_can_both_be_accepted(world):
    pilot = req("Pilot", {PILOT: Pr.MASTER}, rid="r-mp")
    first = world.draft(pilot, title="Artemis", start_days=30, length_days=10)
    second = world.draft(pilot, title="Europa", start_days=60, length_days=10)

    lead = world.lead()
    offer_a = lead.assignments.offer(first.id, pilot.id, 0, CREW[1])
    offer_b = lead.assignments.offer(second.id, pilot.id, 0, CREW[1])

    anya = world.crew(1)
    assert anya.assignments.accept(offer_a.id).state is AS.ACCEPTED
    assert anya.assignments.accept(offer_b.id).state is AS.ACCEPTED


def test_concurrent_double_acceptance_resolves_to_exactly_one(world):
    """Conflict-check and write must be atomic per crew member (§3.3).

    Two threads race to accept overlapping offers for the same person. Exactly
    one must win — and it must be resolved at commitment rather than prevented
    by locking the candidate set, because no lock can span the human gap between
    the matcher running and someone answering.
    """
    pilot = req("Pilot", {PILOT: Pr.MASTER}, rid="r-mp")
    first = world.draft(pilot, title="Artemis", start_days=30)
    second = world.draft(pilot, title="Europa", start_days=32)

    lead = world.lead()
    offers = [
        lead.assignments.offer(first.id, pilot.id, 0, CREW[1]),
        lead.assignments.offer(second.id, pilot.id, 0, CREW[1]),
    ]

    barrier = threading.Barrier(2)
    outcomes: list[object] = []
    lock = threading.Lock()

    def attempt(assignment_id):
        session = world.crew(1)
        barrier.wait()
        try:
            result = session.assignments.accept(assignment_id)
        except Exception as error:  # noqa: BLE001
            result = error
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(o.id,)) for o in offers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    accepted = [o for o in outcomes if not isinstance(o, Exception)]
    refused = [o for o in outcomes if isinstance(o, Exception)]

    assert len(accepted) == 1, outcomes
    assert len(refused) == 1 and isinstance(refused[0], GuardFailed)

    stored = [lead.workspace.assignment(o.id).state for o in offers]
    assert sorted(s.value for s in stored) == ["accepted", "offered"]


def test_the_matcher_does_not_block_writes(world):
    """Matcher runs lock-free (§10): availability edits and acceptances proceed
    during a long match, and an invalid proposal row is refused at ``offer``."""
    from mission_control.matching import match

    pilot = req("Pilot", {PILOT: Pr.PROFICIENT}, count=2)
    mission = world.draft(pilot)
    lead = world.lead()

    snap = lead.workspace.snapshot()  # the matcher's view, captured and released

    # Meanwhile, someone else books one of the candidates.
    other = world.draft(req("Pilot", {PILOT: Pr.MASTER}, rid="r-mp"), title="Europa")
    other_offer = lead.assignments.offer(other.id, "r-mp", 0, CREW[1])
    world.crew(1).assignments.accept(other_offer.id)

    # The stale snapshot still proposes Anya. Nothing blocked; nothing broke.
    stale = match(lead.missions.get(mission.id), snap, lead.workspace.settings, now=T0)
    assert CREW[1] in {s.member_id for s in stale.slots}

    # And the authoritative check refuses the row.
    with pytest.raises(GuardFailed, match="no longer eligible"):
        lead.matching.offer_proposal(mission.id, stale)


# ------------------------------------------------- crew responses and the mission


def test_crew_responses_never_move_the_mission(world):
    """Guards against a well-meaning auto-advance (§10).

    When the last crew member accepts, the mission becomes *eligible* for
    submission — a Lead submits it.
    """
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Medic", {MED: Pr.PROFICIENT}, rid="r-med"),
    )
    lead = world.lead()
    offers = lead.matching.offer_proposal(mission.id, lead.matching.run_matcher(mission.id))

    before = lead.missions.get(mission.id)
    for offer in offers:
        world.session(offer.member_id).assignments.accept(offer.id)

    after = lead.missions.get(mission.id)
    assert after.state is before.state is MS.DRAFT
    assert after.plan.version == before.plan.version
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED


def test_crewing_status_walks_the_three_stages(world):
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Medic", {MED: Pr.PROFICIENT}, rid="r-med"),
    )
    lead = world.lead()
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.UNDER_CREWED

    offers = lead.matching.offer_proposal(mission.id, lead.matching.run_matcher(mission.id))
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.AWAITING_RESPONSES

    world.session(offers[0].member_id).assignments.accept(offers[0].id)
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.AWAITING_RESPONSES

    world.session(offers[1].member_id).assignments.accept(offers[1].id)
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED


def test_crewing_status_ignores_optional_slots(world):
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}),
        req("Observer", {MED: Pr.PROFICIENT}, mandatory=False, rid="r-obs"),
    )
    lead = world.lead()
    pilot_offer = lead.assignments.offer(mission.id, "r-pilot", 0, CREW[1])
    world.crew(1).assignments.accept(pilot_offer.id)

    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED


def test_crewing_status_is_never_stored(world):
    """Derived on read, so there is no cached projection to invalidate (§3.3)."""
    from dataclasses import fields

    from mission_control.domain import Mission

    assert "crewing_status" not in {f.name for f in fields(Mission)}


# --------------------------------------------- mission events drive assignments


def test_cancelling_a_mission_frees_its_crew(world):
    """Without this a cancelled mission would hold its crew's dates forever
    (§8.1)."""
    for state in (MS.PENDING_APPROVAL, MS.APPROVED):
        mission = world.mission_in(state, req("Pilot", {PILOT: Pr.MASTER}, rid="r-mp"))
        booked = world.accepted(mission.id)[0].member_id
        world.director().missions.cancel(mission.id, "programme descoped")

        assignments = world.lead().workspace.assignments_for_mission(mission.id)
        assert all(a.state is AS.RELEASED for a in assignments)
        assert all(a.release_reason for a in assignments)

        # Immediately eligible elsewhere in the same window.
        replacement = world.draft(
            req("Pilot", {PILOT: Pr.MASTER}, rid="r-mp"),
            title=f"Replacement {state.value}",
            start_days=30,
        )
        proposal = world.lead().matching.run_matcher(replacement.id)
        assert booked in {s.member_id for s in proposal.slots}
        world.director().missions.cancel(replacement.id, "cleanup")


def test_changing_the_requirements_releases_acceptances(world):
    """They accepted a different plan, so their consent no longer applies
    (§6.7)."""
    from dataclasses import replace

    mission = world.mission_in(MS.APPROVED, req("Pilot", {PILOT: Pr.PROFICIENT}))
    booked = world.accepted(mission.id)[0].member_id

    harder = replace(
        mission.plan, requirements=(req("Pilot", {PILOT: Pr.MASTER}),)
    )
    world.lead().missions.reopen(mission.id)
    world.lead().missions.update_plan(mission.id, harder)

    assignments = world.lead().workspace.assignments_for_mission(mission.id)
    assert all(a.state is AS.RELEASED for a in assignments)

    notes = [n.subject for n in world.notifications(booked)]
    assert "Assignment released" in notes


def test_changing_only_the_site_keeps_acceptances_within_draft(world):
    """§6.7 releases accepted crew for a requirements or window change, not a
    site change — but only for edits made *inside* DRAFT.

    Reaching DRAFT from a submitted or approved mission goes through ``reopen``,
    which releases everyone regardless, so this rule now applies exactly where a
    mission has not yet been shown to a Director.
    """
    from dataclasses import replace

    mission = world.draft(req("Pilot", {PILOT: Pr.PROFICIENT}))
    world.crew_up(mission.id)
    assert len(world.accepted(mission.id)) == 1

    result = world.lead().missions.update_plan(
        mission.id, replace(mission.plan, site=Location("SLC-41"))
    )

    assert result.state is MS.DRAFT
    assert len(world.accepted(mission.id)) == 1


def test_offers_and_acceptances_are_treated_alike_by_a_plan_edit(world):
    """An open offer is a pending question. If the specification behind it is
    unchanged the question is still the right one, so a cosmetic edit keeps it —
    the same rule that keeps an acceptance (§6.7)."""
    from dataclasses import replace

    pilot = req("Pilot", {PILOT: Pr.PROFICIENT}, count=2)
    mission = world.draft(pilot)
    lead = world.lead()
    accepted_offer = lead.assignments.offer(mission.id, pilot.id, 0, CREW[1])
    world.crew(1).assignments.accept(accepted_offer.id)
    open_offer = lead.assignments.offer(mission.id, pilot.id, 1, CREW[2])

    lead.missions.update_plan(mission.id, replace(mission.plan, site=Location("SLC-41")))
    assert lead.workspace.assignment(open_offer.id).state is AS.OFFERED
    assert lead.workspace.assignment(accepted_offer.id).state is AS.ACCEPTED

    lead.missions.update_plan(mission.id, replace(
        mission.plan, requirements=(req("Pilot", {PILOT: Pr.MASTER}, count=2),)))
    assert lead.workspace.assignment(open_offer.id).state is AS.RELEASED
    assert lead.workspace.assignment(accepted_offer.id).state is AS.RELEASED


def test_completing_a_mission_writes_assignment_history(world):
    mission = world.mission_in(MS.COMPLETED, req("Pilot", {PILOT: Pr.PROFICIENT}))
    assignments = world.lead().workspace.assignments_for_mission(mission.id)
    assert [a.state for a in assignments] == [AS.COMPLETED]


def test_aborting_a_mission_writes_partial_history(world):
    """"Flew it" and "flew part of an aborted mission" should not need
    re-deriving from a mission record years later (§8.1)."""
    mission = world.mission_in(MS.ABORTED, req("Pilot", {PILOT: Pr.PROFICIENT}))
    assignments = world.lead().workspace.assignments_for_mission(mission.id)
    assert [a.state for a in assignments] == [AS.PARTIAL]


def test_completing_a_mission_releases_an_unanswered_offer(world):
    mission = world.draft(
        req("Pilot", {PILOT: Pr.PROFICIENT}, rid="r-main"),
        req("Spare", {PILOT: Pr.PROFICIENT}, mandatory=False, rid="r-spare"),
    )
    lead = world.lead()
    main = lead.assignments.offer(mission.id, "r-main", 0, CREW[1])
    world.crew(1).assignments.accept(main.id)
    spare = lead.assignments.offer(mission.id, "r-spare", 0, CREW[2])

    lead.missions.submit(mission.id)
    world.director().missions.approve(mission.id)
    window = lead.missions.get(mission.id).plan.window
    lead.missions.activate(mission.id, now=window.start)
    lead.missions.complete(mission.id, now=window.end)

    assert lead.workspace.assignment(main.id).state is AS.COMPLETED
    assert lead.workspace.assignment(spare.id).state is AS.RELEASED


def test_deactivating_a_member_releases_their_commitments(world):
    mission = world.mission_in(MS.APPROVED, req("Pilot", {PILOT: Pr.PROFICIENT}))
    booked = world.accepted(mission.id)[0].member_id

    world.director().org.deactivate_member(booked, "left the programme")

    assignments = world.lead().workspace.assignments_for_mission(mission.id)
    assert all(a.state is AS.RELEASED for a in assignments)


# ----------------------------------------------------------- the coupling table


def test_approving_and_activating_do_not_touch_the_assignments(world):
    """The crew are already ACCEPTED, which is the whole point of crewing before
    approval (§6.6, §8.3)."""
    mission = world.mission_in(MS.PENDING_APPROVAL, req("Pilot", {PILOT: Pr.PROFICIENT}))
    before = world.lead().workspace.assignments_for_mission(mission.id)

    world.director().missions.approve(mission.id)
    assert world.lead().workspace.assignments_for_mission(mission.id) == before

    world.lead().missions.activate(mission.id, now=mission.plan.window.start)
    assert world.lead().workspace.assignments_for_mission(mission.id) == before


def test_approval_notifies_the_crew(world):
    mission = world.mission_in(MS.PENDING_APPROVAL, req("Pilot", {PILOT: Pr.PROFICIENT}))
    booked = world.accepted(mission.id)[0].member_id
    world.director().missions.approve(mission.id)

    assert "approved" in [n.subject for n in world.notifications(booked)]
