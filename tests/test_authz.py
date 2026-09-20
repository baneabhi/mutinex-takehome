"""Authorisation (§5, §10).

The interesting tests here are the table-driven ones. A permission added later
has to decide explicitly what each role does with it, rather than defaulting to
whatever the first handler happens to check.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mission_control.authz import (
    AUTHORING_PERMISSIONS,
    ROLE_PERMISSIONS,
    Permission,
    SeparationOfDuties,
    has,
)
from mission_control.domain import Denied, MissionState, NotFound, Role
from mission_control.lifecycle import MISSION_TRANSITIONS

from .conftest import CREW, DIRECTOR

P = Permission

EXPECTED = {
    Role.DIRECTOR: {
        P.ORG_SETTINGS_MANAGE, P.MEMBER_MANAGE, P.AUDIT_READ,
        P.MISSION_APPROVE, P.MISSION_REJECT, P.MISSION_CANCEL,
        P.MISSION_VIEW_ALL, P.MISSION_VIEW_ASSIGNED, P.CREW_PROFILE_READ_ALL,
    },
    Role.MISSION_LEAD: {
        P.MISSION_CREATE, P.MISSION_UPDATE, P.MISSION_SUBMIT,
        P.MATCHER_RUN, P.ASSIGNMENT_OFFER,
        P.MISSION_ACTIVATE, P.MISSION_COMPLETE, P.MISSION_CANCEL,
        P.MISSION_VIEW_ALL, P.MISSION_VIEW_ASSIGNED, P.CREW_PROFILE_READ_ALL,
    },
    Role.CREW: {
        P.MISSION_VIEW_ASSIGNED, P.CREW_PROFILE_WRITE_OWN,
        P.CREW_AVAILABILITY_WRITE_OWN, P.ASSIGNMENT_RESPOND_OWN,
    },
}


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("permission", list(Permission))
def test_the_capability_matrix_is_exactly_section_5_2(role, permission):
    """Every cell, both directions. A new permission fails here until someone
    writes it down."""
    assert has_permission(role, permission) is (permission in EXPECTED[role])


def has_permission(role, permission):
    return permission in ROLE_PERMISSIONS[role]


def test_the_three_roles_hold_disjoint_responsibilities():
    """Directors govern, Leads plan, Crew execute (§5.2)."""
    director = ROLE_PERMISSIONS[Role.DIRECTOR]
    lead = ROLE_PERMISSIONS[Role.MISSION_LEAD]
    crew = ROLE_PERMISSIONS[Role.CREW]

    shared_view = {P.MISSION_VIEW_ALL, P.MISSION_VIEW_ASSIGNED, P.CREW_PROFILE_READ_ALL}
    assert (director & lead) - shared_view == {P.MISSION_CANCEL}
    assert (director & crew) == {P.MISSION_VIEW_ASSIGNED}
    assert (lead & crew) == {P.MISSION_VIEW_ASSIGNED}


def test_directors_hold_no_authoring_permissions():
    """This is what makes separation of duties structural rather than
    policy-dependent: a Director cannot approve their own mission because there
    is no such mission (§5.2)."""
    assert not (ROLE_PERMISSIONS[Role.DIRECTOR] & AUTHORING_PERMISSIONS)


@pytest.mark.parametrize(
    "event",
    sorted(
        {
            event
            for (_, event), transition in MISSION_TRANSITIONS.items()
            if transition.permission in AUTHORING_PERMISSIONS
        }
    ),
)
def test_every_authoring_event_is_denied_to_a_director(world, event):
    """Table-driven over the transition table, so an authoring event added later
    must decide explicitly rather than inheriting a Director's permissions."""
    states = [state for (state, e) in MISSION_TRANSITIONS if e == event]
    for state in states:
        mission = world.mission_in(state)
        with pytest.raises(Denied) as caught:
            world.director().missions.fire(mission.id, event, reason="because")
        assert caught.value.code == "MISSING_PERMISSION"


def test_the_crew_write_surface_is_exactly_two_events():
    """ASSIGNMENT_RESPOND_OWN is the only permission Crew hold that mutates
    anything beyond their own profile, and it appears exactly twice (§8.2)."""
    from mission_control.lifecycle import ASSIGNMENT_TRANSITIONS

    crew_events = sorted(
        event
        for (_, event), transition in ASSIGNMENT_TRANSITIONS.items()
        if transition.permission is P.ASSIGNMENT_RESPOND_OWN
    )
    assert crew_events == ["accept", "decline"]

    mission_events = [
        event
        for (_, event), transition in MISSION_TRANSITIONS.items()
        if transition.permission is not None and has(_crew_actor(), transition.permission)
    ]
    assert mission_events == [], "Crew must not be able to move a mission"


def _crew_actor():
    from mission_control.domain import Actor, MemberId, TenantId

    return Actor(TenantId("t"), MemberId("m"), Role.CREW)


# ------------------------------------------------------------ separation of duties


def test_self_approval_is_denied_by_the_backstop(world):
    """Tests the backstop **as** a backstop, which the normal path cannot reach
    (§10).

    §5.2 makes a Director-authored mission impossible, so the only way here is
    to construct what an import would: a mission whose ``created_by`` is a
    Director. If a role is widened later or a fourth role added, this is what
    still holds.
    """
    mission = world.mission_in(MissionState.PENDING_APPROVAL)
    workspace = world.lead().workspace
    imported = replace(mission, created_by=DIRECTOR, version=mission.version + 1)
    workspace.save_mission(imported)

    with pytest.raises(Denied) as caught:
        world.director().missions.approve(mission.id)
    assert caught.value.code == "SELF_APPROVAL"

    # Another Director is unaffected.
    assert world.other_director().missions.approve(mission.id).state is MissionState.APPROVED


def test_the_backstop_checks_the_submitter_too(world):
    """Checking only ``created_by`` is defeated by asking a colleague to press
    Submit; checking only ``submitted_by`` by asking one to press Create."""
    policy = SeparationOfDuties()
    mission = world.mission_in(MissionState.PENDING_APPROVAL)

    from mission_control.domain import Actor, TenantId

    submitter = Actor(TenantId("t-nasa"), mission.submitted_by, Role.DIRECTOR)
    assert policy.check(submitter, mission, None) is not None

    author = Actor(TenantId("t-nasa"), mission.created_by, Role.DIRECTOR)
    assert policy.check(author, mission, None) is not None

    stranger = Actor(TenantId("t-nasa"), DIRECTOR, Role.DIRECTOR)
    assert policy.check(stranger, mission, None) is None


def test_authorisation_is_checked_before_the_approval_gate(world):
    """Otherwise ``approve`` becomes a way for anyone to bounce someone else's
    mission back to DRAFT."""
    mission = world.mission_in(MissionState.PENDING_APPROVAL)

    with pytest.raises(Denied):
        world.crew(1).missions.approve(mission.id)
    with pytest.raises(Denied):
        world.lead().missions.approve(mission.id)

    assert world.lead().missions.get(mission.id).state is MissionState.PENDING_APPROVAL


def test_system_events_are_not_reachable_from_a_request(world):
    """``invalidated`` and ``expire`` carry no permission because nobody decided
    them. Exposing them on a request path would let a Crew member bounce another
    Lead's mission."""
    mission = world.mission_in(MissionState.PENDING_APPROVAL)
    for session in (world.crew(1), world.lead(), world.director()):
        with pytest.raises(Denied) as caught:
            session.missions.fire(mission.id, "invalidated")
        assert caught.value.code == "SYSTEM_EVENT"
    assert world.lead().missions.get(mission.id).state is MissionState.PENDING_APPROVAL


# ---------------------------------------------------------------------- ownership


def test_a_lead_cannot_act_on_another_leads_mission(world):
    mission = world.draft()
    other = world.other_lead()

    for event, kwargs in [("submit", {}), ("cancel", {"reason": "no"}),
                          ("run_matcher", {})]:
        with pytest.raises(Denied) as caught:
            if event == "run_matcher":
                other.matching.run_matcher(mission.id)
            else:
                other.missions.fire(mission.id, event, **kwargs)
        assert caught.value.code == "NOT_OWNER"


def test_a_director_may_cancel_a_mission_they_did_not_author(world):
    """Cancellation is the one exception, and it is not authorship: approval
    should not be a one-way door (§5.2)."""
    mission = world.mission_in(MissionState.APPROVED)
    cancelled = world.director().missions.cancel(mission.id, "programme descoped")
    assert cancelled.state is MissionState.CANCELLED


# --------------------------------------------------------------------- visibility


def test_crew_see_only_missions_they_are_assigned_to(world):
    assigned = world.draft(title="Artemis VII")
    world.crew_up(assigned.id)
    unrelated = world.draft(title="Europa Clipper")

    anya = world.crew(1)
    visible = [m.id for m in anya.missions.list()]
    assert assigned.id in visible
    assert unrelated.id not in visible

    with pytest.raises(NotFound):
        anya.missions.get(unrelated.id)


def test_listing_filters_but_direct_access_raises(world):
    """Listing filters rather than rejecting; direct access to a non-visible
    mission raises NotFound (§5.3)."""
    mission = world.draft()
    chen = world.crew(3)
    assert chen.missions.list() == []
    with pytest.raises(NotFound):
        chen.missions.get(mission.id)


def test_crew_cannot_read_another_members_profile(world):
    with pytest.raises(NotFound):
        world.crew(1).crew.profile(CREW[2])
    assert world.crew(1).crew.profile(CREW[1]) is not None
    assert world.lead().crew.profile(CREW[2]) is not None
    assert world.director().crew.profile(CREW[2]) is not None


def test_crew_cannot_write_another_members_profile(world):
    from mission_control.domain import CrewSkill, Proficiency

    from .conftest import PILOT

    with pytest.raises(Denied) as caught:
        world.crew(1).crew.set_skills(CREW[2], [CrewSkill(PILOT, Proficiency.MASTER)])
    assert caught.value.code == "NOT_YOUR_PROFILE"


def test_a_director_may_correct_anyones_profile(world):
    from mission_control.domain import CrewSkill, Proficiency

    from .conftest import PILOT

    profile = world.director().crew.set_skills(
        CREW[3], [CrewSkill(PILOT, Proficiency.EXPERT)]
    )
    assert profile.level(PILOT) is Proficiency.EXPERT
