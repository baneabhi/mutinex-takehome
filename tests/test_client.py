"""The typed client.

Runs against the real app in-process, so these are integration tests of the
whole stack — client, HTTP, services, domain — with no server to start. What
they pin is the client's own contributions: parsing into the shared schema
models, preserving error codes, and turning a proposal into offers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mission_control import schemas as s
from mission_control.client import ApiError, make_plan, requirement

START = datetime(2099, 3, 1, tzinfo=timezone.utc)
END = START + timedelta(days=20)


def pilot_plan() -> s.PlanIn:
    return make_plan(
        [requirement("Pilot", {"pilot": "PROFICIENT"}, id="r-pilot")],
        start=START, end=END, site="LC-39A",
    )


def test_it_parses_into_the_shared_models(api):
    lead = api.client("nasa", "mem-lead")
    mission = lead.create_mission("Artemis VII", pilot_plan())

    assert isinstance(mission, s.MissionDetail)
    assert mission.state == "draft"
    assert mission.plan.requirements[0].skills[0].min_level == "PROFICIENT"
    assert isinstance(mission.window.start, datetime)


def test_errors_keep_their_code_and_their_way_out(api):
    lead = api.client("nasa", "mem-lead")
    mission = lead.create_mission("Artemis", pilot_plan())

    with pytest.raises(ApiError) as caught:
        lead.activate(mission.id)
    assert caught.value.status == 409
    assert caught.value.code == "ILLEGAL_TRANSITION"
    assert "submit" in caught.value.available

    with pytest.raises(ApiError) as caught:
        lead.submit(mission.id)
    assert caught.value.code == "PRECONDITION_FAILED"
    assert "under_crewed" in caught.value.reason

    with pytest.raises(ApiError) as caught:
        lead.add_member("X", "crew")
    assert caught.value.status == 403
    assert caught.value.code == "MISSING_PERMISSION"


def test_as_actor_switches_identity_over_one_connection(api):
    director = api.client("nasa", "mem-director")
    lead = director.as_actor(api.token("nasa", "mem-lead"))

    assert director.me().role == "director"
    assert lead.me().role == "mission_lead"
    assert lead._http is director._http


def test_offer_proposal_turns_rows_into_offers_and_skips_pinned(api):
    """The client's convenience over the explicit-rows endpoint — and it must
    not re-offer somebody who already accepted (§6.7)."""
    lead = api.client("nasa", "mem-lead")
    mission = lead.create_mission("Artemis", make_plan(
        [requirement("Pilot", {"pilot": "PROFICIENT"}, id="r-pilot"),
         requirement("Medic", {"med": "EXPERT"}, id="r-medic")],
        start=START, end=END, site="LC-39A"))

    first = lead.run_matcher(mission.id)
    offers = lead.offer_proposal(mission.id, first)
    assert len(offers) == 2

    accepted, declined = offers
    api.client("nasa", accepted.member_id).accept(accepted.id)
    api.client("nasa", declined.member_id).decline(declined.id)

    second = lead.run_matcher(mission.id)
    assert sum(1 for row in second.slots if row.pinned) == 1

    refills = lead.offer_proposal(mission.id, second)
    assert len(refills) == 1, "the pinned row must not be re-offered"
    assert refills[0].member_id != accepted.member_id


def test_a_crew_member_gets_the_redacted_model(api):
    """The client cannot assume MissionDetail, and must not pretend the missing
    fields are None — so it returns a different type (§5.3)."""
    lead = api.client("nasa", "mem-lead")
    mission = lead.create_mission("Artemis", pilot_plan())
    offer = lead.offer_proposal(mission.id, lead.run_matcher(mission.id))[0]

    view = api.client("nasa", offer.member_id).mission(mission.id)
    assert isinstance(view, s.MissionForCrew)
    assert view.my_assignment.state == "offered"
    assert not hasattr(view, "created_by")


def test_the_full_workflow_through_the_client(api):
    lead = api.client("nasa", "mem-lead")
    director = api.client("nasa", "mem-director")

    mission = lead.create_mission("Artemis VII", make_plan(
        [requirement("Pilot", {"pilot": "PROFICIENT"}, id="r-pilot"),
         requirement("Flight Surgeon", {"med": "EXPERT"}, id="r-surgeon")],
        start=START, end=END, site="LC-39A"))

    proposal = lead.run_matcher(mission.id)
    assert proposal.is_complete
    for offer in lead.offer_proposal(mission.id, proposal):
        api.client("nasa", offer.member_id).accept(offer.id)

    assert lead.submit(mission.id).state == "pending_approval"
    assert director.approve(mission.id).state == "approved"

    # Frozen once approved; reopening is the way back, and it costs the approval.
    with pytest.raises(ApiError) as caught:
        lead.update_plan(mission.id, pilot_plan())
    assert "reopen" in caught.value.available

    reopened = lead.reopen(mission.id)
    assert reopened.state == "draft"
    assert reopened.approved_by is None


def test_tenant_isolation_holds_through_the_client(api):
    nasa = api.client("nasa", "mem-lead")
    esa = api.client("esa", "mem-lead")
    mission = nasa.create_mission("Artemis", pilot_plan())

    assert esa.missions() == []
    with pytest.raises(ApiError) as caught:
        esa.mission(mission.id)
    assert caught.value.status == 404
