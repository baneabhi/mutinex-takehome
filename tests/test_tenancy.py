"""Tenant isolation (§3, §10).

Every test here runs against two organisations with colliding ids. A leak shows
up as *the wrong tenant's data*, not as an error — which is exactly why a
single-tenant fixture cannot catch one.
"""

from __future__ import annotations

import inspect

import pytest

from mission_control import services
from mission_control.domain import Denied, NotFound, Role
from mission_control.store import Platform

from .conftest import CREW, DIRECTOR, ESA, NASA


def test_a_mission_created_in_one_tenant_is_invisible_in_the_other(world):
    mission = world.draft(tenant_id=NASA, title="Artemis VII")

    esa_lead = world.lead(ESA)
    assert esa_lead.missions.list() == []
    with pytest.raises(NotFound):
        esa_lead.missions.get(mission.id)


def test_a_valid_foreign_id_returns_not_found_not_forbidden(world):
    """A 403 confirms the record exists, which is an enumeration oracle (§3.2)."""
    mission = world.draft(tenant_id=NASA)

    for session in (world.director(ESA), world.lead(ESA), world.crew(1, ESA)):
        with pytest.raises(NotFound) as caught:
            session.missions.get(mission.id)
        assert not isinstance(caught.value, Denied)


def test_colliding_ids_resolve_to_the_callers_own_tenant(world):
    """Both tenants have a ``mem-crew-1`` named Anya. Reading one must never
    return the other's."""
    nasa_mission = world.draft(tenant_id=NASA, title="Artemis VII")
    esa_mission = world.draft(tenant_id=ESA, title="Rosetta II")

    assert nasa_mission.id == esa_mission.id, "ids must collide for this test to mean anything"

    assert world.lead(NASA).missions.get(nasa_mission.id).title == "Artemis VII"
    assert world.lead(ESA).missions.get(esa_mission.id).title == "Rosetta II"


def test_crew_profiles_and_assignments_do_not_cross(world):
    mission = world.draft(tenant_id=NASA)
    world.crew_up(mission.id, NASA)
    nasa_assignment = world.accepted(mission.id, NASA)[0]

    esa_lead = world.lead(ESA)
    with pytest.raises(NotFound):
        esa_lead.workspace.assignment(nasa_assignment.id)
    assert esa_lead.workspace.assignments_for_mission(mission.id) == []
    assert esa_lead.workspace.assignments_for_member(CREW[1]) == []


def test_audit_logs_do_not_cross(world):
    world.draft(tenant_id=NASA)
    assert world.director(NASA).org.audit()
    assert world.director(ESA).org.audit() == []


def test_a_token_naming_another_tenants_member_is_rejected(world):
    """What a forged cross-tenant token looks like from inside ``session``."""
    platform = world.platform
    only_in_nasa = Role.DIRECTOR
    platform._unsafe_tenant(ESA).members.pop(DIRECTOR)

    actor = platform.actor_for(NASA, DIRECTOR)
    assert actor.role is only_in_nasa

    forged = type(actor)(tenant_id=ESA, member_id=DIRECTOR, role=only_in_nasa)
    with pytest.raises(NotFound):
        platform.session(forged)


def test_a_token_claiming_the_wrong_role_is_rejected(world):
    actor = world.platform.actor_for(NASA, CREW[1])
    escalated = type(actor)(
        tenant_id=NASA, member_id=CREW[1], role=Role.DIRECTOR
    )
    with pytest.raises(Denied) as caught:
        world.platform.session(escalated)
    assert caught.value.code == "ROLE_MISMATCH"


def test_an_unknown_tenant_is_not_found():
    platform = Platform()
    from mission_control.domain import Actor, MemberId, TenantId

    with pytest.raises(NotFound):
        platform.session(Actor(TenantId("nope"), MemberId("x"), Role.DIRECTOR))


def test_no_service_method_accepts_a_tenant_id(world):
    """Structural, not behavioural (§3.2 rule 1).

    A caller cannot ask for another tenant because there is no argument through
    which to. Asserting it by introspection means a method added later that
    takes one fails here rather than in review.
    """
    offenders = []
    for service_name in (
        "MissionService", "MatchingService", "AssignmentService",
        "CrewService", "OrgService", "SystemService",
    ):
        service = getattr(services, service_name)
        for name, method in inspect.getmembers(service, inspect.isfunction):
            if name.startswith("_"):
                continue
            for parameter in inspect.signature(method).parameters:
                if "tenant" in parameter.lower():
                    offenders.append(f"{service_name}.{name}({parameter})")
    assert offenders == []


def test_the_workspace_does_not_expose_the_platform(world):
    """Rule 2: TenantWorkspace closes over exactly one Tenant and never hands
    back a route to the others."""
    workspace = world.lead(NASA).workspace
    reachable = {
        name: getattr(workspace, name)
        for name in dir(workspace)
        if not name.startswith("__")
    }
    assert not any(isinstance(value, Platform) for value in reachable.values())


def test_session_is_the_only_way_to_reach_a_tenant(world):
    """Rule 2, from the other side: the public surface of Platform hands out
    workspaces and ids, never Tenants."""
    public = [
        name
        for name in dir(world.platform)
        if not name.startswith("_") and callable(getattr(world.platform, name))
    ]
    assert sorted(public) == ["actor_for", "create_tenant", "session", "tenant_ids"]
