"""Shared fixtures.

**The two-tenant fixture is the default** (§10). A single-tenant fixture
structurally cannot catch a scoping bug, so every test in the suite runs against
a platform holding two organisations whose member ids, skill ids and member
*names* deliberately collide. If any code path resolves an id without going
through the actor's tenant, something here will return the wrong organisation's
data rather than nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from mission_control.domain import (
    Assignment,
    AssignmentId,
    AssignmentState,
    CrewProfile,
    CrewSkill,
    Location,
    Member,
    MemberId,
    Mission,
    MissionId,
    MissionPlan,
    MissionState,
    Proficiency,
    Requirement,
    RequirementId,
    Role,
    Skill,
    SkillId,
    SkillRequirement,
    TenantId,
    TimeWindow,
    UTC,
)
from mission_control.services import Scope
from mission_control.store import Platform

T0 = datetime(2026, 1, 1, tzinfo=UTC)

NASA = TenantId("t-nasa")
ESA = TenantId("t-esa")

# Identical in both tenants, on purpose.
DIRECTOR = MemberId("mem-director")
DIRECTOR_B = MemberId("mem-director-b")
LEAD = MemberId("mem-lead")
LEAD_B = MemberId("mem-lead-b")
CREW = {n: MemberId(f"mem-crew-{n}") for n in range(1, 7)}

PILOT = SkillId("pilot")
MED = SkillId("med")
PHYS = SkillId("phys")
ENG = SkillId("eng")

Pr = Proficiency

CREW_SKILLS: dict[int, tuple[str, dict[SkillId, Proficiency]]] = {
    1: ("Anya", {PILOT: Pr.MASTER, MED: Pr.EXPERT}),
    2: ("Boris", {PILOT: Pr.EXPERT, MED: Pr.NOVICE}),
    3: ("Chen", {PILOT: Pr.NOVICE, MED: Pr.NOVICE}),
    4: ("Dara", {MED: Pr.EXPERT, PHYS: Pr.PROFICIENT}),
    5: ("Eli", {MED: Pr.PROFICIENT, ENG: Pr.COMPETENT}),
    6: ("Fen", {PILOT: Pr.PROFICIENT, ENG: Pr.EXPERT}),
}

STAFF = [
    (DIRECTOR, "Dana", Role.DIRECTOR),
    (DIRECTOR_B, "Devi", Role.DIRECTOR),
    (LEAD, "Lena", Role.MISSION_LEAD),
    (LEAD_B, "Liam", Role.MISSION_LEAD),
]

SKILL_NAMES = {PILOT: "Piloting", MED: "Medicine", PHYS: "Physiology", ENG: "Engineering"}


def seed(platform: Platform, tenant_id: TenantId, name: str) -> None:
    """Install members, profiles and a skill taxonomy directly.

    Uses the underscored seeding hook rather than OrgService because the fixture
    needs *fixed* ids in order to collide across tenants, and because a tenant's
    first Director cannot be created through a path that requires a Director.
    """
    platform.create_tenant(tenant_id, name)
    tenant = platform._unsafe_tenant(tenant_id)

    for member_id, member_name, role in STAFF:
        tenant.members[member_id] = Member(member_id, member_name, role)
    for skill_id, skill_name in SKILL_NAMES.items():
        tenant.skills[skill_id] = Skill(skill_id, skill_name)
    for index, (member_name, skills) in CREW_SKILLS.items():
        member_id = CREW[index]
        tenant.members[member_id] = Member(member_id, member_name, Role.CREW)
        tenant.crew_profiles[member_id] = CrewProfile(
            member_id, tuple(CrewSkill(s, lvl) for s, lvl in skills.items())
        )


def req(
    label: str,
    skills: dict[SkillId, Proficiency],
    *,
    count: int = 1,
    mandatory: bool = True,
    rid: str | None = None,
) -> Requirement:
    return Requirement(
        id=RequirementId(rid or f"r-{label.lower().replace(' ', '-')}"),
        label=label,
        count=count,
        skills=tuple(SkillRequirement(s, lvl) for s, lvl in skills.items()),
        mandatory=mandatory,
    )


def plan(
    *requirements: Requirement,
    start_days: int = 30,
    length_days: int = 15,
    site: str = "LC-39A",
) -> MissionPlan:
    return MissionPlan(
        requirements=requirements or (req("Pilot", {PILOT: Pr.PROFICIENT}),),
        window=TimeWindow(
            T0 + timedelta(days=start_days), T0 + timedelta(days=start_days + length_days)
        ),
        site=Location(site),
    )


@dataclass
class World:
    """A platform with two seeded tenants and one movable clock."""

    platform: Platform
    now: datetime = T0

    def clock(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now = self.now + timedelta(**delta)
        return self.now

    def session(self, member_id: MemberId, tenant_id: TenantId = NASA) -> Scope:
        actor = self.platform.actor_for(tenant_id, member_id)
        return Scope(self.platform.session(actor), self.clock)

    # -- convenience builders

    def director(self, tenant_id: TenantId = NASA) -> Scope:
        return self.session(DIRECTOR, tenant_id)

    def other_director(self, tenant_id: TenantId = NASA) -> Scope:
        return self.session(DIRECTOR_B, tenant_id)

    def lead(self, tenant_id: TenantId = NASA) -> Scope:
        return self.session(LEAD, tenant_id)

    def other_lead(self, tenant_id: TenantId = NASA) -> Scope:
        return self.session(LEAD_B, tenant_id)

    def crew(self, index: int, tenant_id: TenantId = NASA) -> Scope:
        return self.session(CREW[index], tenant_id)

    def draft(
        self,
        *requirements: Requirement,
        tenant_id: TenantId = NASA,
        title: str = "Artemis VII",
        **plan_kwargs,
    ) -> Mission:
        return self.lead(tenant_id).missions.create(
            title, plan(*requirements, **plan_kwargs)
        )

    def crew_up(self, mission_id: MissionId, tenant_id: TenantId = NASA) -> list[Assignment]:
        """Run the matcher, offer every proposed slot, and have everyone accept.

        This is the DRAFT-phase crewing loop: crew accept *before* submission,
        so ``submit`` asks "is this staffed?" rather than "does this look
        staffable?" (§6.6).
        """
        lead = self.lead(tenant_id)
        proposal = lead.matching.run_matcher(mission_id)
        offers = lead.matching.offer_proposal(mission_id, proposal)
        for offer in offers:
            self.session(offer.member_id, tenant_id).assignments.accept(offer.id)
        return lead.workspace.assignments_for_mission(mission_id)

    def mission_in(
        self,
        state: MissionState,
        *requirements: Requirement,
        tenant_id: TenantId = NASA,
        **plan_kwargs,
    ) -> Mission:
        """Drive a mission to ``state`` through the real transitions.

        No back door: reaching APPROVED means crew genuinely accepted and a
        Director genuinely approved, which is what makes the exhaustive sweep
        meaningful.
        """
        lead = self.lead(tenant_id)
        director = self.director(tenant_id)
        mission = self.draft(*requirements, tenant_id=tenant_id, **plan_kwargs)

        if state is MissionState.DRAFT:
            return lead.missions.get(mission.id)
        if state is MissionState.CANCELLED:
            return lead.missions.cancel(mission.id, "scrubbed")

        self.crew_up(mission.id, tenant_id)
        lead.missions.submit(mission.id)
        if state is MissionState.PENDING_APPROVAL:
            return lead.missions.get(mission.id)

        director.missions.approve(mission.id)
        if state is MissionState.APPROVED:
            return lead.missions.get(mission.id)

        window_start = lead.missions.get(mission.id).plan.window.start
        lead.missions.activate(mission.id, now=window_start)
        if state is MissionState.ACTIVE:
            return lead.missions.get(mission.id)
        if state is MissionState.ABORTED:
            return lead.missions.abort(mission.id, "vehicle anomaly", now=window_start)
        if state is MissionState.COMPLETED:
            window_end = lead.missions.get(mission.id).plan.window.end
            return lead.missions.complete(mission.id, now=window_end)

        raise AssertionError(f"unhandled state {state}")

    def seed_history(
        self,
        member_id: MemberId,
        *,
        ended_days_ago: int,
        length_days: int = 10,
        state: AssignmentState = AssignmentState.COMPLETED,
        tenant_id: TenantId = NASA,
        title: str = "Historic",
    ) -> Mission:
        """Install a finished mission and assignment directly.

        The fairness key reads assignment *history*, and history cannot be
        manufactured through the lifecycle: an offer's expiry is capped at the
        mission's start (§8.2), so a mission in the past can never be crewed.
        Seeding the end state directly keeps these tests about ``rank_key``
        rather than about the clock.
        """
        tenant = self.platform._unsafe_tenant(tenant_id)
        end = T0 - timedelta(days=ended_days_ago)
        window = TimeWindow(end - timedelta(days=length_days), end)
        mission_id = MissionId(f"mis-hist-{len(tenant.missions)}")
        requirement = req("Crew", {PILOT: Proficiency.NOVICE}, rid="r-hist")

        mission = Mission(
            id=mission_id,
            title=title,
            plan=MissionPlan(requirements=(requirement,), window=window, site=Location("Past")),
            created_by=LEAD,
            state=MissionState.COMPLETED,
        )
        assignment = Assignment(
            id=AssignmentId(f"asg-hist-{len(tenant.assignments)}"),
            mission_id=mission_id,
            member_id=member_id,
            requirement_id=requirement.id,
            slot_index=0,
            state=state,
            offered_at=window.start - timedelta(days=30),
            offer_expires_at=window.start,
            responded_at=window.start - timedelta(days=29),
        )
        with tenant.lock:
            tenant.missions[mission_id] = mission
            tenant.assignments[assignment.id] = assignment
            tenant.by_member.setdefault(member_id, []).append(assignment.id)
        return mission

    def accepted(self, mission_id: MissionId, tenant_id: TenantId = NASA) -> list[Assignment]:
        return [
            a
            for a in self.lead(tenant_id).workspace.assignments_for_mission(mission_id)
            if a.state is AssignmentState.ACCEPTED
        ]

    def notifications(self, member_id: MemberId, tenant_id: TenantId = NASA):
        return self.lead(tenant_id).workspace.notifications(member_id)


@pytest.fixture
def world() -> World:
    platform = Platform()
    seed(platform, NASA, "NASA")
    seed(platform, ESA, "European Space Agency")
    return World(platform)


# ------------------------------------------------------------------- HTTP layer


@pytest.fixture
def api():
    """A seeded server plus a TestClient, with no port and no process.

    Two tenants with colliding member ids, same as the in-process fixture — so
    an isolation bug shows up as the wrong organisation's data rather than an
    error.
    """
    from fastapi.testclient import TestClient

    from mission_control.api import create_app, encode_token
    from mission_control.demo_seed import seed_platform

    platform = Platform()
    app = create_app(platform, dev_tokens=True, allow_clock_override=True)
    identities = seed_platform(platform)

    class Api:
        def __init__(self) -> None:
            self.app = app
            self.platform = platform
            self.http = TestClient(app)
            self.identities = identities

        def token(self, tenant: str, member_id: str) -> str:
            return encode_token(tenant, member_id)

        def headers(self, tenant: str, member_id: str) -> dict[str, str]:
            return {"Authorization": f"Bearer {self.token(tenant, member_id)}"}

        def client(self, tenant: str = "nasa", member_id: str = "mem-lead"):
            from mission_control.client import MissionControl

            return MissionControl(
                "http://test", self.token(tenant, member_id), http=self.http
            )

    return Api()


@pytest.fixture
def cli_env(api, tmp_path, monkeypatch):
    """The CLI wired to the in-process server, with a throwaway config file."""
    import mission_control.cli as cli_module
    from mission_control.client import MissionControl

    config_path = tmp_path / "mc.json"
    monkeypatch.setattr(cli_module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        cli_module,
        "MissionControl",
        lambda base_url="http://test", token=None, **kw: MissionControl(
            base_url, token, **{**kw, "http": api.http}
        ),
    )
    cli_module.save_config(
        {
            "url": "http://test",
            "current": "nasa/mem-lead",
            "identities": {
                f"{t}/{m}": {
                    "tenant": t, "member_id": m, "name": n,
                    "role": r, "token": api.token(t, m),
                }
                for t, m, n, r in api.identities
            },
        }
    )
    return cli_module
