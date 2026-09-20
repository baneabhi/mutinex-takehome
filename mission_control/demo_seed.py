"""Seed data for ``mc serve --seed``, and the scripted run behind ``mc demo``.

The roster is shared with :mod:`demo` so there is one cast of characters.

Member ids are **fixed, and identical across both tenants**. That is deliberate
twice over: tokens stay valid across a server restart because they are derived
from ids rather than issued, and the colliding ids exercise the thing the whole
design is about — ``nasa/mem-lead`` and ``esa/mem-lead`` are different people,
and anything that resolves one without the tenant is a bug.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

SKILLS = {
    "pilot": "Piloting",
    "med": "Medicine",
    "eng": "Engineering",
    "geo": "Geology",
}

STAFF = [
    ("mem-director", "Dana Whitfield", "director"),
    ("mem-director-b", "Devi Raman", "director"),
    ("mem-lead", "Lena Sorokin", "mission_lead"),
    ("mem-lead-b", "Liam Foss", "mission_lead"),
]

ROSTER = [
    ("Anya Petrova", {"pilot": "MASTER", "med": "EXPERT"}),
    ("Boris Novak", {"pilot": "EXPERT", "eng": "COMPETENT"}),
    ("Chen Wei", {"pilot": "NOVICE", "geo": "COMPETENT"}),
    ("Dara Okonkwo", {"med": "EXPERT", "geo": "PROFICIENT"}),
    ("Eli Ferrand", {"med": "PROFICIENT", "eng": "COMPETENT"}),
    ("Fen Zhao", {"pilot": "PROFICIENT", "eng": "EXPERT"}),
    ("Gita Rao", {"eng": "MASTER", "pilot": "COMPETENT"}),
    ("Hugo Lind", {"geo": "EXPERT", "eng": "PROFICIENT"}),
    ("Ines Vidal", {"med": "MASTER"}),
    ("Jonas Berg", {"pilot": "PROFICIENT", "med": "COMPETENT"}),
    ("Kiran Shah", {"eng": "EXPERT", "geo": "PROFICIENT"}),
    ("Lucia Marte", {"pilot": "EXPERT", "geo": "NOVICE"}),
    ("Mei Tanaka", {"med": "EXPERT", "eng": "COMPETENT"}),
    ("Nils Haugen", {"geo": "MASTER"}),
    ("Omar Diallo", {"pilot": "COMPETENT", "eng": "PROFICIENT"}),
    ("Priya Nair", {"med": "PROFICIENT", "geo": "EXPERT"}),
    ("Quinn Ryder", {"eng": "PROFICIENT"}),
    ("Rosa Iglesias", {"pilot": "MASTER", "eng": "COMPETENT"}),
    ("Sami Haddad", {"geo": "PROFICIENT", "med": "COMPETENT"}),
    ("Tove Aas", {"eng": "EXPERT", "med": "NOVICE"}),
]

TENANTS = [("nasa", "NASA"), ("esa", "European Space Agency")]


def seed_platform(platform) -> list[tuple[str, str, str, str]]:
    """Install two tenants directly. Returns ``(tenant, member_id, name, role)``.

    Reaches past the service layer on purpose: a tenant's first Director cannot
    be created through a path that requires a Director, and the fixture needs
    *fixed* ids that the normal path would not give.
    """
    from .domain import (
        CrewProfile,
        CrewSkill,
        Member,
        MemberId,
        Proficiency,
        Role,
        Skill,
        SkillId,
        TenantId,
    )

    identities: list[tuple[str, str, str, str]] = []
    for tenant_id, tenant_name in TENANTS:
        platform.create_tenant(TenantId(tenant_id), tenant_name)
        tenant = platform._unsafe_tenant(TenantId(tenant_id))

        for skill_id, skill_name in SKILLS.items():
            tenant.skills[SkillId(skill_id)] = Skill(SkillId(skill_id), skill_name)

        for member_id, name, role in STAFF:
            tenant.members[MemberId(member_id)] = Member(
                MemberId(member_id), name, Role(role)
            )
            identities.append((tenant_id, member_id, name, role))

        for index, (name, skills) in enumerate(ROSTER):
            member_id = MemberId(f"mem-crew-{index:02d}")
            tenant.members[member_id] = Member(member_id, name, Role.CREW)
            tenant.crew_profiles[member_id] = CrewProfile(
                member_id,
                tuple(
                    CrewSkill(SkillId(s), Proficiency[lvl]) for s, lvl in skills.items()
                ),
            )
            identities.append((tenant_id, member_id, name, "crew"))

    return identities


def run_demo(url: str) -> int:
    """The §12 scenario, driven over HTTP by the typed client.

    Creates its own tenant so it never disturbs seeded data, and every line
    goes through the public API — which is the point: it proves the HTTP
    surface is complete enough to run the whole workflow.
    """
    from .client import ApiError, MissionControl, make_plan, requirement

    stamp = int(time.time())
    tenant = f"demo-{stamp}"
    start = datetime.now(timezone.utc) + timedelta(days=60)
    end = start + timedelta(days=21)

    anon = MissionControl(url)
    try:
        anon.health()
    except OSError:
        print(f"cannot reach {url} — is 'mc serve' running?")
        return 3

    director = anon.as_actor(anon.bootstrap(tenant, "Demo Agency", "Dana Whitfield").token)
    for skill_id, name in SKILLS.items():
        director.add_skill(name, skill_id)

    second_director = director.as_actor(
        director.token_for(director.add_member("Devi Raman", "director").id).token
    )
    lead = director.as_actor(
        director.token_for(director.add_member("Lena Sorokin", "mission_lead").id).token
    )

    crew: dict[str, MissionControl] = {}
    for name, skills in ROSTER:
        member = director.add_member(name, "crew")
        director.set_skills(member.id, skills)
        crew[name] = director.as_actor(director.token_for(member.id).token)

    def rule(title: str) -> None:
        print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")

    rule(f"tenant {tenant} — crewing a mission end to end over HTTP")

    plan = make_plan(
        [
            requirement("Pilot", {"pilot": "PROFICIENT"}, id="r-pilot"),
            requirement("Flight Surgeon", {"med": "EXPERT"}, id="r-surgeon"),
            requirement("Engineer", {"eng": "EXPERT"}, count=2, id="r-engineer"),
            requirement("Geologist", {"geo": "MASTER"}, mandatory=False, id="r-geo"),
        ],
        start=start, end=end, site="LC-39A",
    )
    mission = lead.create_mission("Artemis VII", plan)
    print(f"created {mission.id} ({mission.state}); lead can: "
          f"{', '.join(mission.available_events)}")

    proposal = lead.run_matcher(mission.id)
    for slot in proposal.slots:
        print(f"  {slot.label:16} {slot.member_name:16} rank {slot.rank}  load {slot.recent_load}")
    for unfilled in proposal.unfilled:
        print(f"  {unfilled.label:16} — {unfilled.reason}")

    offers = lead.offer_proposal(mission.id, proposal)
    print(f"\noffered {len(offers)} slots; nothing was created by the matcher itself")

    rule("one declines, the re-match pins the rest")
    declined, *accepted = offers
    crew[declined.member_name].decline(declined.id)
    for offer in accepted:
        crew[offer.member_name].accept(offer.id)
    print(f"{declined.member_name} declined; {len(accepted)} accepted")

    second = lead.run_matcher(mission.id)
    for slot in second.slots:
        print(f"  {slot.label:16} {slot.member_name:16} "
              f"{'[pinned]' if slot.pinned else '[new]'}")
    for offer in lead.offer_proposal(mission.id, second):
        crew[offer.member_name].accept(offer.id)
    print(f"crewing: {lead.mission(mission.id).crewing_status}")

    rule("submit, self-approval refused, director approves")
    print(f"submit → {lead.submit(mission.id).state}")
    try:
        lead.approve(mission.id)
    except ApiError as error:
        print(f"lead approves → {error.status} {error.code}")
    try:
        director.create_mission("Shadow", plan)
    except ApiError as error:
        print(f"director authors → {error.status} {error.code}")
    print(f"director approves → {second_director.approve(mission.id).state}")

    rule("the plan is frozen once approved")
    try:
        lead.update_plan(mission.id, make_plan(
            [requirement("Pilot", {"pilot": "MASTER"}, id="r-pilot")],
            start=start, end=end, site="LC-39A"))
    except ApiError as error:
        print(f"edit → {error.status} {error.code}; you can: {', '.join(error.available)}")

    rule("what a crew member sees")
    anya = crew["Anya Petrova"]
    for m in anya.missions():
        print(f"  {m.title} ({m.state})")
    view = anya.mission(mission.id)
    print(f"  fields visible to crew: {', '.join(sorted(view.model_dump()))}")

    rule("tenant isolation")
    try:
        other = anon.as_actor(anon.bootstrap(f"{tenant}-other", "Other", "Someone").token)
        other.mission(mission.id)
    except ApiError as error:
        print(f"another tenant reading {mission.id} → {error.status} {error.code}")

    print(f"\nDone. Explore with:  mc --profile {tenant}/mem-director  &&  mc mission list")
    return 0
