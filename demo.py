"""End-to-end walkthrough (§12 step 7). Run with ``python demo.py``.

Two tenants, twenty crew, three missions. Covers the happy path —
draft, match, offers, one decline, re-match pinning acceptances, fully crewed,
submit, blocked self-approval, Director approval, activation — plus the negative
paths: submitting under-crewed, and a member deactivated between submit and
approve.

Nothing here reaches inside the package. Every line goes through a Scope
belonging to one authenticated Actor, which is the same surface an API would
use.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from mission_control.domain import (
    CrewProfile,
    CrewSkill,
    Location,
    Member,
    MemberId,
    MissionPlan,
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
    UnavailabilityBlock,
    eligibility_failures,
)
from mission_control.services import Scope
from mission_control.store import Platform

NOW = datetime(2026, 1, 1, tzinfo=UTC)
Pr = Proficiency

PILOT, MED, ENG, GEO = (SkillId(s) for s in ("pilot", "med", "eng", "geo"))
SKILLS = {PILOT: "Piloting", MED: "Medicine", ENG: "Engineering", GEO: "Geology"}

# name, {skill: level}
ROSTER = [
    ("Anya Petrova", {PILOT: Pr.MASTER, MED: Pr.EXPERT}),
    ("Boris Novak", {PILOT: Pr.EXPERT, ENG: Pr.COMPETENT}),
    ("Chen Wei", {PILOT: Pr.NOVICE, GEO: Pr.COMPETENT}),
    ("Dara Okonkwo", {MED: Pr.EXPERT, GEO: Pr.PROFICIENT}),
    ("Eli Ferrand", {MED: Pr.PROFICIENT, ENG: Pr.COMPETENT}),
    ("Fen Zhao", {PILOT: Pr.PROFICIENT, ENG: Pr.EXPERT}),
    ("Gita Rao", {ENG: Pr.MASTER, PILOT: Pr.COMPETENT}),
    ("Hugo Lind", {GEO: Pr.EXPERT, ENG: Pr.PROFICIENT}),
    ("Ines Vidal", {MED: Pr.MASTER}),
    ("Jonas Berg", {PILOT: Pr.PROFICIENT, MED: Pr.COMPETENT}),
    ("Kiran Shah", {ENG: Pr.EXPERT, GEO: Pr.PROFICIENT}),
    ("Lucia Marte", {PILOT: Pr.EXPERT, GEO: Pr.NOVICE}),
    ("Mei Tanaka", {MED: Pr.EXPERT, ENG: Pr.COMPETENT}),
    ("Nils Haugen", {GEO: Pr.MASTER}),
    ("Omar Diallo", {PILOT: Pr.COMPETENT, ENG: Pr.PROFICIENT}),
    ("Priya Nair", {MED: Pr.PROFICIENT, GEO: Pr.EXPERT}),
    ("Quinn Ryder", {ENG: Pr.PROFICIENT}),
    ("Rosa Iglesias", {PILOT: Pr.MASTER, ENG: Pr.COMPETENT}),
    ("Sami Haddad", {GEO: Pr.PROFICIENT, MED: Pr.COMPETENT}),
    ("Tove Aas", {ENG: Pr.EXPERT, MED: Pr.NOVICE}),
]


def rule(title: str = "") -> None:
    print(f"\n{'─' * 78}")
    if title:
        print(title)
        print("─" * 78)


def seed(platform: Platform, tenant_id: TenantId, name: str) -> dict[str, MemberId]:
    """Stand in for signup. A tenant's first Director cannot be created through
    a path that requires a Director, so the fixture installs staff directly."""
    platform.create_tenant(tenant_id, name)
    tenant = platform._unsafe_tenant(tenant_id)
    ids: dict[str, MemberId] = {}

    for key, person, role in [
        ("director", "Dana Whitfield", Role.DIRECTOR),
        ("director_b", "Devi Raman", Role.DIRECTOR),
        ("lead", "Lena Sorokin", Role.MISSION_LEAD),
    ]:
        member_id = MemberId(f"{tenant_id}-{key}")
        tenant.members[member_id] = Member(member_id, person, role)
        ids[key] = member_id

    for skill_id, skill_name in SKILLS.items():
        tenant.skills[skill_id] = Skill(skill_id, skill_name)

    for index, (person, skills) in enumerate(ROSTER):
        member_id = MemberId(f"{tenant_id}-crew-{index:02d}")
        tenant.members[member_id] = Member(member_id, person, Role.CREW)
        tenant.crew_profiles[member_id] = CrewProfile(
            member_id, tuple(CrewSkill(s, lvl) for s, lvl in skills.items())
        )
        ids[person] = member_id

    return ids


def window(start_days: int, length_days: int) -> TimeWindow:
    return TimeWindow(
        NOW + timedelta(days=start_days), NOW + timedelta(days=start_days + length_days)
    )


def requirement(
    label: str, skills: dict[SkillId, Proficiency], *, count: int = 1, mandatory: bool = True
) -> Requirement:
    return Requirement(
        id=RequirementId("r-" + label.lower().replace(" ", "-")),
        label=label,
        count=count,
        skills=tuple(SkillRequirement(s, lvl) for s, lvl in skills.items()),
        mandatory=mandatory,
    )


def show_eligibility(session, mission_id) -> None:
    """Who clears each requirement, before any assignment happens.

    Printed because the greedy-versus-global argument is about *scarcity*, and
    scarcity is only visible in the eligible sets — it is a property of the
    whole matrix, not of any one slot (§7.6).
    """
    mission = session.missions.get(mission_id)
    snap = session.workspace.snapshot()
    print("   eligible candidates per requirement:")
    for req in mission.plan.requirements:
        names = [
            snap.members[member_id].name
            for member_id, profile in snap.crew.items()
            if not eligibility_failures(
                req, snap.members[member_id], profile, mission.plan.window, snap
            )
        ]
        print(f"     {req.label:<16} {len(names)}: {', '.join(sorted(names)) or '—'}")


def show_proposal(proposal) -> None:
    for slot in proposal.slots:
        evidence = ", ".join(
            f"{SKILLS[e.skill_id]} {e.held.name}/{e.required.name}" for e in slot.skills
        )
        pin = "  [pinned]" if slot.pinned else ""
        print(
            f"   {slot.label:<16} {slot.member_name:<16} "
            f"rank {slot.rank}  load {slot.recent_load}  ({evidence}){pin}"
        )
    for unfilled in proposal.unfilled:
        flag = "MANDATORY" if unfilled.mandatory else "optional "
        print(f"   {unfilled.label:<16} —  {flag}: {unfilled.reason}")
    if proposal.near_misses:
        print("   near misses:")
        for near in proposal.near_misses[:4]:
            print(f"     {near.member_name:<16} {near.filter:<13} {near.detail}")
    for warning in proposal.team_warnings:
        print(f"   team constraint unmet: {warning}")


def main() -> None:
    platform = Platform()
    nasa = seed(platform, TenantId("nasa"), "NASA")
    esa = seed(platform, TenantId("esa"), "European Space Agency")
    clock = lambda: NOW  # noqa: E731 — one frozen instant, so the run is reproducible

    def session(ids, tenant_id, key):
        actor = platform.actor_for(TenantId(tenant_id), ids[key])
        return Scope(platform.session(actor), clock)

    lead = session(nasa, "nasa", "lead")
    director = session(nasa, "nasa", "director")
    director_b = session(nasa, "nasa", "director_b")

    def crew(name):
        return session(nasa, "nasa", name)

    # ---------------------------------------------------------------- mission 1
    rule("MISSION 1 — the happy path, with one decline and a re-match")

    artemis = lead.missions.create(
        "Artemis VII",
        MissionPlan(
            requirements=(
                requirement("Pilot", {PILOT: Pr.PROFICIENT}),
                requirement("Flight Surgeon", {MED: Pr.EXPERT}),
                requirement("Engineer", {ENG: Pr.EXPERT}, count=2),
                requirement("Geologist", {GEO: Pr.MASTER}, mandatory=False),
            ),
            window=window(60, 21),
            site=Location("LC-39A"),
        ),
        description="Lunar south pole sortie",
    )
    print(f"created {artemis.id} in state {artemis.state.value}")
    print(f"the Lead can: {', '.join(lead.missions.available_events(artemis.id))}")

    print("\n1. run the matcher (DRAFT only — it creates nothing)")
    proposal = lead.matching.run_matcher(artemis.id)
    show_proposal(proposal)
    print(f"   assignments created by the matcher: "
          f"{len(lead.workspace.assignments_for_mission(artemis.id))}")

    print("\n2. the Lead accepts the proposal, which sends offers")
    offers = lead.matching.offer_proposal(artemis.id, proposal)
    for offer in offers:
        print(f"   offered {offer.requirement_id}#{offer.slot_index} to "
              f"{lead.workspace.member(offer.member_id).name}, "
              f"expires {offer.offer_expires_at:%Y-%m-%d %H:%M}")
    print(f"   crewing status: {lead.missions.crewing_status(artemis.id).value}")

    print("\n3. submitting now is refused — crewing completes before approval")
    for event, why in lead.missions.blocked_events(artemis.id).items():
        print(f"   {event}: {why}")

    print("\n4. crew respond. Everyone accepts except the Flight Surgeon")
    surgeon_offer = next(o for o in offers if o.requirement_id == "r-flight-surgeon")
    for offer in offers:
        person = lead.workspace.member(offer.member_id).name
        service = crew(person).assignments
        if offer.id == surgeon_offer.id:
            service.decline(offer.id)
            print(f"   {person:<16} DECLINED")
        else:
            service.accept(offer.id)
            print(f"   {person:<16} accepted")
    print(f"   crewing status: {lead.missions.crewing_status(artemis.id).value}")

    print("\n5. re-run the matcher. Acceptances are pinned to the same slots;")
    print("   only the vacant slot is re-offered, and the decliner is not re-asked")
    second = lead.matching.run_matcher(artemis.id)
    show_proposal(second)
    refill = lead.matching.offer_proposal(artemis.id, second)
    print(f"   new offers: {[o.requirement_id for o in refill]}")
    for offer in refill:
        crew(lead.workspace.member(offer.member_id).name).assignments.accept(offer.id)
    print(f"   crewing status: {lead.missions.crewing_status(artemis.id).value}")

    print("\n6. submit for approval")
    artemis = lead.missions.submit(artemis.id)
    print(f"   state {artemis.state.value}, submitted by "
          f"{lead.workspace.member(artemis.submitted_by).name}, "
          f"deadline {artemis.pending_expires_at:%Y-%m-%d}")

    print("\n7. the Lead cannot approve their own mission")
    try:
        lead.missions.approve(artemis.id)
    except Exception as error:
        print(f"   refused: {type(error).__name__}: {error}")
    print("   a Director holds no authoring permissions, so the case cannot even arise:")
    try:
        director.missions.create("Shadow mission", artemis.plan)
    except Exception as error:
        print(f"   refused: {type(error).__name__}: {error}")

    print("\n8. a Director approves, and the Lead activates")
    artemis = director.missions.approve(artemis.id)
    print(f"   state {artemis.state.value}, approved by "
          f"{director.workspace.member(artemis.approved_by).name} "
          f"against plan v{artemis.approved_plan_version}")
    artemis = lead.missions.activate(artemis.id, now=artemis.plan.window.start)
    print(f"   state {artemis.state.value}")

    # ---------------------------------------------------------------- mission 2
    rule("MISSION 2 — global assignment beats per-slot greedy (§7.6)")

    print("Europa's window overlaps Artemis, so everyone crewed above is booked.")
    print("That leaves two Master pilots free, Anya and Rosa — and only Anya")
    print("also holds Medicine, so she is the *only* candidate for Surgeon-Pilot.")
    print("Filling Command Pilot first would take her and strand the other slot.")

    europa = lead.missions.create(
        "Europa Survey",
        MissionPlan(
            requirements=(
                requirement("Command Pilot", {PILOT: Pr.MASTER}),
                requirement("Surgeon-Pilot", {PILOT: Pr.MASTER, MED: Pr.EXPERT}),
            ),
            window=window(65, 10),
            site=Location("SLC-41"),
        ),
    )
    show_eligibility(lead, europa.id)
    print("\n   the solver's answer:")
    show_proposal(lead.matching.run_matcher(europa.id))
    print("\n   Rosa takes Command Pilot even though Anya outranks her for it —")
    print("   the solver pays rank 1 on one slot to avoid an empty slot on the")
    print("   other. Greedy, taking each slot's first choice, fills only one.")

    print("\nAnd a requirement nobody clears is reported, not raised:")
    impossible = lead.missions.create(
        "Titan Deep Drill",
        MissionPlan(
            requirements=(requirement("Cryogenicist", {GEO: Pr.MASTER, MED: Pr.MASTER}),),
            window=window(200, 30),
            site=Location("Pad 0A"),
        ),
    )
    show_proposal(lead.matching.run_matcher(impossible.id))

    # ---------------------------------------------------------------- mission 3
    rule("MISSION 3 — the negative path: a member lost between submit and approve")

    ceres = lead.missions.create(
        "Ceres Transit",
        MissionPlan(
            requirements=(requirement("Pilot", {PILOT: Pr.EXPERT}),),
            window=window(300, 14),
            site=Location("LC-39B"),
        ),
    )
    proposal = lead.matching.run_matcher(ceres.id)
    offers = lead.matching.offer_proposal(ceres.id, proposal)
    booked = offers[0].member_id
    crew(lead.workspace.member(booked).name).assignments.accept(offers[0].id)
    ceres = lead.missions.submit(ceres.id)
    print(f"submitted with {lead.workspace.member(booked).name} aboard "
          f"({ceres.state.value})")

    print("\nthe crew member declares a clashing holiday — refused, because they")
    print("have already accepted and acceptance is final in this design")
    try:
        crew(lead.workspace.member(booked).name).crew.set_unavailability(
            booked, [UnavailabilityBlock(window(302, 5), "holiday")]
        )
    except Exception as error:
        print(f"   refused: {type(error).__name__}: {error}")

    print("\nthe Director deactivates them instead, then tries to approve")
    director.org.deactivate_member(booked, "reassigned to another programme")
    ceres = director_b.missions.approve(ceres.id)
    print(f"   the gate bounced it: state is now {ceres.state.value}")
    print(f"   crewing status: {lead.missions.crewing_status(ceres.id).value}")
    print(f"   the Lead was notified: "
          f"{[n.subject for n in lead.workspace.notifications(nasa['lead'])][-1]!r}")

    print("\nre-crew and re-submit — the same DRAFT state every other")
    print("backwards transition goes to")
    proposal = lead.matching.run_matcher(ceres.id)
    offers = lead.matching.offer_proposal(ceres.id, proposal)
    for offer in offers:
        crew(lead.workspace.member(offer.member_id).name).assignments.accept(offer.id)
    ceres = lead.missions.submit(ceres.id)
    ceres = director.missions.approve(ceres.id)
    print(f"   {ceres.state.value}, approved by "
          f"{director.workspace.member(ceres.approved_by).name}")

    # ------------------------------------------------------------------- tenancy
    rule("TENANT ISOLATION")

    esa_lead = session(esa, "esa", "lead")
    print(f"NASA has {len(lead.missions.list())} missions; "
          f"ESA has {len(esa_lead.missions.list())}")
    try:
        esa_lead.missions.get(artemis.id)
    except Exception as error:
        print(f"ESA reading a NASA mission id: {type(error).__name__}: {error}")
    print("NotFound, not Forbidden — a 403 would confirm the record exists")

    # -------------------------------------------------------------------- audit
    rule("AUDIT — the event log is the truth; state is a projection of it")

    for event in director.org.audit():
        if event.subject_id != artemis.id:
            continue
        actor = director.workspace.member(event.actor_id).name
        arrow = f"{event.from_state or '—'} → {event.to_state}"
        print(f"   {event.event:<16} {arrow:<30} by {actor} ({event.actor_role.value})")

    rule("VISIBILITY — what a crew member sees")
    boris = crew("Boris Novak")
    for mission in boris.missions.list():
        print(f"   {mission.title} ({mission.state.value})")
    print("   he sees only missions he is assigned to, and nothing else "
          "in the organisation")


if __name__ == "__main__":
    main()
