"""The mission state machine (§6, §10)."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import timedelta

import pytest

from mission_control.domain import (
    AssignmentState,
    CrewingStatus,
    Denied,
    GuardFailed,
    IllegalTransition,
    Location,
    MissionPlan,
    MissionState,
    Proficiency,
    Role,
    TimeWindow,
)
from mission_control.lifecycle import MISSION_TRANSITIONS

from .conftest import MED, PILOT, T0, plan, req

MS = MissionState

ALL_EVENTS = sorted({event for _, event in MISSION_TRANSITIONS})
SYSTEM_EVENTS = {
    event
    for (_, event), transition in MISSION_TRANSITIONS.items()
    if transition.permission is None
}


# --------------------------------------------------------- the exhaustive sweep


def _attempt(world, session, mission, event):
    """Fire ``event`` the way a caller would, supplying whatever it needs."""
    if event == "plan_edited":
        edited = replace(mission.plan, site=Location("SLC-41"))
        return session.missions.update_plan(mission.id, edited)
    if event == "run_matcher":
        session.matching.run_matcher(mission.id)
        return session.missions.get(mission.id)
    if event == "approve":
        return session.missions.approve(mission.id)
    if event == "activate":
        return session.missions.activate(mission.id, now=mission.plan.window.start)
    if event == "complete":
        return session.missions.complete(mission.id, now=mission.plan.window.end)
    return session.missions.fire(mission.id, event, reason="a stated reason")


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("event", ALL_EVENTS)
@pytest.mark.parametrize("state", list(MissionState))
def test_every_state_event_role_triple(world, state, event, role):
    """Every ``(state, event, role)`` triple either transitions per the table or
    raises. **Nothing silently no-ops** (§10).

    This is the test the transition table exists for. Guards against the failure
    mode where a handler quietly returns the unchanged aggregate and the caller
    believes it worked.
    """
    mission = world.mission_in(state)
    session = {
        Role.DIRECTOR: world.director(),
        Role.MISSION_LEAD: world.lead(),
        Role.CREW: world.crew(1),
    }[role]

    legal = (state, event) in MISSION_TRANSITIONS

    try:
        result = _attempt(world, session, mission, event)
    except IllegalTransition:
        assert not legal, f"{role.value} was refused a legal {state.value} -> {event}"
        return
    except (Denied, GuardFailed):
        # Refused on permission, policy or a domain precondition. That is a
        # legitimate outcome for a legal transition, and also for an illegal one
        # — MissionService.approve checks authorisation *before* the state, so a
        # Crew member asking to approve a DRAFT mission learns nothing about it.
        # What matters either way is that nothing moved.
        assert world.lead().missions.get(mission.id).state is state
        return

    assert legal, f"{state.value} -> {event} is not in the table but succeeded"
    expected = MISSION_TRANSITIONS[(state, event)].target
    if event == "approve" and result.state is MS.DRAFT:
        return  # the §6.8 gate bounced it, which is its documented behaviour
    assert result.state is expected


@pytest.mark.parametrize("event", sorted(SYSTEM_EVENTS))
def test_system_events_are_the_only_ones_without_a_permission(event):
    """``expire`` and ``invalidated`` are the only two. Every other mission
    transition is somebody's decision (§6.2)."""
    assert event in {"expire", "invalidated"}


def test_the_table_and_the_documented_state_set_agree():
    sources = {state for state, _ in MISSION_TRANSITIONS}
    targets = {t.target for t in MISSION_TRANSITIONS.values()}
    assert sources == {MS.DRAFT, MS.PENDING_APPROVAL, MS.APPROVED, MS.ACTIVE}, (
        "terminal states must have no outgoing transitions"
    )
    assert targets <= set(MissionState)
    assert MS.COMPLETED in targets and MS.CANCELLED in targets and MS.ABORTED in targets


# ----------------------------------------------------- the one editable state


PLAN_MUTATIONS = {
    "requirements": lambda p: replace(
        p, requirements=(req("Pilot", {PILOT: Proficiency.MASTER}),)
    ),
    "window": lambda p: replace(
        p, window=TimeWindow(p.window.start + timedelta(days=1), p.window.end)
    ),
    "site": lambda p: replace(p, site=Location("SLC-41")),
}


def test_the_mutation_table_covers_every_plan_field():
    """Enumerated from MissionPlan's fields, so a field added later is covered
    by default rather than silently untested (§10)."""
    plan_fields = {f.name for f in fields(MissionPlan)} - {"version"}
    assert plan_fields == set(PLAN_MUTATIONS)


@pytest.mark.parametrize("field_name", sorted(PLAN_MUTATIONS))
def test_the_plan_is_editable_in_draft(world, field_name):
    mission = world.mission_in(MS.DRAFT)
    edited = PLAN_MUTATIONS[field_name](mission.plan)

    result = world.lead().missions.update_plan(mission.id, edited)

    assert result.state is MS.DRAFT
    assert result.plan.version == mission.plan.version + 1


@pytest.mark.parametrize("field_name", sorted(PLAN_MUTATIONS))
@pytest.mark.parametrize("state", [MS.PENDING_APPROVAL, MS.APPROVED, MS.ACTIVE])
def test_the_plan_is_frozen_everywhere_else(world, state, field_name):
    """The invariant (§6.1.1): **the plan is editable in DRAFT and nowhere
    else.**

    Enforced by the absence of a row in the transition table rather than by a
    guard someone could forget at a new call site — so the refusal names the way
    forward for free.
    """
    mission = world.mission_in(state)
    edited = PLAN_MUTATIONS[field_name](mission.plan)

    with pytest.raises(IllegalTransition) as caught:
        world.lead().missions.update_plan(mission.id, edited)

    assert "plan_edited" in str(caught.value)
    assert world.lead().missions.get(mission.id).plan == mission.plan
    if state is not MS.ACTIVE:
        assert "reopen" in caught.value.available


@pytest.mark.parametrize("field_name", sorted(PLAN_MUTATIONS))
@pytest.mark.parametrize("state", [MS.PENDING_APPROVAL, MS.APPROVED])
def test_reopening_is_what_makes_a_plan_editable_again(world, state, field_name):
    """Two steps rather than one, and deliberately: pulling a mission back from
    review is a decision, not a side effect of typing."""
    mission = world.mission_in(state)
    lead = world.lead()

    reopened = lead.missions.reopen(mission.id)
    assert reopened.state is MS.DRAFT

    result = lead.missions.update_plan(
        mission.id, PLAN_MUTATIONS[field_name](mission.plan)
    )
    assert result.state is MS.DRAFT
    assert result.plan.version == mission.plan.version + 1


def test_a_plan_under_review_cannot_change_under_the_director(world):
    """The property freezing PENDING_APPROVAL buys, and the reason for the two
    steps.

    Previously an edit and an approval racing meant the *edit* won: the mission
    dropped to DRAFT and the Director's click failed. Now the approval wins and
    the edit is refused, which is the right way round — approval is the act with
    more ceremony behind it, and a Director should never be reading a plan that
    can move.
    """
    mission = world.mission_in(MS.PENDING_APPROVAL)
    lead, director = world.lead(), world.director()

    with pytest.raises(IllegalTransition):
        lead.missions.update_plan(
            mission.id, replace(mission.plan, site=Location("SLC-41"))
        )

    approved = director.missions.approve(mission.id)
    assert approved.state is MS.APPROVED
    assert approved.approved_plan_version == mission.plan.version

    # The Lead is not stuck — reopening is still theirs to do, it just costs
    # the approval and says so.
    assert lead.missions.reopen(mission.id).approved_by is None


def test_the_plan_can_only_be_changed_from_one_place(world):
    """Structural consequence of the change: the effects that install a plan and
    release crew appear on exactly one row, so there is no second copy to drift."""
    installing = [
        (state, event)
        for (state, event), transition in MISSION_TRANSITIONS.items()
        if any(e.__name__ == "install_new_plan" for e in transition.effects)
    ]
    assert installing == [(MS.DRAFT, "plan_edited")]


@pytest.mark.parametrize("state", [MS.PENDING_APPROVAL, MS.APPROVED])
def test_a_stale_approve_click_fails_naming_the_current_state(world, state):
    mission = world.mission_in(state)
    world.lead().missions.reopen(mission.id)

    with pytest.raises(IllegalTransition) as caught:
        world.director().missions.approve(mission.id)
    assert "cannot approve a draft mission" in str(caught.value)
    # The options offered are the ones *this actor* has, which for a Director on
    # a DRAFT mission is governance only — they cannot submit it themselves.
    assert caught.value.available == ("cancel",)
    assert "submit" in world.lead().missions.available_events(mission.id)


def test_reopening_discards_the_approval(world):
    """A Director authorised a specific plan; reopening is the Lead announcing
    that is about to stop being the plan (§6.1.1)."""
    mission = world.mission_in(MS.APPROVED)
    assert mission.approved_by is not None

    result = world.lead().missions.reopen(mission.id)

    assert result.state is MS.DRAFT
    assert result.approved_by is None
    assert result.approved_plan_version is None
    assert result.submitted_by is None

    lead_notes = [n.subject for n in world.notifications(mission.created_by)]
    assert any("approval has been discarded" in s for s in lead_notes)


@pytest.mark.parametrize(
    "field_name,value",
    [("title", "Artemis VIII"), ("description", "revised"), ("tags", ("crewed",)),
     ("notes", "internal"), ("reference_code", "AR-7")],
)
@pytest.mark.parametrize("state", [MS.DRAFT, MS.PENDING_APPROVAL, MS.APPROVED, MS.ACTIVE])
def test_metadata_edits_do_not_revert_the_mission(world, state, field_name, value):
    """The complement, without which the safe direction is untested (§10).

    Read literally the invariant would un-approve a mission for a typo fix,
    which trains people not to document anything.
    """
    mission = world.mission_in(state)
    before = world.accepted(mission.id)

    result = world.lead().missions.update_metadata(mission.id, **{field_name: value})

    assert result.state is state
    assert getattr(result, field_name) == value
    assert result.plan.version == mission.plan.version
    assert result.approved_by == mission.approved_by
    assert world.accepted(mission.id) == before


def test_metadata_fields_match_the_mission_wrapper():
    """A field added to Mission is metadata by default; one added to MissionPlan
    is plan. This pins that split so it cannot drift."""
    from mission_control.domain import METADATA_FIELDS, Mission

    mission_fields = {f.name for f in fields(Mission)}
    assert METADATA_FIELDS <= mission_fields
    assert not (METADATA_FIELDS & {f.name for f in fields(MissionPlan)})


def test_update_metadata_refuses_plan_fields(world):
    mission = world.mission_in(MS.APPROVED)
    with pytest.raises(ValueError, match="use update_plan"):
        world.lead().missions.update_metadata(mission.id, requirements=())


# ------------------------------------------------------------------ submit gates


def test_a_mission_cannot_be_submitted_under_crewed(world):
    """``submit`` asks "is this staffed?", not "does this look staffable?"
    (§6.6)."""
    mission = world.draft(req("Pilot", {PILOT: Proficiency.PROFICIENT}))
    lead = world.lead()

    proposal = lead.matching.run_matcher(mission.id)
    offers = lead.matching.offer_proposal(mission.id, proposal)
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.AWAITING_RESPONSES

    with pytest.raises(GuardFailed, match="awaiting_responses"):
        lead.missions.submit(mission.id)

    world.session(offers[0].member_id).assignments.accept(offers[0].id)
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED
    assert lead.missions.submit(mission.id).state is MS.PENDING_APPROVAL


def test_an_optional_slot_does_not_block_submission(world):
    mission = world.draft(
        req("Pilot", {PILOT: Proficiency.PROFICIENT}),
        req("Observer", {MED: Proficiency.MASTER}, mandatory=False),
    )
    world.crew_up(mission.id)
    assert world.lead().missions.submit(mission.id).state is MS.PENDING_APPROVAL


def test_a_mission_with_no_requirements_can_be_drafted_but_not_submitted(world):
    """A plan under construction is allowed to be empty — that is what DRAFT is
    for. The ``has_requirements`` guard is on ``submit``, not on the type."""
    empty = MissionPlan(requirements=(), window=plan().window, site=Location("LC-39A"))
    mission = world.lead().missions.create("Sketch", empty)
    assert mission.state is MS.DRAFT

    with pytest.raises(GuardFailed, match="no requirements"):
        world.lead().missions.submit(mission.id)


def test_a_mission_in_the_past_cannot_be_submitted(world):
    mission = world.draft(start_days=30)
    world.crew_up(mission.id)
    with pytest.raises(GuardFailed, match="does not start in the future"):
        world.lead().missions.submit(mission.id, now=T0 + timedelta(days=31))


def test_cancelling_and_aborting_need_a_reason(world):
    draft = world.draft()
    with pytest.raises(GuardFailed, match="reason is required"):
        world.lead().missions.fire(draft.id, "cancel")

    active = world.mission_in(MS.ACTIVE)
    with pytest.raises(GuardFailed, match="reason is required"):
        world.lead().missions.fire(active.id, "abort")


def test_completing_early_needs_a_reason(world):
    mission = world.mission_in(MS.ACTIVE)
    midway = mission.plan.window.start + timedelta(days=1)

    with pytest.raises(GuardFailed, match="completing early needs a reason"):
        world.lead().missions.complete(mission.id, now=midway)

    result = world.lead().missions.complete(mission.id, reason="objectives met", now=midway)
    assert result.state is MS.COMPLETED


# ------------------------------------------------------------- the §6.8 gate


def test_a_member_deactivated_between_submit_and_approve_bounces_the_mission(world):
    """The negative path §12's demo calls out, and the reason the gate exists."""
    mission = world.mission_in(MS.PENDING_APPROVAL)
    crew_member = world.accepted(mission.id)[0].member_id

    world.director().org.deactivate_member(crew_member, "medical grounding")

    result = world.director().missions.approve(mission.id)
    assert result.state is MS.DRAFT
    assert result.submitted_by is None
    assert result.approved_by is None

    lead_notes = [n.subject for n in world.notifications(mission.created_by)]
    assert any("returned to draft" in s for s in lead_notes)


def test_skills_revised_down_between_submit_and_approve_bounces_the_mission(world):
    """The other of the two remaining triggers (§6.8): they are still ACCEPTED,
    but no longer eligible."""
    from mission_control.domain import CrewSkill

    mission = world.mission_in(MS.PENDING_APPROVAL, req("Pilot", {PILOT: Proficiency.EXPERT}))
    crew_member = world.accepted(mission.id)[0].member_id

    world.director().crew.set_skills(crew_member, [CrewSkill(PILOT, Proficiency.NOVICE)])

    result = world.director().missions.approve(mission.id)
    assert result.state is MS.DRAFT

    released = [
        a for a in world.lead().workspace.assignments_for_mission(mission.id)
        if a.member_id == crew_member
    ]
    assert released[0].state is AssignmentState.RELEASED


def test_a_stale_optional_slot_does_not_bounce_the_mission(world):
    """Blocking a Director's judgement about whether a mission should happen over
    an optional slot enforces crewing completeness in the wrong place (§6.8)."""
    from mission_control.domain import CrewSkill

    mission = world.mission_in(
        MS.PENDING_APPROVAL,
        req("Pilot", {PILOT: Proficiency.PROFICIENT}),
        req("Surgeon", {MED: Proficiency.PROFICIENT}, mandatory=False),
    )
    optional = [
        a for a in world.accepted(mission.id) if a.requirement_id == "r-surgeon"
    ][0]
    world.director().crew.set_skills(optional.member_id, [CrewSkill(MED, Proficiency.NOVICE)])

    result = world.director().missions.approve(mission.id)
    assert result.state is MS.APPROVED

    report = world.director().missions.allocation_report(mission.id)
    assert len(report.stale) == 1 and not report.blocking


def test_the_gate_evaluates_against_the_mission_window_not_now(world):
    """A mission approved today for a window six months out must be checked for
    conflicts *in that window* (§6.8)."""
    from mission_control.domain import UnavailabilityBlock

    mission = world.mission_in(MS.PENDING_APPROVAL, start_days=180)
    crew_member = world.accepted(mission.id)[0].member_id

    # A block that is in the past relative to the mission, but the future
    # relative to now. It must not invalidate anything.
    near_block = UnavailabilityBlock(
        TimeWindow(T0 + timedelta(days=5), T0 + timedelta(days=10)), "training"
    )
    world.session(crew_member).crew.set_unavailability(crew_member, [near_block])

    assert world.director().missions.approve(mission.id).state is MS.APPROVED


def test_activation_fails_rather_than_bouncing(world):
    """``approve`` auto-reverts; ``activate`` does not. The Lead has exactly two
    moves — cancel, or edit the plan to return to DRAFT and re-crew (§6.8)."""
    from mission_control.domain import CrewSkill

    mission = world.mission_in(MS.APPROVED, req("Pilot", {PILOT: Proficiency.EXPERT}))
    crew_member = world.accepted(mission.id)[0].member_id
    world.director().crew.set_skills(crew_member, [CrewSkill(PILOT, Proficiency.NOVICE)])

    lead = world.lead()
    with pytest.raises(GuardFailed):
        lead.missions.activate(mission.id, now=mission.plan.window.start)

    assert lead.missions.get(mission.id).state is MS.APPROVED
    available = lead.missions.available_events(mission.id)
    assert "cancel" in available and "reopen" in available


def test_nobody_can_waive_a_missing_mandatory_slot(world):
    mission = world.mission_in(MS.APPROVED)
    accepted = world.accepted(mission.id)[0]
    world.director().org.deactivate_member(accepted.member_id, "left the programme")

    for session in (world.lead(), world.director()):
        with pytest.raises((GuardFailed, Denied)):
            session.missions.activate(mission.id, now=mission.plan.window.start)


# ----------------------------------------------- reopening, and what it costs


@pytest.mark.parametrize("state", [MS.PENDING_APPROVAL, MS.APPROVED])
def test_reopening_keeps_the_crew(world, state):
    """Reopening is not itself a plan change, so nobody is released.

    A Lead who reopens to correct a requirement label or a misspelled site
    should not cost five people their booking and make them answer the same
    question twice. What releases is the *edit*, and only if it changes what the
    matcher consults (§6.7).
    """
    mission = world.mission_in(state)
    booked = [a.member_id for a in world.accepted(mission.id)]
    assert booked

    result = world.lead().missions.reopen(mission.id)

    assert result.state is MS.DRAFT and result.submitted_by is None
    assert [a.member_id for a in world.accepted(mission.id)] == booked


@pytest.mark.parametrize("state", [MS.PENDING_APPROVAL, MS.APPROVED])
def test_a_cosmetic_plan_edit_keeps_the_crew(world, state):
    """The case reopening exists to protect: fixing a typo inside the plan.

    ``Requirement.label`` and the site are plan fields, so correcting them needs
    a reopen — but neither is anything the matcher consults, so the roster
    stands.
    """
    mission = world.mission_in(state, req("Flight Sugeon", {MED: Proficiency.EXPERT}))
    booked = [a.member_id for a in world.accepted(mission.id)]
    lead = world.lead()
    lead.missions.reopen(mission.id)

    corrected = replace(
        mission.plan,
        requirements=(
            replace(mission.plan.requirements[0], label="Flight Surgeon"),
        ),
        site=Location("LC-39A (corrected)"),
    )
    result = lead.missions.update_plan(mission.id, corrected)

    assert result.plan.requirements[0].label == "Flight Surgeon"
    assert result.plan.version == mission.plan.version + 1
    assert [a.member_id for a in world.accepted(mission.id)] == booked
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: replace(p, requirements=(
            req("Pilot", {PILOT: Proficiency.MASTER}),)), id="skills"),
        pytest.param(lambda p: replace(p, requirements=(
            replace(p.requirements[0], count=2),)), id="count"),
        pytest.param(lambda p: replace(p, requirements=(
            replace(p.requirements[0], mandatory=False),)), id="mandatory"),
        pytest.param(lambda p: replace(p, window=TimeWindow(
            p.window.start + timedelta(days=40), p.window.end + timedelta(days=40))),
            id="window"),
    ],
)
def test_a_material_plan_edit_releases_the_crew(world, mutate):
    """The complement: anything the matcher consults invalidates the roster,
    because those people agreed to a different specification."""
    mission = world.mission_in(MS.APPROVED, req("Pilot", {PILOT: Proficiency.PROFICIENT}))
    lead = world.lead()
    lead.missions.reopen(mission.id)

    lead.missions.update_plan(mission.id, mutate(mission.plan))

    assert world.accepted(mission.id) == []
    assert all(
        a.state is AssignmentState.RELEASED
        for a in lead.workspace.assignments_for_mission(mission.id)
    )


def test_the_matching_signature_ignores_presentation(world):
    """Pins the rule itself, so a field added to Requirement has to decide
    explicitly rather than silently becoming material."""
    base = plan(req("Pilot", {PILOT: Proficiency.PROFICIENT}))

    cosmetic = replace(base, site=Location("Elsewhere"),
                       requirements=(replace(base.requirements[0], label="Aviator"),))
    assert cosmetic.matching_signature() == base.matching_signature()

    reordered = plan(req("A", {PILOT: Proficiency.PROFICIENT}, rid="r-a"),
                     req("B", {MED: Proficiency.PROFICIENT}, rid="r-b"))
    swapped = replace(reordered, requirements=tuple(reversed(reordered.requirements)))
    assert swapped.matching_signature() == reordered.matching_signature()

    material = replace(base, requirements=(
        replace(base.requirements[0], count=3),))
    assert material.matching_signature() != base.matching_signature()


@pytest.mark.parametrize("state", [MS.PENDING_APPROVAL, MS.APPROVED])
def test_reopen_recrew_resubmit_is_the_whole_loop(world, state):
    """The DRAFT -> PENDING_APPROVAL gate (§6.6) on the round-trip path: a
    material edit empties the roster, and ``submit`` refuses until it is full
    again."""
    mission = world.mission_in(state, req("Pilot", {PILOT: Proficiency.PROFICIENT}))
    lead = world.lead()

    lead.missions.reopen(mission.id)
    lead.missions.update_plan(mission.id, replace(
        mission.plan, requirements=(req("Pilot", {PILOT: Proficiency.MASTER}),)))
    assert lead.missions.crewing_status(mission.id) is CrewingStatus.UNDER_CREWED

    with pytest.raises(GuardFailed, match="under_crewed"):
        lead.missions.submit(mission.id)

    proposal = lead.matching.run_matcher(mission.id)
    assert proposal.is_complete and not any(row.pinned for row in proposal.slots)
    for offer in lead.matching.offer_proposal(mission.id, proposal):
        world.session(offer.member_id).assignments.accept(offer.id)

    assert lead.missions.crewing_status(mission.id) is CrewingStatus.FULLY_CREWED
    assert lead.missions.submit(mission.id).state is MS.PENDING_APPROVAL





# ------------------------------------------------------------ reject and expire


def test_request_changes_returns_to_draft_with_feedback(world):
    mission = world.mission_in(MS.PENDING_APPROVAL)
    result = world.director().missions.request_changes(mission.id, "window clashes with LC-39A")
    assert result.state is MS.DRAFT

    events = [e for e in world.director().org.audit() if e.event == "request_changes"]
    assert events[-1].reason == "window clashes with LC-39A"


def test_a_pending_approval_expires(world):
    """Bounds how long a fully-crewed mission can wait on a Director while
    holding real people (§6.7)."""
    mission = world.mission_in(MS.PENDING_APPROVAL)
    assert mission.pending_expires_at == T0 + timedelta(days=7)

    world.advance(days=8)
    expired = world.director().system.expire_pending_approvals()

    assert [m.id for m in expired] == [mission.id]
    assert world.lead().missions.get(mission.id).state is MS.DRAFT


def test_an_unexpired_pending_approval_is_left_alone(world):
    mission = world.mission_in(MS.PENDING_APPROVAL)
    world.advance(days=3)
    assert world.director().system.expire_pending_approvals() == []
    assert world.lead().missions.get(mission.id).state is MS.PENDING_APPROVAL


# ------------------------------------------------------------ available_events


def test_available_events_is_scoped_to_the_actor(world):
    mission = world.mission_in(MS.PENDING_APPROVAL)

    assert set(world.director().missions.available_events(mission.id)) == {
        "approve", "cancel", "request_changes"
    }
    assert set(world.lead().missions.available_events(mission.id)) == {
        "cancel", "reopen"
    }
    assert world.other_lead().missions.available_events(mission.id) == ()


def test_blocked_events_explain_why(world):
    mission = world.draft()
    blocked = world.lead().missions.blocked_events(mission.id)
    assert "under_crewed" in blocked["submit"]
    assert "reason is required" in blocked["cancel"]


def test_an_illegal_transition_lists_what_is_possible(world):
    mission = world.draft()
    with pytest.raises(IllegalTransition) as caught:
        world.director().missions.approve(mission.id)
    assert "cannot approve a draft mission" in str(caught.value)


# -------------------------------------------------------------------- the audit


def test_the_event_log_answers_who_approved_what_against_which_plan(world):
    """For an approval workflow this is not a nice-to-have, it is the reason the
    workflow exists (§6.3)."""
    mission = world.mission_in(MS.APPROVED)
    audit = world.director().org.audit()

    approval = [e for e in audit if e.event == "approve"][-1]
    assert approval.actor_role is Role.DIRECTOR
    assert approval.from_state == "pending_approval" and approval.to_state == "approved"
    assert approval.plan_version == mission.plan.version
    assert approval.metadata["approved_roster"]

    assert [e.event for e in audit if e.subject_id == mission.id] == [
        "create", "run_matcher", "submit", "approve",
    ]
    # Crew responses are logged against the *assignment*, because they do not
    # move the mission (§8.3). They carry the mission id in metadata so the two
    # streams can be read together.
    assignment_events = [
        e.event for e in audit
        if e.subject_kind == "assignment" and e.metadata.get("mission_id") == mission.id
    ]
    assert assignment_events == ["offer", "accept"]


def test_a_plan_edit_records_whether_it_forced_a_rematch(world):
    """The audit says not just *what* changed but whether it mattered — which is
    the question a reviewer asks when the crew changed underneath an approval."""
    mission = world.mission_in(MS.APPROVED, req("Pilot", {PILOT: Proficiency.PROFICIENT}))
    lead = world.lead()
    lead.missions.reopen(mission.id)

    lead.missions.update_plan(mission.id, replace(mission.plan, site=Location("SLC-41")))
    cosmetic = [e for e in world.director().org.audit() if e.event == "plan_edited"][-1]
    assert cosmetic.metadata["rematch_needed"] is False

    lead.missions.update_plan(mission.id, replace(
        mission.plan, requirements=(req("Pilot", {PILOT: Proficiency.MASTER}),)))
    material = [e for e in world.director().org.audit() if e.event == "plan_edited"][-1]
    assert material.metadata["rematch_needed"] is True


def test_mission_state_is_a_projection_of_the_log(world):
    mission = world.mission_in(MS.ACTIVE)
    transitions = [
        e for e in world.director().org.audit()
        if e.subject_id == mission.id and e.to_state in {s.value for s in MissionState}
    ]
    assert transitions[-1].to_state == mission.state.value
    for earlier, later in zip(transitions, transitions[1:]):
        assert earlier.to_state == later.from_state, "the log must be a chain"
