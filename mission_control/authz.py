"""Authorisation: two layers (§5.1).

===============  ==========================================  =================
Layer            Question                                    Depends on
===============  ==========================================  =================
Capability       May a Lead approve missions *at all*?       role only
Contextual       May *this* Lead approve *this* mission?     actor + target + state
===============  ==========================================  =================

Capability alone cannot express "not your own mission" — the permission is held
in general and denied in a specific case. Conflating them produces the classic
bug where separation of duties lives as an ``if`` in one handler and is missing
from the bulk endpoint added six months later.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .domain import (
    Actor,
    Assignment,
    Denied,
    Mission,
    MissionId,
    Role,
)


class Permission(str, Enum):
    # governance — Directors
    ORG_SETTINGS_MANAGE = "org_settings_manage"
    MEMBER_MANAGE = "member_manage"
    AUDIT_READ = "audit_read"

    # authoring — Leads
    MISSION_CREATE = "mission_create"
    MISSION_UPDATE = "mission_update"
    MISSION_SUBMIT = "mission_submit"
    MATCHER_RUN = "matcher_run"
    ASSIGNMENT_OFFER = "assignment_offer"
    MISSION_ACTIVATE = "mission_activate"
    MISSION_COMPLETE = "mission_complete"

    # adjudication — Directors
    MISSION_APPROVE = "mission_approve"
    MISSION_REJECT = "mission_reject"

    # governance that is not authorship — both
    MISSION_CANCEL = "mission_cancel"

    # visibility
    MISSION_VIEW_ALL = "mission_view_all"
    MISSION_VIEW_ASSIGNED = "mission_view_assigned"
    CREW_PROFILE_READ_ALL = "crew_profile_read_all"

    # self-service — Crew
    CREW_PROFILE_WRITE_OWN = "crew_profile_write_own"
    CREW_AVAILABILITY_WRITE_OWN = "crew_availability_write_own"
    ASSIGNMENT_RESPOND_OWN = "assignment_respond_own"


P = Permission

AUTHORING_PERMISSIONS = frozenset(
    {
        P.MISSION_CREATE,
        P.MISSION_UPDATE,
        P.MISSION_SUBMIT,
        P.MATCHER_RUN,
        P.ASSIGNMENT_OFFER,
        P.MISSION_ACTIVATE,
        P.MISSION_COMPLETE,
    }
)
"""Planning a mission. Directors hold none of these, which is what makes
separation of duties structural rather than policy-dependent (§5.2)."""


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    # Directors govern. They cannot author a mission — the brief assigns
    # planning to Leads, and more importantly it means a Director cannot approve
    # their own mission *because there is no such mission* (§5.2).
    Role.DIRECTOR: frozenset(
        {
            P.ORG_SETTINGS_MANAGE,
            P.MEMBER_MANAGE,
            P.AUDIT_READ,
            P.MISSION_APPROVE,
            P.MISSION_REJECT,
            P.MISSION_CANCEL,  # governance, not authorship: approval is not a one-way door
            P.MISSION_VIEW_ALL,
            P.MISSION_VIEW_ASSIGNED,
            P.CREW_PROFILE_READ_ALL,
        }
    ),
    # Leads plan. Everything here except the view permissions is scoped to their
    # own missions by OwnershipPolicy.
    Role.MISSION_LEAD: frozenset(
        {
            P.MISSION_CREATE,
            P.MISSION_UPDATE,
            P.MISSION_SUBMIT,
            P.MATCHER_RUN,
            P.ASSIGNMENT_OFFER,
            P.MISSION_ACTIVATE,
            P.MISSION_COMPLETE,
            P.MISSION_CANCEL,
            P.MISSION_VIEW_ALL,
            P.MISSION_VIEW_ASSIGNED,
            P.CREW_PROFILE_READ_ALL,  # skills and availability only; redacted on serialisation
        }
    ),
    # Crew execute. ASSIGNMENT_RESPOND_OWN is the only permission here that
    # mutates anything beyond their own profile (§8.2).
    Role.CREW: frozenset(
        {
            P.MISSION_VIEW_ASSIGNED,
            P.CREW_PROFILE_WRITE_OWN,
            P.CREW_AVAILABILITY_WRITE_OWN,
            P.ASSIGNMENT_RESPOND_OWN,
        }
    ),
}

assert not (ROLE_PERMISSIONS[Role.DIRECTOR] & AUTHORING_PERMISSIONS), (
    "Directors must hold no authoring permissions — that is what makes "
    "self-approval unrepresentable rather than policy-checked (§5.2)"
)


def has(actor: Actor, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[actor.role]


def require(actor: Actor, permission: Permission) -> None:
    """Layer one: does this role hold the permission at all?"""
    if not has(actor, permission):
        raise Denied(
            "MISSING_PERMISSION",
            f"a {actor.role.value} cannot {permission.value}",
        )


# -------------------------------------------------------------------- layer two


@dataclass(frozen=True)
class Denial:
    code: str
    message: str


class PolicyCtx(Protocol):
    """What a policy may look at. Narrow on purpose: a policy that could reach
    the whole platform would be able to reach another tenant."""

    actor: Actor

    def assignments_for_mission(self, mission_id: MissionId) -> Sequence[Assignment]: ...


class Policy(Protocol):
    def check(self, actor: Actor, target: Any, ctx: PolicyCtx) -> Denial | None: ...


@dataclass(frozen=True)
class OwnershipPolicy:
    """A Lead may act only on missions they created (§5.3).

    Without it, "Leads manage missions" means any Lead can cancel any other's.
    Directors bypass it, since governance is not scoped to what you authored —
    in practice that only matters for ``cancel``, because Directors hold no
    authoring permissions anyway.
    """

    def check(self, actor: Actor, target: Any, ctx: PolicyCtx) -> Denial | None:
        if actor.role is Role.DIRECTOR:
            return None
        if getattr(target, "created_by", None) == actor.member_id:
            return None
        return Denial(
            "NOT_OWNER",
            "you can only act on missions you created",
        )


@dataclass(frozen=True)
class MissionOwnershipPolicy:
    """Ownership of the mission an assignment belongs to, for ``offer`` (§8.2).

    The target there is an Assignment, so ownership has to be read off
    ``ctx.mission`` instead of the target itself.
    """

    def check(self, actor: Actor, target: Any, ctx: PolicyCtx) -> Denial | None:
        mission = getattr(ctx, "mission", None)
        if mission is None:
            return Denial("NO_MISSION_CONTEXT", "assignment is not attached to a mission")
        return OwnershipPolicy().check(actor, mission, ctx)


@dataclass(frozen=True)
class SeparationOfDuties:
    """A **backstop**, since §5.2 makes self-approval structurally impossible.

    Kept cheaply as defence in depth: if a role is widened later, a fourth role
    added, or missions imported with an arbitrary ``created_by``, the structural
    guarantee weakens and this still holds.

    It checks ``created_by`` **and** ``submitted_by`` — checking only the
    submitter is defeated by asking a colleague to press Submit.
    """

    def check(self, actor: Actor, target: Any, ctx: PolicyCtx) -> Denial | None:
        authors = {getattr(target, "created_by", None), getattr(target, "submitted_by", None)}
        if actor.member_id in authors:
            return Denial(
                "SELF_APPROVAL",
                "you cannot approve or reject a mission you authored or submitted — "
                "ask another Director",
            )
        return None


@dataclass(frozen=True)
class OwnAssignmentPolicy:
    """Crew may respond only to their own assignment (§8.2)."""

    def check(self, actor: Actor, target: Any, ctx: PolicyCtx) -> Denial | None:
        if getattr(target, "member_id", None) == actor.member_id:
            return None
        return Denial("NOT_YOUR_ASSIGNMENT", "you can only respond to your own assignments")


def evaluate(
    policies: Iterable[Policy], actor: Actor, target: Any, ctx: PolicyCtx
) -> Denial | None:
    for policy in policies:
        denial = policy.check(actor, target, ctx)
        if denial is not None:
            return denial
    return None


def enforce(policies: Iterable[Policy], actor: Actor, target: Any, ctx: PolicyCtx) -> None:
    denial = evaluate(policies, actor, target, ctx)
    if denial is not None:
        raise Denied(denial.code, denial.message)


# -------------------------------------------------------------------- visibility


def visible_missions(actor: Actor, ctx: PolicyCtx, missions: Iterable[Mission]) -> Iterator[Mission]:
    """Crew have limited visibility, which is a *scoping* concern (§5.3).

    Listing filters rather than rejecting; direct access to a non-visible
    mission raises NotFound. Crew additionally see a redacted view —
    requirements and window, not other members' details — with redaction in the
    serialisation layer against the Actor, so a field added later is not exposed
    by default.
    """
    if has(actor, P.MISSION_VIEW_ALL):
        yield from missions
        return
    for mission in missions:
        if any(a.member_id == actor.member_id for a in ctx.assignments_for_mission(mission.id)):
            yield mission
