"""The tenant sandbox (§3).

Each tenant is self-contained; all its state lives inside it. In a shared-table
design isolation depends on every query remembering a tenant predicate, so a
leak is one forgotten ``WHERE`` away *and fails silently*. Here, reading across
tenants requires deliberately iterating the platform map — the leak is not
guarded against, it is unrepresentable (§3.1).

Three structural rules (§3.2):

1. **No service function takes a tenant_id.** Scope resolves once, from the
   Actor. A caller cannot ask for another tenant because there is no argument
   through which to.
2. **TenantWorkspace is the only handle to data**, closing over exactly one
   Tenant. It never exposes Platform.
3. **Unknown or foreign ids raise NotFound, never Forbidden** — a 403 confirms
   the record exists.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

from .domain import (
    Assignment,
    AssignmentId,
    Actor,
    CrewProfile,
    Denied,
    Member,
    MemberId,
    MemberStatus,
    Mission,
    MissionEvent,
    MissionId,
    NotFound,
    Notification,
    Skill,
    SkillId,
    StaleVersion,
    TenantId,
    TenantSettings,
)


@dataclass
class Tenant:
    """An organisation: space agency, research lab, private company.

    The one mutable object in the design — everything it holds is frozen, so a
    write is a dict swap and a snapshot is a reference copy (§2).
    """

    id: TenantId
    name: str
    settings: TenantSettings = field(default_factory=TenantSettings)

    skills: dict[SkillId, Skill] = field(default_factory=dict)
    members: dict[MemberId, Member] = field(default_factory=dict)
    crew_profiles: dict[MemberId, CrewProfile] = field(default_factory=dict)
    missions: dict[MissionId, Mission] = field(default_factory=dict)
    assignments: dict[AssignmentId, Assignment] = field(default_factory=dict)

    by_member: dict[MemberId, list[AssignmentId]] = field(default_factory=dict)
    """A *lookup* index (§7.8): no semantic value, rebuildable from
    ``assignments``, updated in the same critical section as the assignment
    write — which is why it does not violate the derive-don't-store rule applied
    to CrewingStatus and availability.

    It holds assignment ids, not mission ids, because the state that matters
    lives on the assignment: a DECLINED assignment has a mission with perfectly
    good dates and no load from it. Ids rather than object references, since
    with copy-on-write a stored reference would pin a superseded version.
    """

    audit: list[MissionEvent] = field(default_factory=list)
    notifications: list[Notification] = field(default_factory=list)

    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    """One lock per tenant, with **no computation inside it**.

    The problem with a tenant-wide lock is *duration*, not scope: every critical
    section here is read-version -> validate -> swap. It cannot deadlock,
    because only one lock exists. Per-aggregate locks are a documented next step
    (one per crew member and per mission, acquired in the total order
    ``crew < mission``), to be introduced by a profile rather than by
    anticipation (§3.3).

    Reentrant because a mission effect fires assignment events that take the
    same lock.
    """

    _ids: itertools.count = field(default_factory=lambda: itertools.count(1), repr=False)

    def next_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids):04d}"


@dataclass(frozen=True)
class TenantSnapshot:
    """An immutable, coherent read of one tenant.

    Captured under the lock in microseconds — the values are frozen, so the
    snapshot stays coherent with nothing held. Copy-on-write plus immutable
    values is a poor man's MVCC, already paid for by the domain being frozen
    (§3.3).
    """

    members: Mapping[MemberId, Member]
    crew: Mapping[MemberId, CrewProfile]
    missions: Mapping[MissionId, Mission]
    assignments: Mapping[AssignmentId, Assignment]
    by_member: Mapping[MemberId, tuple[AssignmentId, ...]]

    def assignments_for(self, member_id: MemberId) -> tuple[Assignment, ...]:
        return tuple(
            self.assignments[a] for a in self.by_member.get(member_id, ()) if a in self.assignments
        )

    def assignments_for_mission(self, mission_id: MissionId) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assignments.values() if a.mission_id == mission_id)


class TenantWorkspace:
    """The only handle to a tenant's data, closing over exactly one Tenant.

    Satisfies ``lifecycle.Store`` structurally. Never exposes the Platform, so
    there is no route from a workspace to another tenant.
    """

    def __init__(self, tenant: Tenant, actor: Actor) -> None:
        self._tenant = tenant
        self.actor = actor

    # -- scope

    @property
    def settings(self) -> TenantSettings:
        return self._tenant.settings

    @property
    def tenant_name(self) -> str:
        return self._tenant.name

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """A critical section. Keep it free of computation: the matcher runs
        outside, on a snapshot (§3.3)."""
        with self._tenant.lock:
            yield

    def snapshot(self) -> TenantSnapshot:
        with self._tenant.lock:  # microseconds: reference copies only
            return TenantSnapshot(
                members=dict(self._tenant.members),
                crew=dict(self._tenant.crew_profiles),
                missions=dict(self._tenant.missions),
                assignments=dict(self._tenant.assignments),
                by_member={k: tuple(v) for k, v in self._tenant.by_member.items()},
            )

    def new_id(self, prefix: str) -> str:
        with self._tenant.lock:
            return self._tenant.next_id(prefix)

    # -- reads. Every one raises NotFound for an id this tenant does not hold,
    #    which is also what a *foreign* id looks like from in here.

    def mission(self, mission_id: MissionId) -> Mission:
        try:
            return self._tenant.missions[mission_id]
        except KeyError:
            raise NotFound("mission", mission_id) from None

    def member(self, member_id: MemberId) -> Member:
        try:
            return self._tenant.members[member_id]
        except KeyError:
            raise NotFound("member", member_id) from None

    def crew_profile(self, member_id: MemberId) -> CrewProfile:
        try:
            return self._tenant.crew_profiles[member_id]
        except KeyError:
            raise NotFound("crew profile", member_id) from None

    def assignment(self, assignment_id: AssignmentId) -> Assignment:
        try:
            return self._tenant.assignments[assignment_id]
        except KeyError:
            raise NotFound("assignment", assignment_id) from None

    def skill(self, skill_id: SkillId) -> Skill:
        try:
            return self._tenant.skills[skill_id]
        except KeyError:
            raise NotFound("skill", skill_id) from None

    def assignments_for_mission(self, mission_id: MissionId) -> list[Assignment]:
        return [a for a in self._tenant.assignments.values() if a.mission_id == mission_id]

    def assignments_for_member(self, member_id: MemberId) -> list[Assignment]:
        return [
            self._tenant.assignments[a]
            for a in self._tenant.by_member.get(member_id, ())
            if a in self._tenant.assignments
        ]

    def missions(self) -> list[Mission]:
        return list(self._tenant.missions.values())

    def members(self) -> list[Member]:
        return list(self._tenant.members.values())

    def skills(self) -> list[Skill]:
        return list(self._tenant.skills.values())

    def audit(self) -> list[MissionEvent]:
        return list(self._tenant.audit)

    def notifications(self, member_id: MemberId | None = None) -> list[Notification]:
        return [
            n for n in self._tenant.notifications if member_id is None or n.to == member_id
        ]

    # -- writes. Compare-and-swap on version: the authoritative check happens
    #    here, at the point of mutation, not back when the caller read (§3.3).

    def save_mission(self, mission: Mission) -> None:
        with self._tenant.lock:
            current = self._tenant.missions.get(mission.id)
            if current is not None and current.version != mission.version - 1:
                raise StaleVersion("mission", mission.id, mission.version - 1, current.version)
            self._tenant.missions[mission.id] = mission

    def save_assignment(self, assignment: Assignment) -> None:
        with self._tenant.lock:
            current = self._tenant.assignments.get(assignment.id)
            if current is not None and current.version != assignment.version - 1:
                raise StaleVersion(
                    "assignment", assignment.id, assignment.version - 1, current.version
                )
            self._tenant.assignments[assignment.id] = assignment
            index = self._tenant.by_member.setdefault(assignment.member_id, [])
            if assignment.id not in index:
                index.append(assignment.id)

    def save_crew_profile(self, profile: CrewProfile) -> None:
        with self._tenant.lock:
            current = self._tenant.crew_profiles.get(profile.member_id)
            if current is not None and current.version != profile.version - 1:
                raise StaleVersion(
                    "crew profile", profile.member_id, profile.version - 1, current.version
                )
            self._tenant.crew_profiles[profile.member_id] = profile

    def save_member(self, member: Member) -> None:
        with self._tenant.lock:
            current = self._tenant.members.get(member.id)
            if current is not None and current.version != member.version - 1:
                raise StaleVersion("member", member.id, member.version - 1, current.version)
            self._tenant.members[member.id] = member

    def save_skill(self, skill: Skill) -> None:
        with self._tenant.lock:
            self._tenant.skills[skill.id] = skill

    def save_settings(self, settings: TenantSettings) -> None:
        with self._tenant.lock:
            self._tenant.settings = settings

    def record_event(self, event: MissionEvent) -> None:
        with self._tenant.lock:
            self._tenant.audit.append(event)

    def notify(
        self, member_id: MemberId, subject: str, body: str, *, at: datetime
    ) -> None:
        """Recorded rather than sent. Keeps the core I/O-free and effects
        assertable in tests.

        ``at`` is passed in rather than read from the clock, for the same reason
        every other timestamp is: a run has to be reproducible.
        """
        with self._tenant.lock:
            self._tenant.notifications.append(
                Notification(at=at, to=member_id, subject=subject, body=body)
            )


class Platform:
    """The only global structure: ``dict[TenantId, Tenant]``.

    This is the in-memory analogue of database-per-tenant — near-perfect
    isolation, at the cost of awkward cross-tenant analytics (§3.1).
    """

    def __init__(self) -> None:
        self._tenants: dict[TenantId, Tenant] = {}
        self._lock = threading.RLock()

    # -- administration. These take a tenant_id because they *are* the tenancy
    #    boundary; nothing downstream of session() does.

    def create_tenant(
        self, tenant_id: TenantId, name: str, settings: TenantSettings | None = None
    ) -> Tenant:
        with self._lock:
            if tenant_id in self._tenants:
                raise ValueError(f"tenant {tenant_id!r} already exists")
            tenant = Tenant(id=tenant_id, name=name, settings=settings or TenantSettings())
            self._tenants[tenant_id] = tenant
            return tenant

    def session(self, actor: Actor) -> TenantWorkspace:
        """Resolve scope once, from the authenticated Actor.

        Raises NotFound when the actor's member record is not in the tenant it
        claims — which is exactly what a forged cross-tenant token looks like.
        """
        with self._lock:
            tenant = self._tenants.get(actor.tenant_id)
        if tenant is None:
            raise NotFound("tenant", actor.tenant_id)

        member = tenant.members.get(actor.member_id)
        if member is None:
            raise NotFound("member", actor.member_id)
        if member.role is not actor.role:
            # The token's claim disagrees with the record. Denied rather than
            # NotFound: the member exists in this tenant, so there is nothing to
            # leak, and a role change mid-session is worth surfacing.
            raise Denied(
                "ROLE_MISMATCH",
                f"token claims {actor.role.value}, record says {member.role.value}",
            )
        if member.status is not MemberStatus.ACTIVE:
            raise Denied("MEMBER_INACTIVE", f"{member.name} is deactivated")

        return TenantWorkspace(tenant, actor)

    def actor_for(self, tenant_id: TenantId, member_id: MemberId) -> Actor:
        """Stand-in for authentication: builds the Actor a token would carry."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise NotFound("tenant", tenant_id)
        member = tenant.members.get(member_id)
        if member is None:
            raise NotFound("member", member_id)
        return Actor(tenant_id=tenant_id, member_id=member_id, role=member.role)

    def tenant_ids(self) -> Sequence[TenantId]:
        """Deliberately the *only* way to enumerate tenants, and it returns ids
        rather than Tenants — a test fixture and an admin tool need it; no
        request path does."""
        with self._lock:
            return tuple(self._tenants)

    def _unsafe_tenant(self, tenant_id: TenantId) -> Tenant:
        """Seeding hook for fixtures and the demo. Underscored because reaching
        a Tenant without an Actor is exactly what §3.2 forbids at runtime."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise NotFound("tenant", tenant_id)
        return tenant


def bootstrap_role(platform: Platform, tenant_id: TenantId, member: Member) -> Member:
    """Create the first member of a tenant, before anyone can authenticate.

    A tenant with no Director cannot have one created through the normal path,
    since MEMBER_MANAGE is a Director permission. Real systems solve this at
    signup; here it is one honest hole rather than a special-cased Actor.
    """
    tenant = platform._unsafe_tenant(tenant_id)
    with tenant.lock:
        tenant.members[member.id] = member
    return member
