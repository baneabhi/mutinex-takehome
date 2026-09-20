"""HTTP API.

A thin adapter over :mod:`mission_control.services`. It decides nothing — every
authorisation question, guard and state transition happens below this layer.
What it contributes is three translations:

1. **Credentials to an Actor.** The token names an identity; the *role* is read
   from the member record, never from the token (:func:`current_scope`).
2. **Domain errors to status codes**, preserving the distinctions §6.4 cares
   about — including the ``available`` events on an illegal transition.
3. **Domain objects to the wire**, via :mod:`mission_control.schemas`, where
   Crew get a different model rather than a filtered one (§5.3).

**No route path contains a tenant id.** It is ``/missions``, not
``/tenants/{id}/missions`` — the HTTP-level form of §3.2's rule that scope
resolves once, from the caller's identity, and is not something a request can
ask for. ``test_api.py`` asserts that structurally.

**There is no session store.** An Actor, a TenantWorkspace and a Scope are
built per request and discarded; the only thing that outlives a request is the
Platform. Two callers with different tokens are simply two requests.

Operational note: run with **one worker**. Each worker process would hold its
own in-memory Platform, so two workers means two divergent universes. That is
the in-memory storage decision (§11.1) surfacing at the deployment layer, and
the fix is persistence, not fewer requests.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from . import schemas as s
from .authz import ROLE_PERMISSIONS, Permission, has
from .domain import (
    Denied,
    DomainError,
    GuardFailed,
    IllegalTransition,
    Location,
    MemberId,
    MissionId,
    MissionPlan,
    NotFound,
    Requirement,
    RequirementId,
    Role,
    SkillId,
    SkillRequirement,
    StaleVersion,
    TenantId,
    TimeWindow,
    UnavailabilityBlock,
    CrewSkill,
    Member,
    as_utc,
    utc_now,
)
from .services import Scope
from .store import Platform, bootstrap_role

# --------------------------------------------------------------------- tokens


def encode_token(tenant_id: str, member_id: str) -> str:
    raw = f"{tenant_id}:{member_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_token(token: str) -> tuple[TenantId, MemberId]:
    """Turn a bearer token into the identity it claims.

    **This is the only auth-aware line in the system, and it is a stub.** The
    token is a reversible encoding of ``tenant:member`` — readable, trivially
    forgeable, and not authentication in any sense. Identity is *not* verified;
    anyone who can name a pair can act as them.

    Authorisation, by contrast, is fully enforced below: role, ownership,
    separation of duties and visibility all apply to whoever this resolves to.
    Replacing this with a signed token or a session lookup changes this function
    and nothing else, because everything downstream takes an Actor the server
    derived rather than one the caller supplied.
    """
    padded = token + "=" * (-len(token) % 4)
    try:
        tenant_id, _, member_id = base64.urlsafe_b64decode(padded).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        raise HTTPException(401, "malformed token") from None
    if not tenant_id or not member_id:
        raise HTTPException(401, "malformed token")
    return TenantId(tenant_id), MemberId(member_id)


# --------------------------------------------------------------- dependencies


def get_platform(request: Request) -> Platform:
    return request.app.state.platform


def _clock(request: Request, simulated: str | None):
    """The domain threads ``now`` through everything, so a test can pin it.

    Exposed as a header only when the server was started with
    ``allow_clock_override``, because it is a test affordance: it makes offer
    expiry demonstrable without waiting 72 hours.
    """
    if simulated and request.app.state.allow_clock_override:
        moment = as_utc(datetime.fromisoformat(simulated))
        return lambda: moment
    return utc_now


def current_scope(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_simulated_now: Annotated[str | None, Header()] = None,
) -> Scope:
    """Derive the caller's Actor and open a Scope over their tenant.

    The role comes from the member record via ``Platform.actor_for`` — not from
    the token — so there is no authority claim in the credential to tamper
    with. ``Platform.session`` then re-validates tenant, member, role and
    status, which also catches an Actor built before somebody's role changed.
    """
    scheme, _, raw = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not raw.strip():
        raise HTTPException(401, "missing bearer token")
    tenant_id, member_id = decode_token(raw.strip())

    platform = get_platform(request)
    actor = platform.actor_for(tenant_id, member_id)  # role read from the record
    return Scope(platform.session(actor), _clock(request, x_simulated_now))


CurrentScope = Annotated[Scope, Depends(current_scope)]


# ------------------------------------------------------------- error mapping


ERROR_STATUS: dict[type[DomainError], int] = {
    # A foreign tenant's id looks exactly like a missing one, deliberately: a
    # 403 would confirm the record exists somewhere (§3.2).
    NotFound: 404,
    Denied: 403,
    # "You may do this, but not yet" and "the world moved under you" are both
    # conflicts with current state rather than bad requests.
    GuardFailed: 409,
    IllegalTransition: 409,
    StaleVersion: 409,
}


def _error_body(exc: DomainError) -> dict:
    if isinstance(exc, NotFound):
        return {"error": "NOT_FOUND", "message": str(exc), "kind": exc.kind, "id": exc.id}
    if isinstance(exc, Denied):
        return {"error": exc.code, "message": exc.message}
    if isinstance(exc, IllegalTransition):
        return {
            "error": "ILLEGAL_TRANSITION", "message": str(exc),
            "state": exc.state, "attempted": exc.event,
            # What you *can* do instead. Free, because available_events already
            # exists for the UI (§6.4).
            "available": list(exc.available),
        }
    if isinstance(exc, GuardFailed):
        return {
            "error": "PRECONDITION_FAILED", "message": str(exc),
            "attempted": exc.event, "reason": exc.reason,
        }
    if isinstance(exc, StaleVersion):
        return {"error": "STALE_VERSION", "message": str(exc)}
    return {"error": "DOMAIN_ERROR", "message": str(exc)}


# --------------------------------------------------------------- conversions


def _plan(payload: s.PlanIn) -> MissionPlan:
    return MissionPlan(
        requirements=tuple(
            Requirement(
                id=RequirementId(
                    r.id or f"r-{r.label.lower().replace(' ', '-')}-{index}"
                ),
                label=r.label,
                count=r.count,
                mandatory=r.mandatory,
                skills=tuple(
                    SkillRequirement(SkillId(sk.skill_id), s.parse_level(sk.min_level))
                    for sk in r.skills
                ),
            )
            for index, r in enumerate(payload.requirements)
        ),
        window=TimeWindow(payload.window.start, payload.window.end),
        site=Location(payload.site),
    )


def _mission_view(scope: Scope, mission_id: MissionId):
    """MissionDetail or MissionForCrew, decided by permission (§5.3)."""
    mission = scope.missions.get(mission_id)
    assignments = scope.workspace.assignments_for_mission(mission.id)
    crewing = scope.missions.crewing_status(mission.id).value

    if not has(scope.actor, Permission.MISSION_VIEW_ALL):
        mine = next(
            (a for a in assignments if a.member_id == scope.actor.member_id), None
        )
        return s.MissionForCrew(
            **s.MissionSummary.of(mission, crewing).model_dump(),
            plan=s.PlanOut.of(mission.plan),
            my_assignment=s.AssignmentOut.of(mine) if mine else None,
        )

    names = {m.id: m.name for m in scope.workspace.members()}
    return s.MissionDetail(
        **s.MissionSummary.of(mission, crewing).model_dump(),
        plan=s.PlanOut.of(mission.plan),
        description=mission.description,
        tags=list(mission.tags),
        notes=mission.notes,
        reference_code=mission.reference_code,
        created_by=mission.created_by,
        submitted_by=mission.submitted_by,
        approved_by=mission.approved_by,
        approved_plan_version=mission.approved_plan_version,
        pending_expires_at=mission.pending_expires_at,
        closed_reason=mission.closed_reason,
        version=mission.version,
        roster=[s.AssignmentOut.of(a, names.get(a.member_id)) for a in assignments],
        available_events=list(scope.missions.available_events(mission.id)),
        blocked_events=scope.missions.blocked_events(mission.id),
    )


# ----------------------------------------------------------------- the app


def create_app(
    platform: Platform | None = None,
    *,
    dev_tokens: bool = False,
    allow_clock_override: bool = False,
) -> FastAPI:
    app = FastAPI(
        title="Mission Control",
        version="0.1.0",
        description=(
            "Multi-tenant crewing platform. Authenticate with "
            "`Authorization: Bearer <token>`; every route is scoped to the "
            "tenant that token belongs to, which is why no path contains a "
            "tenant id."
        ),
    )
    app.state.platform = platform or Platform()
    app.state.dev_tokens = dev_tokens
    app.state.allow_clock_override = allow_clock_override

    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        status = next(
            (code for kind, code in ERROR_STATUS.items() if isinstance(exc, kind)), 400
        )
        return JSONResponse(status_code=status, content=_error_body(exc))

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400, content={"error": "INVALID", "message": str(exc)}
        )

    # -- health and identity

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "tenants": len(app.state.platform.tenant_ids())}

    @app.get("/me", response_model=s.MeOut, tags=["meta"])
    def me(scope: CurrentScope) -> s.MeOut:
        """The Actor the server derived from your credentials, with the
        permissions it carries — layer one of §5.1, made inspectable."""
        member = scope.workspace.member(scope.actor.member_id)
        return s.MeOut(
            tenant_id=scope.actor.tenant_id,
            tenant_name=scope.workspace.tenant_name,
            member_id=scope.actor.member_id,
            name=member.name,
            role=scope.actor.role.value,
            permissions=sorted(p.value for p in ROLE_PERMISSIONS[scope.actor.role]),
        )

    # -- bootstrap: the one unauthenticated route

    @app.post("/bootstrap", response_model=s.TokenOut, status_code=201, tags=["auth"])
    def bootstrap(body: s.BootstrapRequest) -> s.TokenOut:
        """Create a tenant and its first Director.

        Unauthenticated because it has to be: MEMBER_MANAGE is a Director
        permission, so a tenant's first Director cannot be created through the
        normal path. Stands in for signup.
        """
        platform = app.state.platform
        platform.create_tenant(TenantId(body.tenant_id), body.name)
        member = Member(
            id=MemberId("mem-director"), name=body.director_name, role=Role.DIRECTOR
        )
        bootstrap_role(platform, TenantId(body.tenant_id), member)
        return s.TokenOut(
            token=encode_token(body.tenant_id, member.id),
            tenant_id=body.tenant_id, member_id=member.id,
            name=member.name, role=member.role.value,
        )

    @app.post("/tokens", response_model=s.TokenOut, tags=["auth"])
    def mint_token(body: s.TokenRequest, scope: CurrentScope) -> s.TokenOut:
        """Mint a token for another member of your tenant.

        **Development affordance, and wrong in production** — it lets a Director
        act as any crew member. Enabled with `--dev-tokens`; the CLI needs it
        because a demo has to accept offers as several different people. A real
        deployment issues tokens from a login endpoint instead.
        """
        if not app.state.dev_tokens:
            raise HTTPException(404, "token minting is disabled")
        member = scope.workspace.member(MemberId(body.member_id))
        return s.TokenOut(
            token=encode_token(scope.actor.tenant_id, member.id),
            tenant_id=scope.actor.tenant_id, member_id=member.id,
            name=member.name, role=member.role.value,
        )

    # -- organisation

    @app.get("/members", response_model=list[s.MemberOut], tags=["org"])
    def list_members(scope: CurrentScope) -> list[s.MemberOut]:
        return [s.MemberOut.of(m) for m in scope.workspace.members()]

    @app.post("/members", response_model=s.MemberOut, status_code=201, tags=["org"])
    def add_member(body: s.MemberCreate, scope: CurrentScope) -> s.MemberOut:
        return s.MemberOut.of(scope.org.add_member(body.name, Role(body.role)))

    @app.post("/members/{member_id}/deactivate", response_model=s.MemberOut, tags=["org"])
    def deactivate(
        member_id: str, body: s.DeactivateRequest, scope: CurrentScope
    ) -> s.MemberOut:
        return s.MemberOut.of(
            scope.org.deactivate_member(MemberId(member_id), body.reason)
        )

    @app.get("/skills", response_model=list[s.SkillOut], tags=["org"])
    def list_skills(scope: CurrentScope) -> list[s.SkillOut]:
        return [s.SkillOut.of(x) for x in scope.workspace.skills()]

    @app.post("/skills", response_model=s.SkillOut, status_code=201, tags=["org"])
    def add_skill(body: s.SkillCreate, scope: CurrentScope) -> s.SkillOut:
        return s.SkillOut.of(
            scope.org.define_skill(body.name, skill_id=SkillId(body.id) if body.id else None)
        )

    @app.get("/audit", response_model=list[s.EventOut], tags=["org"])
    def audit(scope: CurrentScope) -> list[s.EventOut]:
        return [s.EventOut.of(e) for e in scope.org.audit()]

    # -- crew

    @app.get("/crew/{member_id}", response_model=s.CrewProfileOut, tags=["crew"])
    def crew_profile(member_id: str, scope: CurrentScope) -> s.CrewProfileOut:
        profile = scope.crew.profile(MemberId(member_id))
        return s.CrewProfileOut.of(profile, scope.workspace.member(MemberId(member_id)))

    @app.put("/crew/{member_id}/skills", response_model=s.CrewProfileOut, tags=["crew"])
    def set_skills(
        member_id: str, body: s.SkillsRequest, scope: CurrentScope
    ) -> s.CrewProfileOut:
        profile = scope.crew.set_skills(
            MemberId(member_id),
            [CrewSkill(SkillId(k.skill_id), s.parse_level(k.level)) for k in body.skills],
        )
        return s.CrewProfileOut.of(profile, scope.workspace.member(MemberId(member_id)))

    @app.put("/crew/{member_id}/availability", response_model=s.CrewProfileOut, tags=["crew"])
    def set_availability(
        member_id: str, body: s.AvailabilityRequest, scope: CurrentScope
    ) -> s.CrewProfileOut:
        profile = scope.crew.set_unavailability(
            MemberId(member_id),
            [
                UnavailabilityBlock(TimeWindow(b.start, b.end), b.reason)
                for b in body.unavailability
            ],
        )
        return s.CrewProfileOut.of(profile, scope.workspace.member(MemberId(member_id)))

    # -- missions

    @app.get("/missions", response_model=list[s.MissionSummary], tags=["missions"])
    def list_missions(scope: CurrentScope) -> list[s.MissionSummary]:
        """Listing filters rather than rejecting; a mission you cannot see is
        simply absent (§5.3)."""
        return [
            s.MissionSummary.of(m, scope.missions.crewing_status(m.id).value)
            for m in scope.missions.list()
        ]

    @app.post("/missions", status_code=201, tags=["missions"])
    def create_mission(body: s.MissionCreate, scope: CurrentScope):
        mission = scope.missions.create(
            body.title, _plan(body.plan),
            description=body.description, tags=tuple(body.tags),
            notes=body.notes, reference_code=body.reference_code,
        )
        return _mission_view(scope, mission.id)

    @app.get("/missions/{mission_id}", tags=["missions"])
    def get_mission(mission_id: str, scope: CurrentScope):
        return _mission_view(scope, MissionId(mission_id))

    @app.put("/missions/{mission_id}/plan", tags=["missions"])
    def update_plan(mission_id: str, body: s.PlanIn, scope: CurrentScope):
        """Editable in DRAFT only. Anywhere else this is a 409 naming ``reopen``
        (§6.1.1)."""
        scope.missions.update_plan(MissionId(mission_id), _plan(body))
        return _mission_view(scope, MissionId(mission_id))

    @app.patch("/missions/{mission_id}", tags=["missions"])
    def update_metadata(mission_id: str, body: s.MetadataPatch, scope: CurrentScope):
        """Metadata only — title, description, tags, notes, reference code.
        These never move the mission, which is what stops a typo fix costing an
        approval (§6.1.1)."""
        fields = body.fields_set()
        if fields:
            scope.missions.update_metadata(MissionId(mission_id), **fields)
        return _mission_view(scope, MissionId(mission_id))

    @app.post("/missions/{mission_id}/events/{event}", tags=["missions"])
    def fire_event(
        mission_id: str, event: str, scope: CurrentScope, body: s.EventRequest | None = None
    ):
        """Every mission transition, through one endpoint.

        The transition table is the source of truth, and ``available_events`` on
        a GET tells you exactly what is postable here — so a new transition
        needs no new route, and the API cannot disagree with the state machine
        about what exists (§6.2).
        """
        scope.missions.fire(
            MissionId(mission_id), event, reason=body.reason if body else None
        )
        return _mission_view(scope, MissionId(mission_id))

    @app.get("/missions/{mission_id}/roster", response_model=list[s.AssignmentOut],
             tags=["missions"])
    def roster(mission_id: str, scope: CurrentScope) -> list[s.AssignmentOut]:
        names = {m.id: m.name for m in scope.workspace.members()}
        return [
            s.AssignmentOut.of(a, names.get(a.member_id))
            for a in scope.missions.roster(MissionId(mission_id))
        ]

    @app.get("/missions/{mission_id}/allocation-report",
             response_model=s.AllocationReportOut, tags=["missions"])
    def allocation_report(mission_id: str, scope: CurrentScope) -> s.AllocationReportOut:
        return s.AllocationReportOut.of(
            scope.missions.allocation_report(MissionId(mission_id))
        )

    # -- matching

    @app.post("/missions/{mission_id}/match", response_model=s.MatchProposalOut,
              tags=["matching"])
    def run_matcher(mission_id: str, scope: CurrentScope) -> s.MatchProposalOut:
        """Returns a proposal and creates nothing (§7.9). Safe to call
        repeatedly; a re-run pins whoever has already accepted."""
        return s.MatchProposalOut.of(scope.matching.run_matcher(MissionId(mission_id)))

    @app.post("/missions/{mission_id}/offers", response_model=list[s.AssignmentOut],
              status_code=201, tags=["matching"])
    def make_offers(
        mission_id: str, body: s.OfferRequest, scope: CurrentScope
    ) -> list[s.AssignmentOut]:
        created = [
            scope.assignments.offer(
                MissionId(mission_id), RequirementId(row.requirement_id),
                row.slot_index, MemberId(row.member_id),
            )
            for row in body.offers
        ]
        names = {m.id: m.name for m in scope.workspace.members()}
        return [s.AssignmentOut.of(a, names.get(a.member_id)) for a in created]

    # -- assignments

    @app.get("/assignments", response_model=list[s.AssignmentOut], tags=["assignments"])
    def list_assignments(
        scope: CurrentScope, mission_id: Annotated[str | None, Query()] = None
    ) -> list[s.AssignmentOut]:
        names = {m.id: m.name for m in scope.workspace.members()}
        found = (
            scope.assignments.for_mission(MissionId(mission_id))
            if mission_id
            else scope.assignments.mine()
        )
        return [s.AssignmentOut.of(a, names.get(a.member_id)) for a in found]

    @app.post("/assignments/{assignment_id}/accept", response_model=s.AssignmentOut,
              tags=["assignments"])
    def accept(assignment_id: str, scope: CurrentScope) -> s.AssignmentOut:
        return s.AssignmentOut.of(scope.assignments.accept(assignment_id))

    @app.post("/assignments/{assignment_id}/decline", response_model=s.AssignmentOut,
              tags=["assignments"])
    def decline(assignment_id: str, scope: CurrentScope) -> s.AssignmentOut:
        return s.AssignmentOut.of(scope.assignments.decline(assignment_id))

    # -- clock-driven sweeps

    @app.post("/system/expire-offers", response_model=list[s.AssignmentOut],
              tags=["system"])
    def expire_offers(scope: CurrentScope) -> list[s.AssignmentOut]:
        return [s.AssignmentOut.of(a) for a in scope.system.expire_offers()]

    @app.post("/system/expire-approvals", response_model=list[s.MissionSummary],
              tags=["system"])
    def expire_approvals(scope: CurrentScope) -> list[s.MissionSummary]:
        return [
            s.MissionSummary.of(m, scope.missions.crewing_status(m.id).value)
            for m in scope.system.expire_pending_approvals()
        ]

    return app
