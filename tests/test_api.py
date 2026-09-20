"""The HTTP surface (§13).

The API decides nothing, so these do not re-test authorisation or the state
machine — that is what the other files are for. What they pin is the three
things this layer *does* contribute: deriving an Actor from credentials,
mapping domain errors onto status codes without losing the distinctions §6.4
cares about, and redacting on the way out.
"""

from __future__ import annotations

import pytest

from mission_control.api import create_app, decode_token, encode_token

WINDOW = {"start": "2099-03-01T00:00:00Z", "end": "2099-03-21T00:00:00Z"}
PILOT_PLAN = {
    "requirements": [
        {"id": "r-pilot", "label": "Pilot",
         "skills": [{"skill_id": "pilot", "min_level": "PROFICIENT"}]}
    ],
    "window": WINDOW,
    "site": "LC-39A",
}


# ------------------------------------------------------------------ structure


def test_no_route_path_contains_a_tenant_id(api):
    """The HTTP form of §3.2's rule.

    Scope resolves once, from the caller's identity. A path segment for the
    tenant would be exactly the argument the design says must not exist —
    asserted structurally so a route added later fails here, not in review.
    """
    offenders = [
        route.path
        for route in api.app.routes
        if "tenant" in route.path.lower()
    ]
    assert offenders == []


def test_only_bootstrap_is_reachable_without_credentials(api):
    """Everything else needs a token. Enumerated from the route table so a new
    unauthenticated route has to be a deliberate act."""
    anonymous = []
    for route in api.app.routes:
        methods = getattr(route, "methods", set()) - {"HEAD", "OPTIONS"}
        if not methods or route.path in {"/openapi.json", "/docs", "/redoc",
                                         "/docs/oauth2-redirect"}:
            continue
        for method in methods:
            response = api.http.request(method, route.path.replace("{mission_id}", "x")
                                        .replace("{member_id}", "x")
                                        .replace("{assignment_id}", "x")
                                        .replace("{event}", "submit"))
            if response.status_code != 401:
                anonymous.append(f"{method} {route.path}")
    assert sorted(anonymous) == ["GET /health", "POST /bootstrap"]


# ----------------------------------------------------------------------- auth


def test_a_token_names_an_identity_and_nothing_more(api):
    """The role is read from the member record, not the credential — so there
    is no authority claim in a token to tamper with."""
    assert decode_token(encode_token("nasa", "mem-lead")) == ("nasa", "mem-lead")

    me = api.http.get("/me", headers=api.headers("nasa", "mem-lead")).json()
    assert me["role"] == "mission_lead"
    assert me["tenant_id"] == "nasa"
    assert "mission_create" in me["permissions"]
    assert "mission_approve" not in me["permissions"]


@pytest.mark.parametrize(
    "header",
    [None, {"Authorization": "Bearer !!!!"}, {"Authorization": "Basic abc"},
     {"Authorization": "Bearer "}],
)
def test_bad_credentials_are_401(api, header):
    assert api.http.get("/missions", headers=header or {}).status_code == 401


def test_a_token_for_an_unknown_member_is_404(api):
    """Not 401: the credential is well-formed, the identity simply is not
    there — which is also what a forged cross-tenant token looks like."""
    response = api.http.get("/me", headers=api.headers("nasa", "mem-nobody"))
    assert response.status_code == 404


def test_token_minting_is_off_unless_asked_for():
    app = create_app(dev_tokens=False)
    from fastapi.testclient import TestClient

    http = TestClient(app)
    token = http.post("/bootstrap", json={
        "tenant_id": "x", "name": "X", "director_name": "D"}).json()["token"]
    response = http.post("/tokens", json={"member_id": "mem-director"},
                         headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 404, "impersonation must not be on by default"


# ------------------------------------------------------------------- tenancy


def test_a_mission_is_invisible_to_the_other_tenant(api):
    nasa = api.headers("nasa", "mem-lead")
    esa = api.headers("esa", "mem-lead")

    mission_id = api.http.post(
        "/missions", json={"title": "Artemis VII", "plan": PILOT_PLAN}, headers=nasa
    ).json()["id"]

    assert api.http.get("/missions", headers=esa).json() == []
    response = api.http.get(f"/missions/{mission_id}", headers=esa)
    assert response.status_code == 404
    assert response.json()["error"] == "NOT_FOUND"


def test_colliding_member_ids_resolve_within_the_callers_tenant(api):
    """Both tenants have a ``mem-lead`` named Lena Sorokin, on purpose."""
    nasa = api.http.get("/me", headers=api.headers("nasa", "mem-lead")).json()
    esa = api.http.get("/me", headers=api.headers("esa", "mem-lead")).json()

    assert nasa["member_id"] == esa["member_id"] == "mem-lead"
    assert nasa["tenant_name"] == "NASA"
    assert esa["tenant_name"] == "European Space Agency"


# ------------------------------------------------------------- error mapping


def test_missing_permission_is_403_with_a_code(api):
    response = api.http.post("/members", json={"name": "X", "role": "crew"},
                             headers=api.headers("nasa", "mem-lead"))
    assert response.status_code == 403
    assert response.json()["error"] == "MISSING_PERMISSION"


def test_an_unmet_precondition_is_409_naming_the_reason(api):
    nasa = api.headers("nasa", "mem-lead")
    mission_id = api.http.post(
        "/missions", json={"title": "Artemis", "plan": PILOT_PLAN}, headers=nasa
    ).json()["id"]

    response = api.http.post(f"/missions/{mission_id}/events/submit", headers=nasa)
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "PRECONDITION_FAILED"
    assert body["attempted"] == "submit"
    assert "under_crewed" in body["reason"]


def test_an_illegal_transition_is_409_carrying_what_you_can_do(api):
    """The distinction §6.4 exists for: the API does not just refuse, it says
    what is possible instead."""
    nasa = api.headers("nasa", "mem-lead")
    mission_id = api.http.post(
        "/missions", json={"title": "Artemis", "plan": PILOT_PLAN}, headers=nasa
    ).json()["id"]

    response = api.http.post(f"/missions/{mission_id}/events/activate", headers=nasa)
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "ILLEGAL_TRANSITION"
    assert body["state"] == "draft"
    assert set(body["available"]) == {"cancel", "plan_edited", "run_matcher", "submit"}


def test_a_malformed_body_is_422_not_a_500(api):
    response = api.http.post(
        "/missions",
        json={"title": "X", "plan": PILOT_PLAN, "nonsense": 1},
        headers=api.headers("nasa", "mem-lead"),
    )
    assert response.status_code == 422


def test_an_unknown_proficiency_is_400_listing_the_valid_ones(api):
    bad = {**PILOT_PLAN, "requirements": [
        {"id": "r", "label": "Pilot",
         "skills": [{"skill_id": "pilot", "min_level": "WIZARD"}]}]}
    response = api.http.post("/missions", json={"title": "X", "plan": bad},
                             headers=api.headers("nasa", "mem-lead"))
    assert response.status_code == 400
    assert "MASTER" in response.json()["message"]


# ----------------------------------------------------------------- redaction


def test_crew_get_a_different_model_not_a_filtered_one(api):
    """§5.3. The absent fields are absent because MissionForCrew does not have
    them, so a field added to MissionDetail is withheld by default."""
    lead = api.headers("nasa", "mem-lead")
    mission_id = api.http.post(
        "/missions", json={"title": "Artemis", "plan": PILOT_PLAN}, headers=lead
    ).json()["id"]
    proposal = api.http.post(f"/missions/{mission_id}/match", headers=lead).json()
    row = proposal["slots"][0]
    api.http.post(f"/missions/{mission_id}/offers", headers=lead, json={"offers": [
        {"requirement_id": row["requirement_id"], "slot_index": row["slot_index"],
         "member_id": row["member_id"]}]})

    crew = api.headers("nasa", row["member_id"])
    view = api.http.get(f"/missions/{mission_id}", headers=crew).json()

    assert set(view) == {"id", "title", "state", "window", "site",
                         "crewing_status", "plan", "my_assignment"}
    assert view["my_assignment"]["member_id"] == row["member_id"]

    full = api.http.get(f"/missions/{mission_id}", headers=lead).json()
    assert {"created_by", "roster", "available_events", "blocked_events"} <= set(full)


def test_crew_only_see_missions_they_are_on(api):
    lead = api.headers("nasa", "mem-lead")
    api.http.post("/missions", json={"title": "Unrelated", "plan": PILOT_PLAN},
                  headers=lead)
    crew = api.headers("nasa", "mem-crew-05")
    assert api.http.get("/missions", headers=crew).json() == []


# ------------------------------------------------------------- the workflow


def test_the_whole_lifecycle_over_http(api):
    lead = api.headers("nasa", "mem-lead")
    director = api.headers("nasa", "mem-director")

    plan = {
        "requirements": [
            {"id": "r-pilot", "label": "Pilot",
             "skills": [{"skill_id": "pilot", "min_level": "PROFICIENT"}]},
            {"id": "r-surgeon", "label": "Flight Surgeon",
             "skills": [{"skill_id": "med", "min_level": "EXPERT"}]},
        ],
        "window": WINDOW, "site": "LC-39A",
    }
    mission_id = api.http.post("/missions", json={"title": "Artemis VII", "plan": plan},
                               headers=lead).json()["id"]

    proposal = api.http.post(f"/missions/{mission_id}/match", headers=lead).json()
    assert proposal["is_complete"]
    assert api.http.get(f"/missions/{mission_id}/roster", headers=lead).json() == [], \
        "the matcher creates nothing (§7.9)"

    offers = api.http.post(f"/missions/{mission_id}/offers", headers=lead, json={
        "offers": [{"requirement_id": r["requirement_id"],
                    "slot_index": r["slot_index"], "member_id": r["member_id"]}
                   for r in proposal["slots"]]}).json()
    for offer in offers:
        api.http.post(f"/assignments/{offer['id']}/accept",
                      headers=api.headers("nasa", offer["member_id"]))

    assert api.http.post(f"/missions/{mission_id}/events/submit",
                         headers=lead).json()["state"] == "pending_approval"
    assert api.http.post(f"/missions/{mission_id}/events/approve",
                         headers=lead).status_code == 403
    assert api.http.post(f"/missions/{mission_id}/events/approve",
                         headers=director).json()["state"] == "approved"

    audit = api.http.get("/audit", headers=director).json()
    approve = [e for e in audit if e["event"] == "approve"][-1]
    assert approve["actor_role"] == "director"
    assert approve["metadata"]["approved_roster"]


def test_a_crew_member_cannot_answer_someone_elses_offer(api):
    lead = api.headers("nasa", "mem-lead")
    mission_id = api.http.post("/missions", json={"title": "A", "plan": PILOT_PLAN},
                               headers=lead).json()["id"]
    proposal = api.http.post(f"/missions/{mission_id}/match", headers=lead).json()
    row = proposal["slots"][0]
    offer = api.http.post(f"/missions/{mission_id}/offers", headers=lead, json={
        "offers": [{"requirement_id": row["requirement_id"],
                    "slot_index": row["slot_index"], "member_id": row["member_id"]}]
    }).json()[0]

    other = "mem-crew-19" if row["member_id"] != "mem-crew-19" else "mem-crew-18"
    response = api.http.post(f"/assignments/{offer['id']}/accept",
                             headers=api.headers("nasa", other))
    assert response.status_code == 404, "someone else's record is missing, not forbidden"


def test_the_clock_can_be_pinned_so_expiry_is_demonstrable(api):
    """Offers last 72 hours; a test should not."""
    lead = api.headers("nasa", "mem-lead")
    mission_id = api.http.post("/missions", json={"title": "A", "plan": PILOT_PLAN},
                               headers=lead).json()["id"]
    proposal = api.http.post(f"/missions/{mission_id}/match", headers=lead).json()
    row = proposal["slots"][0]
    api.http.post(f"/missions/{mission_id}/offers", headers=lead, json={"offers": [
        {"requirement_id": row["requirement_id"], "slot_index": row["slot_index"],
         "member_id": row["member_id"]}]})

    later = {**lead, "X-Simulated-Now": "2099-02-28T00:00:00+00:00"}
    expired = api.http.post("/system/expire-offers", headers=later).json()
    assert [a["state"] for a in expired] == ["declined"]
