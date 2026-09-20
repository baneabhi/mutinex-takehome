# Mission Control

A multi-tenant platform for crewing missions. Organisations are tenants; members
hold one of three roles; missions move through an approval lifecycle; an
auto-matching engine suggests crew based on skills, availability and constraints.

**Design document: [docs/DESIGN.md](docs/DESIGN.md).** Section references in the
code (`§6.2`, `§7.5`, …) point at it.

Python 3.12+. The domain is standard library only; SciPy backs the matching solver
and FastAPI the HTTP layer — see [dependencies](#dependencies).

## Install

```bash
pip install -e ".[dev]"
python -m pytest tests/       # 586 tests, ~2s
```

That puts `mc` on your PATH. If you would rather not install, `python -m
mission_control.cli` is the same thing and works everywhere `mc` appears below.

## Three ways to run it

```bash
python demo.py        # 1. in-process walkthrough — no server, no setup
mc serve --seed       # 2. API on :8000, Swagger at /docs, 48 identities seeded
mc demo               # 3. the same walkthrough over HTTP (needs 2 running)
```

**`python demo.py`** is the fastest look: it drives the domain directly and narrates
the matcher, the approval gate and tenant isolation. Nothing to start.

**`mc serve --seed`** is the one to use for anything interactive.

## Layout

| File | |
|---|---|
| [domain.py](mission_control/domain.py) | ids, enums, frozen value objects, errors, availability, the eligibility predicate |
| [authz.py](mission_control/authz.py) | permissions, the role matrix, the three contextual policies |
| [lifecycle.py](mission_control/lifecycle.py) | both transition tables, `apply()`, `available_events()`, the approval gate |
| [matching.py](mission_control/matching.py) | candidate generation, ranking, team assembly, team constraints, explanation |
| [solver.py](mission_control/solver.py) | SciPy sparse min-weight bipartite matching, plus a greedy baseline used only by a test |
| [store.py](mission_control/store.py) | `Platform`, `Tenant`, `TenantWorkspace`, `TenantSnapshot` |
| [services.py](mission_control/services.py) | the callable operations, bundled per-actor as a `Scope` |
| [schemas.py](mission_control/schemas.py) | the wire contract, shared by server and client |
| [api.py](mission_control/api.py) | FastAPI: auth dependency, routes, error mapping |
| [client.py](mission_control/client.py) | typed Python client over HTTP |
| [cli.py](mission_control/cli.py) | `mc` — a client of the client |

`matching` and `lifecycle` reach storage through a Protocol rather than an
import, so they depend on `domain` alone and are testable without constructing a
platform. The core imports none of FastAPI, httpx or pydantic — there is a test
that blocks them and imports `services` to prove it.

## Using the CLI

### Start the server

```bash
$ mc serve --seed
  profile              name            role
  ───────────────────  ──────────────  ────────────
  nasa/mem-director    Dana Whitfield  director
  nasa/mem-lead        Lena Sorokin    mission_lead
  nasa/mem-crew-00     Anya Petrova    crew
  …                                                  (48 in two tenants)

Wrote 48 profiles to ~/.mission-control.json
Current profile: nasa/mem-lead
Dev token minting is ON (see mc token --help).
```

Two tenants, twenty crew each, every identity written to the CLI's config — so in a
second terminal you can start working immediately, with no tokens to copy.

### Who you are is sticky

```bash
$ mc whoami
nasa (NASA) / Lena Sorokin
  member    mem-lead
  role      mission_lead
  can       assignment_offer, crew_profile_read_all, matcher_run, mission_activate, …

$ mc --profile nasa/dana          # switch — and stay switched
Now acting as nasa / Dana Whitfield (director)
```

Resolve by id, by name, or `tenant/either`. A bare name that exists in both tenants is
an error rather than a guess, because guessing which organisation you meant is the
mistake this system exists to prevent:

```
$ mc --profile lena
error: 'lena' matches 2 identities:
    esa/mem-lead     Lena Sorokin    mission_lead
    nasa/mem-lead    Lena Sorokin    mission_lead
```

`MC_PROFILE=nasa/dana mc …` acts as someone for **one command** without changing what's
stored — that's what scripts should use.

### A whole mission

```bash
$ mc --profile nasa/lena

$ mc mission create --title "Artemis VII" \
      --start 2099-03-01 --end 2099-03-21 --site LC-39A \
      --require "Pilot: pilot>=PROFICIENT" \
      --require "Flight Surgeon: med>=EXPERT" \
      --require "Engineer x2: eng>=EXPERT" \
      --require "Geologist (optional): geo>=MASTER"
[nasa/Lena Sorokin · mission_lead] created mis-0001  draft
```

The `--require` syntax is `"Label [xN] [(optional)]: skill>=LEVEL[, skill>=LEVEL]"`.
Several skills in **one** `--require` means one person holding all of them; `x2` means
two people at the same specification; two different specifications are two flags.

```bash
$ mc match run mis-0001
[nasa/Lena Sorokin · mission_lead] proposal for mis-0001 (complete)
  requirement     slot  member        rank  load  evidence
  ──────────────  ────  ────────────  ────  ────  ───────────────────────
  Pilot           0     Boris Novak   1     0     pilot EXPERT/PROFICIENT
  Flight Surgeon  0     Anya Petrova  0     0     med EXPERT/EXPERT
  Geologist       0     Nils Haugen   0     0     geo MASTER/MASTER
```

Note Boris takes Pilot at **rank 1** while Anya is rank 0 for it — the solver pays a
worse pilot to keep the only eligible Flight Surgeon available. That's §7.6, live.
Running the matcher creates nothing; offering is a separate, human act:

```bash
$ mc match offer mis-0001
[nasa/Lena Sorokin · mission_lead] offered 3 slot(s)

$ MC_PROFILE=nasa/mem-crew-01 mc assignment accept asg-0002
[nasa/Boris Novak · crew] asg-0002 → accepted
$ MC_PROFILE=nasa/mem-crew-00 mc assignment accept asg-0003
[nasa/Anya Petrova · crew] asg-0003 → accepted

$ mc mission submit mis-0001
[nasa/Lena Sorokin · mission_lead] mis-0001  draft → pending_approval

$ mc mission approve mis-0001
error: MISSING_PERMISSION: a mission_lead cannot mission_approve

$ MC_PROFILE=nasa/dana mc mission approve mis-0001
[nasa/Dana Whitfield · director] mis-0001  pending_approval → approved
```

`mc mission show` tells you what you can do next **and why you can't yet**:

```
  you can   cancel, plan_edited, run_matcher, submit
    cancel: a reason is required
    submit: mission is awaiting_responses; every mandatory slot must be ACCEPTED first
```

And a crew member sees a different, smaller thing — no owner, no approver, no roster
beyond their own row:

```bash
$ MC_PROFILE=nasa/mem-crew-00 mc mission show mis-0001
mis-0001  Artemis VII
  state     approved
  crewing   fully_crewed
  …
  your assignment: accepted (r-flight-surgeon-1)
```

### Starting from an empty server

`mc serve` without `--seed` starts empty. `bootstrap` is the only unauthenticated route
— a tenant's first Director cannot be created by a Director:

```bash
$ mc serve                                    # terminal 1, no --seed

$ mc bootstrap --tenant acme --name "Acme Space" --director "Ada Rowe"
Created tenant acme with director mem-director (Ada Rowe)
Now acting as acme/mem-director

$ mc skill add --id pilot --name Piloting
[acme/Ada Rowe · director] defined skill pilot (Piloting)

$ mc member add --name "Lena Sorokin" --role mission_lead --save-profile
[acme/Ada Rowe · director] created mem-0001 (Lena Sorokin, mission_lead)
  saved profile acme/mem-0001

$ mc member add --name "Anya Petrova" --role crew --save-profile
$ mc crew skills mem-0002 --skill pilot=MASTER
[acme/Ada Rowe · director] Anya Petrova: pilot=MASTER

$ mc --profile acme/lena                      # only a Lead can author a mission
Now acting as acme / Lena Sorokin (mission_lead)

$ mc mission create --title "First Flight" --start 2099-05-01 --end 2099-05-10 \
      --site Pad-1 --require "Pilot: pilot>=PROFICIENT"
[acme/Lena Sorokin · mission_lead] created mis-0003  draft
```

Two things that trip people up, both of them the design working as intended:

- **Do the Director's work before switching.** Defining skills, adding members and
  correcting anyone's profile all need `MEMBER_MANAGE`, which only Directors hold —
  so `--save-profile --use` on the Lead would leave you unable to add the next member.
  Directors and Leads have disjoint permissions on purpose (§5.2).
- **Ids are sequential per tenant across every kind of object**, so the first mission
  here is `mis-0003` — `mem-0001` and `mem-0002` took the first two.

`--save-profile` mints and stores a token for the new member, so you never handle one;
add `--use` to switch to them at the same time.

### Command reference

| | |
|---|---|
| `mc serve [--seed] [--port] [--dev-tokens] [--allow-clock-override]` | run the API |
| `mc --profile <who>` · `mc profiles` · `mc whoami` | identity |
| `mc bootstrap` · `mc token <member>` | create a tenant, mint a token |
| `mc member add\|list\|deactivate` · `mc skill add\|list` | organisation (Director) |
| `mc crew show\|skills\|unavailable` | profiles and availability |
| `mc mission create\|list\|show\|metadata` | missions |
| `mc mission submit\|reopen\|approve\|activate\|complete\|cancel\|abort\|request_changes` | transitions |
| `mc match run\|offer` | the matcher, and acting on it |
| `mc assignment list\|accept\|decline` | offers and responses |
| `mc audit [--mission-id]` · `mc system expire-offers\|expire-approvals` | log and sweeps |

`mc <command> --help` for any of them.

## Using the API directly

Swagger UI is at **<http://localhost:8000/docs>** with every route, schema and example.

Authenticate with `Authorization: Bearer <token>`, where a token is
`base64url("tenant:member")`. With `--seed` the tokens are deterministic, so you can
write them by hand:

```bash
$ python3 -c "from mission_control.api import encode_token; print(encode_token('nasa','mem-lead'))"
bmFzYTptZW0tbGVhZA

$ curl -s localhost:8000/me -H "Authorization: Bearer bmFzYTptZW0tbGVhZA"
{"tenant_id":"nasa","tenant_name":"NASA","member_id":"mem-lead",
 "name":"Lena Sorokin","role":"mission_lead","permissions":[...]}
```

**No route contains a tenant id** — it's `/missions`, not `/tenants/nasa/missions`.
Scope comes from the token, so there's no argument through which to ask for someone
else's data. Reading another tenant's id gives a 404, not a 403, because a 403 confirms
the record exists:

```bash
$ curl -s localhost:8000/missions/mis-0001 -H "Authorization: Bearer ZXNhOm1lbS1sZWFk"
{"error":"NOT_FOUND","message":"mission 'mis-0001' not found","kind":"mission","id":"mis-0001"}
  HTTP 404
```

Errors carry a machine-readable code and, where there is one, a way forward:

```bash
$ curl -s -X POST localhost:8000/missions/mis-0001/events/activate \
       -H "Authorization: Bearer bmFzYTptZW0tbGVhZA"
{"error":"ILLEGAL_TRANSITION",
 "message":"cannot activate a draft mission; you can: cancel, plan_edited, run_matcher, submit",
 "state":"draft","attempted":"activate",
 "available":["cancel","plan_edited","run_matcher","submit"]}
  HTTP 409
```

Every mission transition goes through one endpoint,
`POST /missions/{id}/events/{event}`. The transition table is the source of truth and a
`GET` returns `available_events`, so clients discover what's postable rather than
hard-coding it.

| | |
|---|---|
| `NotFound` | **404** — including foreign-tenant ids |
| `Denied` | **403** — `error` is the policy code (`SELF_APPROVAL` ≠ `MISSING_PERMISSION`) |
| `IllegalTransition` | **409** — carries `available` |
| `GuardFailed` | **409** — carries `attempted` and `reason` |

### From Python

```python
from mission_control.client import MissionControl, make_plan, requirement

lead = MissionControl("http://localhost:8000", token="bmFzYTptZW0tbGVhZA")
mission = lead.create_mission("Artemis VII", make_plan(
    [requirement("Pilot", {"pilot": "PROFICIENT"})],
    start=..., end=..., site="LC-39A"))

proposal = lead.run_matcher(mission.id)
lead.offer_proposal(mission.id, proposal)
```

Responses parse into the same models the server serialises from, so the contract is
defined once. `lead.as_actor(token)` gives a client for another identity over the same
connection.

See [§13](docs/DESIGN.md) for the auth model in full — including that authentication is
a deliberate stub while authorisation is not.

## The load-bearing ideas

Each of these replaces a runtime check with something a bad state cannot be
expressed in.

- **A tenant is a sandbox object** in `dict[TenantId, Tenant]`, with no global
  collections. No service function takes a `tenant_id`; scope resolves once from
  the authenticated `Actor`. There is no cross-tenant query to get wrong.
- **Directors govern, Leads plan, Crew execute** — disjoint permission sets. A
  Director holds no authoring permissions, so *a Director cannot approve their
  own mission because there is no such mission*. The `SeparationOfDuties` policy
  is kept as a backstop for imported data, not as the mechanism.
- **The plan is editable in `DRAFT` and nowhere else.** Plan fields live in a
  frozen `MissionPlan`, so the only way to change one is to replace it — and the
  transition table has exactly one row that can. Submitted and approved missions
  are reached through an explicit `reopen`. "You cannot approve something that
  changed under you" holds for the strongest available reason: while a Director
  is reading a plan, it cannot change at all.
- **Crewing completes before approval.** A mission can't be submitted until
  every mandatory slot is `ACCEPTED`, so a Director approves a committed crew
  rather than a hopeful one. This deletes ranked alternates in the plan, an
  `at_risk` flag, backfill proposals, and most post-approval invalidation.
- **Matching solves a global assignment problem**, not per-slot greedy. Greedy
  strands slots whose only eligible candidate was taken elsewhere — a
  *feasibility* result, independent of ranking, which is why there being no
  preference model doesn't weaken it.
- **No scoring.** Clearing a requirement is binary; eligible crew order by
  `(load, last flew, id)`; the solver's cost is that rank. The requirement *is*
  the statement of suitability, so preferring surplus is a soft preference
  needing a weight nobody can source.
- **Derive, don't store.** Crewing status, availability and the invalidation
  report are computed on read, so there is no cached projection to invalidate.

## Dependencies

| | why |
|---|---|
| `scipy`, `numpy` | the matching solver — [see below](#why-sparse-matching) |
| `fastapi`, `uvicorn`, `httpx` | the HTTP layer (§13) |

The domain imports none of them. `domain`, `authz`, `lifecycle`, `matching`, `store`
and `services` are standard library apart from NumPy/SciPy reaching `solver` — there is
a test that blocks FastAPI, httpx and pydantic from `sys.meta_path` and imports the core
to prove it.

## Why sparse matching

The solver uses
`scipy.sparse.csgraph.min_weight_full_bipartite_matching` rather than the better-known
`scipy.optimize.linear_sum_assignment`.

**Infeasibility is not the discriminator.** Both raise when the slots cannot all be
filled — `linear_sum_assignment` gives `ValueError: cost matrix is infeasible`,
`min_weight_full_bipartite_matching` gives `ValueError: no full matching exists`. §7.5
needs unfillable slots *reported* with a reason, so either way we supply an escape: one
extra column per slot, reachable only by that slot, priced above any real assignment. A
slot taking its own column is one nobody could fill. Priced that way, it also makes the
solver fill as many slots as possible *first* and only then minimise cost, which is
§7.6's point.

What differs is the cost of that escape, and how the work scales:

- **A dense matrix has no absent cell.** Every impossible pair still carries a number
  the solver considers — 23 M of the 40 M cells at our target shape — so its work grows
  with the **roster**, not with how many people are actually eligible. Holding eligible
  pairs fixed at 120 000 and growing the roster with people who qualify for nothing:
  dense goes 1 ms → 74 ms as the roster goes 300 → 64 000; sparse goes 15 ms → 25 ms.
- **Sparse is what makes the pruning possible.** Each slot only needs its `R` cheapest
  candidates (§7.5 has the proof). In a dense matrix you cannot drop a cell — the solver
  still walks rows × columns — so that reduction has no dense equivalent. It is where
  most of the speed actually comes from.
- **Memory.** 320 MB for a 2000 × 20 000 float64 matrix; 2 GB at 5000 × 50 000.

Measured on identical input — 2000 slots, 20 000 crew, 16.8 M eligible pairs of 40 M
cells:

| | time | |
|---|---:|---|
| dense + sentinel | 41 s | 320 MB |
| sparse, all edges | 20 s | |
| sparse + `R`-cheapest pruning | **4.2 s** | 4 M edges |

**The crossover is a few thousand crew.** Below it dense is *faster*, because building
the sparse structure costs more than the cells it saves. This is a scale-driven choice,
not an unconditional one: at the tens-of-slots scale the design originally targeted,
`linear_sum_assignment` was the right call.

**What it costs.** ~130 MB with NumPy and 270 ms of import before any work happens. And
tie-breaking is no longer ours: equally optimal teams may differ across SciPy versions,
which matters because `count: 2` makes ties common. The pure-Python Hungarian this
replaced is kept at [tests/reference_solver.py](tests/reference_solver.py) — SciPy can't
be its own oracle, and it doubles as a working fallback.

## Places the code departs from the design document

Each is commented at the site, with the reasoning.

1. **`MissionPlan.roster`** is derived rather than a field. §6.5 types it as
   `tuple[Assignment, ...]`; since an `Assignment` changes state when crew
   respond, holding them would make every acceptance a plan edit and bounce the
   mission to `DRAFT` — contradicting §8.3 and making crewing impossible. The
   `approve` event records the approved members instead.
2. **`run_matcher` creates nothing**, per §7.9 rather than §6.2's shorthand.
3. **`approve` also guards on crewing status**, not just roster validity: a
   member deactivated between submit and approve leaves a *hole* rather than a
   stale slot, which a roster-only check would let through.
4. **`abort` is scoped by ownership.** §6.2 leaves its policy column blank,
   which would let any Lead abort another's active mission when `cancel` — the
   same act before launch — does not.
5. **A decline excludes that member from the mission's candidate set.** §7.3
   lists four filters and not this one; without it a decline is information the
   engine discards, so re-running the matcher proposes the same person for the
   same slot forever. A Lead can still offer to them by hand.
6. **The solver is sparse, with one "unfilled" column per slot**, rather than
   §7.5's dense matrix with zero-cost dummy padding. Zero-cost padding would be
   *cheaper* than a real person, so the solver would prefer it; and a dense
   matrix makes the solver's work grow with the roster rather than with the
   eligible pairs. See [why sparse matching](#why-sparse-matching).
7. **Near-misses are capped per requirement** (`near_miss_depth`, default 10).
   §7.9 doesn't cap them; at 20 crew that's fine, but "excluded by exactly one
   filter" describes most of the roster at 5 000, and one match was producing
   154 000 rows. They're ordered by how actionable the exclusion is — booked
   before under-qualified, and closest-to-the-bar first — so the rows dropped
   are the ones nobody would read.
8. **Each slot keeps only its `R` cheapest candidates**, `R` being the number of
   slots. Lossless, not a heuristic (§7.5 has the swapping argument), and where
   most of the speed comes from: 20 s to 4.2 s on the solve at 2000 slots. It
   has no dense equivalent, which is the main reason the solver is sparse.

## Known limits

Carried over from §11, and unchanged by the implementation: acceptance is
irreversible in both directions — neither a crew member nor a Lead can undo a
booking without a material plan change, and `release_crew` is deferred to a
later version (§11.4); team constraints
are a repair heuristic rather than CP-SAT; there is no cross-mission
optimisation; and the pre-submission window is unbounded, so a Lead can hold
crew behind a fully-crewed draft. `SystemService.long_held_drafts` sketches the
reporting side of that last one.
