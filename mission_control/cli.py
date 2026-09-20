"""Command line interface.

A thin layer over :mod:`mission_control.client`, which is itself a client of the
HTTP API — so running the CLI exercises the real server rather than a private
back door.

**The acting identity is sticky.** ``mc --profile nasa/lena`` switches who you
are and every later command uses it, in the style of ``kubectl use-context``.
The cost of that convenience is that a destructive command looks identical
whoever you are, so every mutating command echoes the actor it acted as::

    $ mc mission submit mis-0001
    [nasa/Lena Sorokin · mission_lead] mis-0001  draft → pending_approval

For scripts, ``MC_PROFILE=anya mc assignment accept asg-3`` applies for one
command **without** changing the stored current — a demo that rewrote your
interactive state would be rude.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import ApiError, MissionControl, make_plan, requirement

CONFIG_PATH = Path(os.environ.get("MC_CONFIG", Path.home() / ".mission-control.json"))
DEFAULT_URL = os.environ.get("MC_URL", "http://localhost:8000")


# ------------------------------------------------------------------- config


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {"url": DEFAULT_URL, "current": None, "identities": {}}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def remember(config: dict[str, Any], token_out: Any, *, make_current: bool = False) -> str:
    key = f"{token_out.tenant_id}/{token_out.member_id}"
    config["identities"][key] = {
        "tenant": token_out.tenant_id,
        "member_id": token_out.member_id,
        "name": token_out.name,
        "role": token_out.role,
        "token": token_out.token,
    }
    if make_current or config.get("current") is None:
        config["current"] = key
    return key


class ProfileError(Exception):
    pass


def resolve_profile(config: dict[str, Any], wanted: str) -> str:
    """Find one identity from a fragment: an id, a name, or ``tenant/either``.

    Names collide across tenants on purpose in the seed data — both
    organisations have a Lena Sorokin — so an ambiguous fragment is an error
    listing the candidates rather than a guess. That puts tenancy in front of
    you, which is the right place for it in a multi-tenant tool.
    """
    identities = config["identities"]
    if wanted in identities:
        return wanted

    tenant_hint, _, fragment = wanted.rpartition("/")
    fragment = fragment.lower()

    matches = [
        key
        for key, who in identities.items()
        if (not tenant_hint or who["tenant"] == tenant_hint)
        and (
            who["member_id"].lower() == fragment
            or fragment in who["name"].lower()
        )
    ]
    if not matches:
        raise ProfileError(
            f"no known identity matches {wanted!r}. "
            "Run 'mc profiles' to see what is known, or 'mc bootstrap'."
        )
    if len(matches) > 1:
        lines = "\n".join(
            f"    {k:28} {identities[k]['name']:20} {identities[k]['role']}"
            for k in sorted(matches)
        )
        raise ProfileError(
            f"{wanted!r} matches {len(matches)} identities:\n{lines}\n"
            f"  qualify it, e.g. mc --profile {sorted(matches)[0]}"
        )
    return matches[0]


def client_for(config: dict[str, Any], key: str | None) -> tuple[MissionControl, dict]:
    if key is None:
        raise ProfileError("no profile selected — run 'mc --profile <who>' first")
    who = config["identities"][key]
    return MissionControl(config.get("url", DEFAULT_URL), who["token"]), who


def acting(who: dict) -> str:
    return f"[{who['tenant']}/{who['name']} · {who['role']}]"


# ------------------------------------------------------------------ printing


def table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return "  (none)"
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    out = ["  " + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  " + "  ".join("─" * w for w in widths))
    out += [
        "  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        for row in rows
    ]
    return "\n".join(out)


def day(value: datetime | str) -> str:
    text = value.isoformat() if isinstance(value, datetime) else str(value)
    return text[:10]


# --------------------------------------------------------- requirement syntax

REQUIRE_RE = re.compile(
    r"^\s*(?P<label>[^:]+?)\s*(?:x(?P<count>\d+))?\s*(?P<optional>\(optional\))?\s*:\s*(?P<skills>.+)$",
    re.IGNORECASE,
)


def parse_requirement(spec: str):
    """``"Flight Surgeon x2 (optional): med>=EXPERT, phys>=PROFICIENT"``.

    All the skills listed belong to **one person** (§7.1); ``x2`` asks for two
    such people. Two different specifications are two ``--require`` flags.
    """
    match = REQUIRE_RE.match(spec)
    if not match:
        raise SystemExit(
            f"cannot parse --require {spec!r}\n"
            '  expected:  "Label [xN] [(optional)]: skill>=LEVEL[, skill>=LEVEL]"\n'
            '  example:   "Flight Surgeon: med>=EXPERT, phys>=PROFICIENT"'
        )
    skills = {}
    for part in match["skills"].split(","):
        skill, sep, level = part.partition(">=")
        if not sep:
            raise SystemExit(f"cannot parse skill {part!r}; expected skill>=LEVEL")
        skills[skill.strip()] = level.strip().upper()
    return requirement(
        match["label"].strip(),
        skills,
        count=int(match["count"] or 1),
        mandatory=not match["optional"],
    )


def parse_when(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------- commands


def cmd_serve(args, config) -> int:
    import uvicorn

    from .api import create_app, encode_token
    from .demo_seed import seed_platform
    from .store import Platform

    platform = Platform()
    app = create_app(
        platform,
        dev_tokens=args.dev_tokens or args.seed,
        allow_clock_override=args.allow_clock_override,
    )

    if args.seed:
        identities = seed_platform(platform)
        config.setdefault("identities", {})
        config["url"] = f"http://{args.host}:{args.port}"
        for tenant_id, member_id, name, role in identities:
            config["identities"][f"{tenant_id}/{member_id}"] = {
                "tenant": tenant_id, "member_id": member_id, "name": name,
                "role": role, "token": encode_token(tenant_id, member_id),
            }
        config["current"] = next(
            f"{t}/{m}" for t, m, _, r in identities if r == "mission_lead"
        )
        save_config(config)
        print(
            table(
                [[f"{t}/{m}", n, r] for t, m, n, r in identities],
                ["profile", "name", "role"],
            )
        )
        print(f"\nWrote {len(identities)} profiles to {CONFIG_PATH}")
        print(f"Current profile: {config['current']}")
        print("Dev token minting is ON (see mc token --help).\n")
        # uvicorn.run blocks and never returns, so anything still sitting in
        # the buffer is lost the moment output is redirected to a file.
        sys.stdout.flush()

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_profile(args, config) -> int:
    """``mc --profile <who>``: switch, and stay switched."""
    if not args.profile:
        current = config.get("current")
        if not current:
            print("no profile selected")
            return 1
        who = config["identities"][current]
        print(f"{current}  {who['name']} ({who['role']})")
        return 0
    key = resolve_profile(config, args.profile)
    config["current"] = key
    save_config(config)
    who = config["identities"][key]
    print(f"Now acting as {who['tenant']} / {who['name']} ({who['role']})")
    return 0


def cmd_profiles(args, config) -> int:
    current = config.get("current")
    rows = [
        ["*" if key == current else "", key, who["name"], who["role"]]
        for key, who in sorted(config["identities"].items())
    ]
    print(table(rows, ["", "profile", "name", "role"]))
    return 0


def cmd_whoami(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    me = mc.me()
    print(f"{me.tenant_id} ({me.tenant_name}) / {me.name}")
    print(f"  member    {me.member_id}")
    print(f"  role      {me.role}")
    print(f"  can       {', '.join(me.permissions)}")
    return 0


def cmd_bootstrap(args, config) -> int:
    mc = MissionControl(config.get("url", DEFAULT_URL))
    token = mc.bootstrap(args.tenant, args.name, args.director)
    key = remember(config, token, make_current=True)
    save_config(config)
    print(f"Created tenant {args.tenant} with director {token.member_id} ({token.name})")
    print(f"Now acting as {key}")
    return 0


def cmd_token(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    token = mc.token_for(args.member_id)
    key = remember(config, token, make_current=args.use)
    save_config(config)
    print(f"{acting(who)} saved profile {key}" + ("  (now current)" if args.use else ""))
    return 0


def cmd_member(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    if args.action == "list":
        print(table([[m.id, m.name, m.role, m.status] for m in mc.members()],
                    ["id", "name", "role", "status"]))
    elif args.action == "add":
        member = mc.add_member(args.name, args.role)
        line = f"{acting(who)} created {member.id} ({member.name}, {member.role})"
        if args.save_profile:
            key = remember(config, mc.token_for(member.id), make_current=args.use)
            save_config(config)
            line += f"\n  saved profile {key}"
        print(line)
    elif args.action == "deactivate":
        member = mc.deactivate_member(args.member_id, args.reason)
        print(f"{acting(who)} {member.id} is now {member.status}")
    return 0


def cmd_skill(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    if args.action == "list":
        print(table([[s.id, s.name] for s in mc.skills()], ["id", "name"]))
    else:
        skill = mc.add_skill(args.name, args.id)
        print(f"{acting(who)} defined skill {skill.id} ({skill.name})")
    return 0


def cmd_crew(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    if args.action == "show":
        profile = mc.crew_profile(args.member_id)
        print(f"{profile.name} ({profile.member_id})")
        print(table([[s.skill_id, s.level] for s in profile.skills], ["skill", "level"]))
        if profile.unavailability:
            print("  unavailable:")
            print(table(
                [[day(b.start), day(b.end), b.reason or ""] for b in profile.unavailability],
                ["from", "to", "reason"]))
    elif args.action == "skills":
        skills = dict(pair.split("=", 1) for pair in args.skill)
        profile = mc.set_skills(args.member_id, skills)
        print(f"{acting(who)} {profile.name}: "
              + ", ".join(f"{s.skill_id}={s.level}" for s in profile.skills))
    elif args.action == "unavailable":
        profile = mc.set_availability(
            args.member_id, [(parse_when(args.start), parse_when(args.end), args.reason)]
        )
        print(f"{acting(who)} {profile.name} unavailable "
              f"{day(args.start)}..{day(args.end)}")
    return 0


def cmd_mission(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))

    if args.action == "list":
        print(table(
            [[m.id, m.title, m.state, m.crewing_status, day(m.window.start)]
             for m in mc.missions()],
            ["id", "title", "state", "crewing", "starts"]))
        return 0

    if args.action == "create":
        plan = make_plan(
            [parse_requirement(spec) for spec in args.require],
            start=parse_when(args.start), end=parse_when(args.end), site=args.site,
        )
        mission = mc.create_mission(args.title, plan)
        print(f"{acting(who)} created {mission.id}  {mission.state}")
        return 0

    if args.action == "show":
        mission = mc.mission(args.mission_id)
        print(f"{mission.id}  {mission.title}")
        print(f"  state     {mission.state}")
        print(f"  crewing   {mission.crewing_status}")
        print(f"  window    {day(mission.window.start)} .. {day(mission.window.end)}")
        print(f"  site      {mission.site}")
        print(table(
            [[r.id, r.label, str(r.count), "yes" if r.mandatory else "no",
              ", ".join(f"{s.skill_id}>={s.min_level}" for s in r.skills)]
             for r in mission.plan.requirements],
            ["requirement", "label", "n", "mandatory", "skills"]))
        roster = getattr(mission, "roster", None)
        if roster is not None:
            print("  roster:")
            print(table(
                [[a.member_name or a.member_id, a.requirement_id, str(a.slot_index), a.state]
                 for a in roster],
                ["member", "requirement", "slot", "state"]))
            print(f"  you can   {', '.join(mission.available_events) or '(nothing)'}")
            for event, why in sorted(mission.blocked_events.items()):
                print(f"    {event}: {why}")
        elif mission.my_assignment:
            print(f"  your assignment: {mission.my_assignment.state} "
                  f"({mission.my_assignment.requirement_id})")
        return 0

    if args.action == "metadata":
        fields = {k: v for k, v in
                  (("title", args.title), ("description", args.description),
                   ("notes", args.notes)) if v is not None}
        mission = mc.update_metadata(args.mission_id, **fields)
        print(f"{acting(who)} {mission.id} metadata updated (still {mission.state})")
        return 0

    # everything else is a lifecycle event
    before = mc.mission(args.mission_id).state
    mission = mc.fire(args.mission_id, args.action, getattr(args, "reason", None))
    print(f"{acting(who)} {mission.id}  {before} → {mission.state}")
    return 0


def cmd_match(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    if args.action == "run":
        proposal = mc.run_matcher(args.mission_id)
        print(f"{acting(who)} proposal for {args.mission_id} "
              f"({'complete' if proposal.is_complete else 'INCOMPLETE'})")
        print(table(
            [[s.label, str(s.slot_index), s.member_name, str(s.rank), str(s.recent_load),
              "pinned" if s.pinned else "",
              ", ".join(f"{e.skill_id} {e.held}/{e.required}" for e in s.skills)]
             for s in proposal.slots],
            ["requirement", "slot", "member", "rank", "load", "", "evidence"]))
        for u in proposal.unfilled:
            print(f"  UNFILLED {u.label}#{u.slot_index} "
                  f"({'mandatory' if u.mandatory else 'optional'}): {u.reason}")
        if proposal.near_misses:
            print("  near misses:")
            print(table(
                [[n.member_name, n.requirement_id, n.filter, n.detail]
                 for n in proposal.near_misses[:8]],
                ["member", "requirement", "excluded by", "detail"]))
        for warning in proposal.team_warnings:
            print(f"  team constraint unmet: {warning}")
        return 0

    proposal = mc.run_matcher(args.mission_id)
    slots = None
    if args.slot:
        slots = [(s.split("#")[0], int(s.split("#")[1])) for s in args.slot]
    offers = mc.offer_proposal(args.mission_id, proposal, slots=slots)
    print(f"{acting(who)} offered {len(offers)} slot(s)")
    print(table(
        [[o.member_name or o.member_id, o.requirement_id, str(o.slot_index),
          o.offer_expires_at.isoformat(timespec="minutes")] for o in offers],
        ["member", "requirement", "slot", "expires"]))
    return 0


def cmd_assignment(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    if args.action == "list":
        found = mc.assignments(args.mission_id)
        print(table(
            [[a.id, a.mission_id, a.member_name or a.member_id, a.requirement_id, a.state]
             for a in found],
            ["id", "mission", "member", "requirement", "state"]))
        return 0
    method = mc.accept if args.action == "accept" else mc.decline
    assignment = method(args.assignment_id)
    print(f"{acting(who)} {assignment.id} → {assignment.state}")
    return 0


def cmd_audit(args, config) -> int:
    mc, _ = client_for(config, current_key(config, args))
    events = mc.audit()
    if args.mission_id:
        events = [
            e for e in events
            if e.subject_id == args.mission_id
            or e.metadata.get("mission_id") == args.mission_id
        ]
    print(table(
        [[e.at.isoformat(timespec="seconds"), e.event, e.subject_id,
          f"{e.from_state or '—'} → {e.to_state}", e.actor_role, e.reason or ""]
         for e in events],
        ["at", "event", "subject", "transition", "by", "reason"]))
    return 0


def cmd_system(args, config) -> int:
    mc, who = client_for(config, current_key(config, args))
    if args.action == "expire-offers":
        expired = mc.expire_offers()
        print(f"{acting(who)} expired {len(expired)} offer(s)")
    else:
        expired = mc.expire_approvals()
        print(f"{acting(who)} returned {len(expired)} mission(s) to draft")
    return 0


def cmd_demo(args, config) -> int:
    from .demo_seed import run_demo

    return run_demo(config.get("url", DEFAULT_URL))


# -------------------------------------------------------------------- wiring


def current_key(config: dict[str, Any], args) -> str | None:
    """Precedence: ``--token`` > ``MC_PROFILE`` > the stored current profile.

    The env var deliberately does not persist — scripts should not rewrite an
    interactive session's identity.
    """
    override = os.environ.get("MC_PROFILE")
    if override:
        return resolve_profile(config, override)
    return config.get("current")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mc",
        description="Mission Control — multi-tenant crewing platform.",
        epilog=(
            "Identity is sticky: 'mc --profile nasa/lena' switches actor and every "
            "later command uses it. MC_PROFILE=<who> overrides for one command "
            "without persisting."
        ),
    )
    parser.add_argument("--profile", metavar="WHO", nargs="?", const="",
                        help="switch acting identity (id, name, or tenant/either)")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--seed", action="store_true",
                       help="seed two tenants and save every profile")
    serve.add_argument("--dev-tokens", action="store_true",
                       help="allow minting tokens for other members (implied by --seed)")
    serve.add_argument("--allow-clock-override", action="store_true",
                       help="honour X-Simulated-Now, so expiry is demonstrable")
    serve.add_argument("--log-level", default="warning")
    serve.set_defaults(func=cmd_serve, needs_profile=False)

    profiles = sub.add_parser("profiles", help="list known identities")
    profiles.set_defaults(func=cmd_profiles, needs_profile=False)

    whoami = sub.add_parser("whoami", help="what the server says you are")
    whoami.set_defaults(func=cmd_whoami)

    boot = sub.add_parser("bootstrap", help="create a tenant and its first director")
    boot.add_argument("--tenant", required=True)
    boot.add_argument("--name", required=True)
    boot.add_argument("--director", required=True)
    boot.set_defaults(func=cmd_bootstrap, needs_profile=False)

    token = sub.add_parser("token", help="save a profile for another member")
    token.add_argument("member_id")
    token.add_argument("--use", action="store_true", help="switch to it as well")
    token.set_defaults(func=cmd_token)

    member = sub.add_parser("member", help="manage members")
    msub = member.add_subparsers(dest="action", required=True)
    msub.add_parser("list")
    add = msub.add_parser("add")
    add.add_argument("--name", required=True)
    add.add_argument("--role", required=True,
                     choices=["director", "mission_lead", "crew"])
    add.add_argument("--save-profile", action="store_true",
                     help="mint and store a profile for the new member")
    add.add_argument("--use", action="store_true")
    off = msub.add_parser("deactivate")
    off.add_argument("member_id")
    off.add_argument("--reason", required=True)
    member.set_defaults(func=cmd_member)

    skill = sub.add_parser("skill", help="manage the skill taxonomy")
    ssub = skill.add_subparsers(dest="action", required=True)
    ssub.add_parser("list")
    sadd = ssub.add_parser("add")
    sadd.add_argument("--name", required=True)
    sadd.add_argument("--id")
    skill.set_defaults(func=cmd_skill)

    crew = sub.add_parser("crew", help="crew profiles and availability")
    csub = crew.add_subparsers(dest="action", required=True)
    cshow = csub.add_parser("show")
    cshow.add_argument("member_id")
    cskills = csub.add_parser("skills")
    cskills.add_argument("member_id")
    cskills.add_argument("--skill", action="append", required=True, metavar="ID=LEVEL")
    cunav = csub.add_parser("unavailable")
    cunav.add_argument("member_id")
    cunav.add_argument("--start", required=True)
    cunav.add_argument("--end", required=True)
    cunav.add_argument("--reason")
    crew.set_defaults(func=cmd_crew)

    mission = sub.add_parser("mission", help="missions and their lifecycle")
    misub = mission.add_subparsers(dest="action", required=True)
    misub.add_parser("list")
    create = misub.add_parser("create")
    create.add_argument("--title", required=True)
    create.add_argument("--start", required=True, metavar="ISO")
    create.add_argument("--end", required=True, metavar="ISO")
    create.add_argument("--site", required=True)
    create.add_argument(
        "--require", action="append", required=True, metavar="SPEC",
        help='e.g. "Flight Surgeon x2 (optional): med>=EXPERT, phys>=PROFICIENT"')
    show = misub.add_parser("show")
    show.add_argument("mission_id")
    meta = misub.add_parser("metadata")
    meta.add_argument("mission_id")
    meta.add_argument("--title")
    meta.add_argument("--description")
    meta.add_argument("--notes")
    for event in ("submit", "reopen", "approve", "activate", "complete",
                  "cancel", "abort", "request_changes"):
        ev = misub.add_parser(event)
        ev.add_argument("mission_id")
        ev.add_argument("--reason")
    mission.set_defaults(func=cmd_mission)

    match = sub.add_parser("match", help="run the matcher and make offers")
    masub = match.add_subparsers(dest="action", required=True)
    run = masub.add_parser("run")
    run.add_argument("mission_id")
    offer = masub.add_parser("offer")
    offer.add_argument("mission_id")
    offer.add_argument("--slot", action="append", metavar="REQ#N",
                       help="offer only these slots; default is all unpinned")
    match.set_defaults(func=cmd_match)

    assignment = sub.add_parser("assignment", help="offers and responses")
    asub = assignment.add_subparsers(dest="action", required=True)
    alist = asub.add_parser("list")
    alist.add_argument("--mission-id", dest="mission_id")
    for verb in ("accept", "decline"):
        av = asub.add_parser(verb)
        av.add_argument("assignment_id")
    assignment.set_defaults(func=cmd_assignment)

    audit = sub.add_parser("audit", help="the event log")
    audit.add_argument("--mission-id", dest="mission_id")
    audit.set_defaults(func=cmd_audit)

    system = sub.add_parser("system", help="clock-driven sweeps")
    sysub = system.add_subparsers(dest="action", required=True)
    sysub.add_parser("expire-offers")
    sysub.add_parser("expire-approvals")
    system.set_defaults(func=cmd_system)

    demo = sub.add_parser("demo", help="run the end-to-end scenario against the server")
    demo.set_defaults(func=cmd_demo, needs_profile=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()

    try:
        # `mc --profile X` with no subcommand switches and stops.
        if args.profile is not None and args.command is None:
            return cmd_profile(args, config)
        if args.profile:
            config["current"] = resolve_profile(config, args.profile)
            save_config(config)
        if args.command is None:
            parser.print_help()
            return 1
        return args.func(args, config)
    except ProfileError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except ApiError as error:
        print(f"error: {error.code}: {error.message}", file=sys.stderr)
        if error.available:
            print(f"  you can: {', '.join(error.available)}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: cannot reach the server — {error}", file=sys.stderr)
        print("  is 'mc serve' running?", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
