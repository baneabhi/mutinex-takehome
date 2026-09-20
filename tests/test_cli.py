"""The CLI.

Two things worth testing here and not much else: the profile model, because a
sticky actor is state a user can get wrong, and the ``--require`` syntax,
because it is the one place the CLI parses something non-trivial. The commands
themselves are thin, so they get a smoke pass over the real stack.
"""

from __future__ import annotations

import pytest

from mission_control.cli import ProfileError, main, parse_requirement, resolve_profile


def run(cli_env, *argv, profile=None, monkeypatch=None):
    if profile is not None:
        monkeypatch.setenv("MC_PROFILE", profile)
    elif monkeypatch is not None:
        monkeypatch.delenv("MC_PROFILE", raising=False)
    return main(list(argv))


# ------------------------------------------------------------------ profiles


def test_a_profile_resolves_by_id_or_by_name(cli_env):
    config = cli_env.load_config()
    assert resolve_profile(config, "nasa/mem-lead") == "nasa/mem-lead"
    assert resolve_profile(config, "nasa/lena") == "nasa/mem-lead"
    assert resolve_profile(config, "nasa/Lena Sorokin") == "nasa/mem-lead"
    assert resolve_profile(config, "esa/dana") == "esa/mem-director"


def test_a_name_that_exists_in_both_tenants_is_an_error(cli_env):
    """The seed data collides names across tenants on purpose. Guessing which
    organisation you meant is exactly the mistake this system is built to
    prevent, so the CLI refuses and shows both."""
    config = cli_env.load_config()
    with pytest.raises(ProfileError) as caught:
        resolve_profile(config, "lena")
    assert "matches 2 identities" in str(caught.value)
    assert "nasa/mem-lead" in str(caught.value)
    assert "esa/mem-lead" in str(caught.value)


def test_an_unknown_profile_is_an_error_not_an_invention(cli_env):
    with pytest.raises(ProfileError):
        resolve_profile(cli_env.load_config(), "nobody")


def test_switching_profile_sticks(cli_env, capsys, monkeypatch):
    run(cli_env, "--profile", "nasa/dana", monkeypatch=monkeypatch)
    assert cli_env.load_config()["current"] == "nasa/mem-director"

    run(cli_env, "whoami", monkeypatch=monkeypatch)
    assert "Dana Whitfield" in capsys.readouterr().out

    run(cli_env, "--profile", "esa/lena", monkeypatch=monkeypatch)
    assert cli_env.load_config()["current"] == "esa/mem-lead"


def test_the_env_override_does_not_persist(cli_env, capsys, monkeypatch):
    """Scripts should not rewrite an interactive session's identity."""
    run(cli_env, "--profile", "nasa/lena", monkeypatch=monkeypatch)
    before = cli_env.load_config()["current"]

    run(cli_env, "whoami", profile="nasa/dana", monkeypatch=monkeypatch)
    assert "Dana Whitfield" in capsys.readouterr().out
    assert cli_env.load_config()["current"] == before


def test_mutating_commands_say_who_they_acted_as(cli_env, capsys, monkeypatch):
    """A sticky actor makes 'who am I right now' invisible at the moment it
    matters, so every mutation prints it."""
    run(cli_env, "--profile", "nasa/lena", monkeypatch=monkeypatch)
    capsys.readouterr()
    run(cli_env, "mission", "create", "--title", "Artemis VII",
        "--start", "2099-03-01", "--end", "2099-03-21", "--site", "LC-39A",
        "--require", "Pilot: pilot>=PROFICIENT", monkeypatch=monkeypatch)
    out = capsys.readouterr().out
    assert "[nasa/Lena Sorokin · mission_lead]" in out


# --------------------------------------------------------- requirement syntax


@pytest.mark.parametrize(
    "spec,label,count,mandatory,skills",
    [
        ("Pilot: pilot>=PROFICIENT", "Pilot", 1, True, {"pilot": "PROFICIENT"}),
        ("Engineer x3: eng>=EXPERT", "Engineer", 3, True, {"eng": "EXPERT"}),
        ("Observer (optional): med>=PROFICIENT", "Observer", 1, False,
         {"med": "PROFICIENT"}),
        ("Surgeon x2 (optional): med>=EXPERT, geo>=NOVICE", "Surgeon", 2, False,
         {"med": "EXPERT", "geo": "NOVICE"}),
        ("Flight Surgeon: med>=expert", "Flight Surgeon", 1, True, {"med": "EXPERT"}),
    ],
)
def test_requirement_specs_parse(spec, label, count, mandatory, skills):
    parsed = parse_requirement(spec)
    assert parsed.label == label
    assert parsed.count == count
    assert parsed.mandatory is mandatory
    assert {s.skill_id: s.min_level for s in parsed.skills} == skills


def test_several_skills_in_one_spec_mean_one_person(cli_env):
    """Not two requirements — one person holding both (§7.1)."""
    parsed = parse_requirement("Flight Surgeon: med>=EXPERT, geo>=PROFICIENT")
    assert parsed.count == 1
    assert len(parsed.skills) == 2


@pytest.mark.parametrize("spec", ["no colon here", "Pilot: pilot", "Pilot:"])
def test_a_malformed_spec_explains_the_syntax(spec):
    with pytest.raises(SystemExit) as caught:
        parse_requirement(spec)
    assert "expected" in str(caught.value) or "skill>=LEVEL" in str(caught.value)


# ---------------------------------------------------------------- smoke pass


def test_a_whole_mission_through_the_cli(cli_env, capsys, monkeypatch):
    run(cli_env, "--profile", "nasa/lena", monkeypatch=monkeypatch)
    run(cli_env, "mission", "create", "--title", "Artemis VII",
        "--start", "2099-03-01", "--end", "2099-03-21", "--site", "LC-39A",
        "--require", "Pilot: pilot>=PROFICIENT",
        "--require", "Flight Surgeon: med>=EXPERT", monkeypatch=monkeypatch)
    mission_id = capsys.readouterr().out.split("created ")[1].split()[0]

    run(cli_env, "match", "offer", mission_id, monkeypatch=monkeypatch)
    assert "offered 2 slot(s)" in capsys.readouterr().out

    from mission_control.client import MissionControl

    lead = cli_env.MissionControl("http://test", cli_env.load_config()
                                  ["identities"]["nasa/mem-lead"]["token"])
    assert isinstance(lead, MissionControl)
    for assignment in lead.assignments(mission_id):
        run(cli_env, "assignment", "accept", assignment.id,
            profile=f"nasa/{assignment.member_id}", monkeypatch=monkeypatch)
    capsys.readouterr()

    run(cli_env, "mission", "submit", mission_id, monkeypatch=monkeypatch)
    assert "draft → pending_approval" in capsys.readouterr().out

    run(cli_env, "mission", "approve", mission_id, monkeypatch=monkeypatch)
    assert "MISSING_PERMISSION" in capsys.readouterr().err

    run(cli_env, "mission", "approve", mission_id, profile="nasa/dana",
        monkeypatch=monkeypatch)
    assert "pending_approval → approved" in capsys.readouterr().out


def test_an_api_error_prints_the_way_out(cli_env, capsys, monkeypatch):
    run(cli_env, "--profile", "nasa/lena", monkeypatch=monkeypatch)
    run(cli_env, "mission", "create", "--title", "A", "--start", "2099-03-01",
        "--end", "2099-03-21", "--site", "X", "--require", "Pilot: pilot>=PROFICIENT",
        monkeypatch=monkeypatch)
    mission_id = capsys.readouterr().out.split("created ")[1].split()[0]

    code = run(cli_env, "mission", "activate", mission_id, monkeypatch=monkeypatch)
    captured = capsys.readouterr()
    assert code == 1
    assert "ILLEGAL_TRANSITION" in captured.err
    assert "you can: cancel, plan_edited, run_matcher, submit" in captured.err
