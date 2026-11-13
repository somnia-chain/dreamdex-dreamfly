import pytest

from flybot.cli import ROOT, SCRIPTS, main


def test_help_lists_the_commands(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("run", "site", "doctor", "warm", "extract"):
        assert cmd in out


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit) as e:
        main(["fly-away"])
    assert "unknown command" in str(e.value)


def test_every_script_command_points_at_a_real_script():
    for name in SCRIPTS.values():
        assert (ROOT / "scripts" / name).exists()
