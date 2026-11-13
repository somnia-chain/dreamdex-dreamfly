import json
import re
from pathlib import Path

from flybot.live.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "config" / "launcher.json").read_text())
READ_ELSEWHERE = {"STATE_DIR", "PORT", "DASHBOARD_PASSWORD"}      # read by the supervisor and the dashboard, not Settings


def envs_read_by_settings() -> set[str]:
    src = (ROOT / "flybot" / "live" / "settings.py").read_text()
    return set(re.findall(r'e\.get\("([A-Z_]+)"\)', src))


def test_every_catalog_param_is_a_setting_the_kit_reads_and_the_other_way_round():
    listed = {p["env"] for p in CATALOG["params"]}
    assert listed - READ_ELSEWHERE == envs_read_by_settings()
    for name in READ_ELSEWHERE:
        assert name in (ROOT / "flybot" / "live" / "supervisor.py").read_text() + (ROOT / "scripts" / "serve_site.py").read_text()


def test_catalog_defaults_are_the_kit_defaults_and_make_a_valid_dry_run():
    env = {p["env"]: str(p["default"]) for p in CATALOG["params"] if str(p["default"]) != ""}
    s = Settings.from_env(env)                     # the placeholder key and the defaults must pass validation
    assert s.dry_run and not s.has_key and s.network == "testnet" and s.brain == "delta"
    assert s.live_stake_usdso == 20.0 and s.max_open_orders == 6 and s.concurrent_episodes == 3
    assert s.markets == Settings().markets
    for p in CATALOG["params"]:
        if p["type"] == "select":
            assert str(p["default"]) in p["options"]
