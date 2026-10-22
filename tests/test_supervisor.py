import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from flybot.live.settings import Settings, SettingsError
from flybot.live.supervisor import Job, Supervisor, build_jobs, seed_state, state_dir, state_name

ROOT = Path(__file__).resolve().parents[1]


def names(jobs):
    return [j.name for j in jobs]


def test_state_lives_on_the_mounted_volume_unless_told_otherwise(tmp_path):
    # A hosted container whose variables were pasted over without STATE_DIR must still keep its brains on the volume.
    volume = tmp_path / "data"
    assert state_dir({}, root=tmp_path, volume=volume) == tmp_path / "state" / "live"   # no volume: in the repo
    volume.mkdir()
    assert state_dir({}, root=tmp_path, volume=volume) == volume / "state"
    assert state_dir({"STATE_DIR": "/elsewhere"}, root=tmp_path, volume=volume) == Path("/elsewhere")
    assert state_dir({"STATE_DIR": ""}, root=tmp_path, volume=volume) == volume / "state"   # blank counts as unset


def test_one_spot_market_starts_only_the_runner_and_the_dashboard(tmp_path):
    s = Settings(markets=("WETH:USDso",))
    jobs = build_jobs(s, ROOT, tmp_path, 8123)
    assert names(jobs) == ["runner", "dashboard"]
    runner = jobs[0].argv
    assert runner[runner.index("--symbols") + 1:][:1] == ["WETH:USDso"] and "--perp-tape" not in runner
    assert runner[runner.index("--brain") + 1] == "delta"                      # the kit default
    assert build_jobs(Settings(markets=("WETH:USDso",), brain="mb"), ROOT, tmp_path, 8123)[0].argv[-1] == "mb"
    assert runner[runner.index("--network") + 1] == "testnet"
    assert jobs[1].argv[-1] == "8123"


def test_perps_and_event_contracts_add_the_recorders_and_the_tapes(tmp_path):
    s = Settings.from_env({"NETWORK": "mainnet", "MARKETS": "WETH:USDso", "EC_ENABLED": "true"})
    jobs = build_jobs(s, ROOT, tmp_path, 8000)
    assert names(jobs) == ["ec-recorder", "runner", "dashboard"]
    runner = jobs[1].argv
    assert "--ec-tape" in runner and "--perp-tape" not in runner
    perps = build_jobs(Settings.from_env({"NETWORK": "hideki", "PERP_MARKETS": "BTC/USDso:USDso"}), ROOT, tmp_path, 8000)
    assert names(perps) == ["perp-recorder", "runner", "dashboard"]
    assert "--perp-tape" in perps[1].argv and "BTC/USDso:USDso" in perps[1].argv


def test_a_perps_run_is_on_hideki_with_no_spot_markets(tmp_path):
    s = Settings.from_env({"NETWORK": "hideki", "PERP_MARKETS": "all"})
    assert s.markets == () and s.perp_markets == ("all",) and not s.ec
    jobs = build_jobs(s, ROOT, tmp_path, 8000)
    assert names(jobs) == ["perp-recorder", "runner", "dashboard"]
    rec, runner = jobs[0].argv, jobs[1].argv
    assert rec[rec.index("--network") + 1] == "hideki"
    assert runner[runner.index("--network") + 1] == "hideki"
    assert runner[runner.index("--perp-prefix") + 1] == "hideki_perp_"
    assert runner[runner.index("--symbols") + 1] == "--brain"                      # no spot markets at all


def test_each_market_kind_is_refused_on_the_wrong_network():
    with pytest.raises(SettingsError, match="no perp markets"):
        Settings.from_env({"NETWORK": "testnet", "PERP_MARKETS": "all"})          # Shannon
    with pytest.raises(SettingsError, match="no perp markets"):
        Settings.from_env({"NETWORK": "mainnet", "PERP_MARKETS": "BTC/USDso:USDso"})
    with pytest.raises(SettingsError, match="perp markets only"):
        Settings.from_env({"NETWORK": "hideki", "MARKETS": "SOMI:USDso", "PERP_MARKETS": "all"})
    with pytest.raises(SettingsError, match="perp markets only"):
        Settings.from_env({"NETWORK": "hideki", "EC_ENABLED": "true", "PERP_MARKETS": "all"})
    with pytest.raises(SettingsError, match="PERP_MARKETS is empty"):
        Settings.from_env({"NETWORK": "hideki"})
    with pytest.raises(SettingsError, match="hideki for perps"):
        Settings.from_env({"NETWORK": "shannon"})
    # spot and event contracts are fine on both of their networks, and a perps run on hideki needs no mainnet confirm
    assert Settings.from_env({"NETWORK": "testnet", "EC_ENABLED": "true"}).ec
    assert Settings.from_env({"NETWORK": "mainnet", "EC_ENABLED": "true"}).markets == Settings().markets
    assert Settings.from_env({"NETWORK": "hideki", "PERP_MARKETS": "all", "DRY_RUN": "false", "PRIVATE_KEY": "0x1"}).perp_live


def test_warm_brains_are_copied_once_and_never_over_existing_state(tmp_path):
    root, state = tmp_path / "repo", tmp_path / "state"
    for n in ("WETH_USDso", "EC"):
        (root / "config" / "warm" / n).mkdir(parents=True)
        np.save(root / "config" / "warm" / n / "w_kc_mbon.npy", np.zeros(3))
        (root / "config" / "warm" / n / "state.json").write_text("{}")
    s = Settings(markets=("WETH:USDso", "SOMI:USDso"), ec=True, brain="mb")
    assert sorted(seed_state(s, root, state)) == ["EC", "WETH_USDso"]     # SOMI has nothing bundled
    np.save(state / "WETH_USDso" / "w_kc_mbon.npy", np.ones(3))            # the fly has learned since
    assert seed_state(s, root, state) == []
    assert np.load(state / "WETH_USDso" / "w_kc_mbon.npy").sum() == 3
    assert state_name("BTC/USDso:USDso") == "BTC_USDso_USDso"


def test_the_delta_brain_is_seeded_from_its_own_replayed_folder(tmp_path):
    root, state = tmp_path / "repo", tmp_path / "state"
    (root / "config" / "warm_delta" / "WETH_USDso").mkdir(parents=True)
    np.save(root / "config" / "warm_delta" / "WETH_USDso" / "w_delta.npy", np.zeros((4, 3)))
    (root / "config" / "warm" / "SOMI_USDso").mkdir(parents=True)                  # an original-brain fly is not used for it
    np.save(root / "config" / "warm" / "SOMI_USDso" / "w_kc_mbon.npy", np.zeros(3))
    s = Settings(markets=("WETH:USDso", "SOMI:USDso"), brain="delta")
    assert seed_state(s, root, state) == ["WETH_USDso"]
    assert (state / "WETH_USDso" / "w_delta.npy").exists() and not (state / "SOMI_USDso").exists()
    np.save(state / "WETH_USDso" / "w_delta.npy", np.ones((4, 3)))
    assert seed_state(s, root, state) == []
    with pytest.raises(SettingsError, match="BRAIN"):
        Settings.from_env({"BRAIN": "other"})


def test_a_crashed_child_is_restarted_and_stop_ends_everything(tmp_path):
    flag = tmp_path / "runs"
    code = f"import pathlib,sys,time; p=pathlib.Path({str(flag)!r}); p.open('a').write('x'); time.sleep(0.05); sys.exit(1)"
    job = Job("flaky", [sys.executable, "-c", code], tmp_path, pause=0.01)
    sleeper = Job("steady", [sys.executable, "-c", "import time; time.sleep(60)"], tmp_path)
    sup = Supervisor([job, sleeper], log=lambda m: None)
    t = threading.Thread(target=sup.run, kwargs={"poll_s": 0.02})
    t.start()
    end = time.time() + 20
    while time.time() < end and (not flag.exists() or len(flag.read_text()) < 2):
        time.sleep(0.05)
    sup.stop()
    t.join(timeout=20)
    assert len(flag.read_text()) >= 2                                   # it ran again after crashing
    assert sleeper.proc.poll() is not None and not t.is_alive()         # and stop took everything down


def test_a_job_that_must_not_restart_stops_the_run(tmp_path):
    job = Job("once", [sys.executable, "-c", "pass"], tmp_path, restart=False)
    sup = Supervisor([job], log=lambda m: None)
    assert sup.run(poll_s=0.02) == 0
