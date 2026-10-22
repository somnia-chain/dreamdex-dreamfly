import json
from pathlib import Path

import numpy as np
import pytest

from flybot.ec import EC_ARMS, ECCosts
from flybot.live.ec_live import ECFly, book_prob_up, features_of, plan_take, seller_ok
from flybot.live.tape import Tail
from flybot.mb.graph import load_graph

MB_DIR = Path(__file__).resolve().parents[1] / "data" / "mb"
needs_mb = pytest.mark.skipif(not (MB_DIR / "edges.parquet").exists(), reason="run scripts/extract_mb.py first")
MID = "0x7c21"


def book(left: float, up_ask=None, down_ask=None, prob_up=None, z=None, wall="2026-09-29T14:59:35.000Z", interval="5m"):
    r = {"type": "ec_book", "marketId": MID, "symbol": "BTC-0-29SEP26-1500/USDso", "asset": "BTC", "interval": interval,
         "expiry": 1790694000, "secondsLeft": left, "wall": wall, "sigma_bps": 0.6, "lead_bps": 20.0,
         "prob_up": prob_up, "z": z,
         "up": {"bids": [], "asks": [[up_ask, 100]] if up_ask is not None else []},
         "down": {"bids": [], "asks": [[down_ask, 100]] if down_ask is not None else []}}
    return r


def test_book_prob_up_combines_both_sides_and_falls_back():
    assert abs(book_prob_up(book(20, up_ask=0.96, down_ask=0.06)) - 0.95) < 1e-9
    assert abs(book_prob_up(book(20, down_ask=0.06)) - 0.94) < 1e-9
    assert np.isnan(book_prob_up(book(20)))
    f = features_of(book(20, up_ask=0.96, z=2.0))
    assert f["fav_prob"] == 0.96 and f["z_abs"] == 2.0 and f["is_5m"] == 1.0 and f["is_btc"] == 1.0


def test_plan_take_late_needs_confidence_and_z():
    assert plan_take("ec_late", book(20, prob_up=0.97, z=2.0), {}) == ("up", 0.97 - 0.06)
    assert plan_take("ec_late", book(20, prob_up=0.03, z=-2.0), {}) == ("down", 0.97 - 0.06)
    assert plan_take("ec_late", book(20, prob_up=0.90, z=2.0), {}) is None
    assert plan_take("ec_late", book(20, prob_up=0.97, z=1.0), {}) is None
    assert plan_take("ec_late", book(20, prob_up=None, z=None), {}) is None


def test_plan_take_sharp_5m_only_and_threshold():
    assert plan_take("ec_sharp", book(20, up_ask=0.93), {}) == ("up", 0.95)
    assert plan_take("ec_sharp", book(20, up_ask=0.80), {}) is None
    assert plan_take("ec_sharp", book(20, up_ask=0.93, interval="15m"), {}) is None
    assert plan_take("flat", book(20, up_ask=0.93), {}) is None


def test_seller_gate():
    assert seller_ok(book(20, up_ask=0.90), "up", 0.91, {}) == (0.90, 100.0)
    assert seller_ok(book(20, up_ask=0.92), "up", 0.91, {}) is None  # above cap
    assert seller_ok(book(20, up_ask=0.80), "up", 0.91, {}) is None  # too generous
    assert seller_ok(book(20), "up", 0.91, {}) is None  # nobody there


@needs_mb
def test_window_decides_fills_on_confirmation_and_settles(tmp_path):
    g = load_graph(MB_DIR)
    ec = ECFly(g, tmp_path, ECCosts(stake=20.0), {}, seed=0)
    ec.fly.explore = 0.0
    ec.scaler.n, ec.scaler.scale = 25, 1.0  # past warm-up, unit scale: reward = P&L over stake
    ec.on_row({"type": "ec_open", "marketId": MID, "symbol": "BTC-0-29SEP26-1500/USDso", "asset": "BTC", "interval": "5m", "expiry": 1790694000})
    ec.on_row(book(40, up_ask=0.93, prob_up=0.97, z=2.0))  # outside the action band: no decision yet
    assert ec.windows[MID].arm is None
    ec.on_row(book(24, up_ask=0.93, prob_up=0.97, z=2.0))
    w = ec.windows[MID]
    assert w.arm in EC_ARMS and w.last is not None
    assert (tmp_path / "EC" / "decisions.jsonl").exists()
    w.arm = "ec_sharp"  # force a taking arm so the fill path is exercised
    ec.on_row(book(21, up_ask=0.93, prob_up=0.97, z=2.0))  # candidate seen
    assert w.pending == ("up", 0.95) and w.side is None
    ec.on_row(book(18, up_ask=0.94, prob_up=0.97, z=2.0))  # still there on the next sample: filled at the later ask
    assert w.side == "up" and w.price == 0.94 and abs(w.units - 20 / 0.94) < 1e-9
    done = ec.on_row({"type": "ec_resolution", "marketId": MID, "symbol": w.symbol, "truth": "up", "voided": False,
                      "wall": "2026-09-29T15:00:32.000Z"})
    assert done["pnl"] > 0 and done["reward"] > 0 and ec.n_episodes == 1 and MID not in ec.windows
    assert (tmp_path / "EC" / "w_kc_mbon.npy").exists()
    snap = ec.snapshot()
    assert snap["kind"] == "ec" and snap["episodes"] == 1


@needs_mb
def test_window_loss_and_void_and_no_fill(tmp_path):
    g = load_graph(MB_DIR)
    ec = ECFly(g, tmp_path, ECCosts(stake=20.0), {}, seed=1)
    ec.scaler.n, ec.scaler.scale = 25, 1.0
    ec.on_row(book(24, up_ask=0.93, prob_up=0.97, z=2.0))
    w = ec.windows[MID]
    w.arm = "ec_sharp"
    ec.on_row(book(21, up_ask=0.93))
    ec.on_row(book(18, up_ask=0.93))
    assert w.side == "up"
    lost = ec.on_row({"type": "ec_resolution", "marketId": MID, "symbol": w.symbol, "truth": "down", "voided": False, "wall": "2026-09-29T15:00:32Z"})
    assert abs(lost["pnl"] + 20.0) < 1e-9 and lost["reward"] == -1.0
    # a window the arm never filled settles at zero
    ec.on_row(book(24, up_ask=0.60, prob_up=0.6, z=0.3))
    ec.windows[MID].arm = "ec_late"
    ec.on_row(book(18, up_ask=0.60, prob_up=0.6, z=0.3))
    none = ec.on_row({"type": "ec_resolution", "marketId": MID, "symbol": "x", "truth": "up", "voided": False, "wall": "2026-09-29T15:00:32Z"})
    assert none["pnl"] == 0.0 and none["side"] is None


def test_tail_follows_new_lines_and_rolls_files(tmp_path):
    f1 = tmp_path / "mainnet_ec_20260929.jsonl"
    f1.write_text(json.dumps({"a": 1}) + "\n")
    t = Tail(tmp_path, "mainnet_ec_", from_start=True)
    assert [r["a"] for r in t.read_new()] == [1]
    with f1.open("a") as fh:
        fh.write(json.dumps({"a": 2}) + "\n" + '{"a": 3')  # partial line stays buffered
    assert [r["a"] for r in t.read_new()] == [2]
    with f1.open("a") as fh:
        fh.write("}\n")
    assert [r["a"] for r in t.read_new()] == [3]
    (tmp_path / "mainnet_ec_20260930.jsonl").write_text(json.dumps({"a": 4}) + "\n")
    assert [r["a"] for r in t.read_new()] == [4]
    late = Tail(tmp_path, "mainnet_ec_")  # default: start at the end of the newest file
    assert late.read_new() == []
