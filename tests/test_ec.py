import numpy as np
import pandas as pd

from flybot import ec


def row(fav_prob: float, truth_matches: bool, interval: str = "5m", z_abs: float = 2.0) -> pd.Series:
    fav_up = 1
    return pd.Series({"fav_prob": fav_prob, "fav_up": fav_up, "truth": fav_up if truth_matches else 0,
                      "interval": interval, "z_abs": z_abs})


def test_buy_favourite_payoff_is_binary():
    c = ec.ECCosts(stake=20.0, haircut=0.02)
    win = ec._buy_favourite(row(0.90, True), c)
    loss = ec._buy_favourite(row(0.90, False), c)
    assert loss == -20.0
    assert abs(win - 20.0 / 0.92 * 0.08) < 1e-9


def test_late_taker_filters():
    c = ec.ECCosts()
    assert ec.ec_late(row(0.96, True, z_abs=2.0), c, {}) > 0
    assert ec.ec_late(row(0.90, True, z_abs=2.0), c, {}) == 0.0
    assert ec.ec_late(row(0.96, True, z_abs=1.0), c, {}) == 0.0


def test_sharpened_book_is_5m_only_and_thresholded():
    c = ec.ECCosts()
    assert ec.ec_sharp(row(0.85, True), c, {}) > 0  # logit(0.85)*1.3 -> ~0.91
    assert ec.ec_sharp(row(0.60, True), c, {}) == 0.0
    assert ec.ec_sharp(row(0.85, True, interval="15m"), c, {}) == 0.0


def test_run_ec_rewards_are_stake_scaled():
    w = pd.DataFrame([row(0.9, True), row(0.9, False)])
    w["ts"] = pd.to_datetime([1, 2], unit="s", utc=True)
    w["asset"] = "BTC"

    class Always:
        def choose(self, r):
            return ec.EC_ARMS.index("ec_sharp")

        def update(self, r):
            pass

    res = ec.run_ec(w, Always(), ec.ECCosts(stake=10.0), adaptive_reward=False)  # raw P&L over stake
    assert list(res["arm"]) == ["ec_sharp", "ec_sharp"]
    assert res["reward"].iloc[1] == -1.0
    assert 0 < res["reward"].iloc[0] < 1
