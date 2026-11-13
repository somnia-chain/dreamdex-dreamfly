from flybot.arms import Costs
from flybot.live.paper import PaperEpisode

C = Costs(taker_bps=0.0, slip_bps=0.0, gas_per_action=0.0, stake=1000.0)


def test_maker_fills_when_touch_crosses_and_earns_spread():
    ep = PaperEpisode("maker", C, {"spread_bps": 20, "skew_bps": 0, "clip_frac": 0.5})
    ep.start(0, 99.99, 100.01)
    assert len(ep.bids) == 1 and len(ep.asks) == 1
    ep.on_touch(1, 99.7, 99.75)  # ask dropped to our bid at 99.8 -> we buy
    assert ep.inv > 0 and len(ep.fills) == 1 and not ep.fills[0].taker
    ep.on_touch(2, 100.3, 100.35)  # bid rose above our ask -> we sell
    assert abs(ep.inv) < 1e-9 and len(ep.fills) == 2
    pnl = ep.finish(3, 100.0, 100.02)
    assert pnl > 0


def test_maker_inventory_cap():
    ep = PaperEpisode("maker", C, {"spread_bps": 20, "skew_bps": 0, "clip_frac": 1.0})
    ep.start(0, 100.0, 100.0)
    ep.on_touch(1, 99.0, 99.0)
    ep.on_touch(2, 98.0, 98.0)  # already at max inventory: no second buy
    assert len(ep.fills) == 1


def test_trend_enters_at_touch_and_flattens():
    ep = PaperEpisode("trend", Costs(taker_bps=0, slip_bps=0, gas_per_action=0), {})
    ep.start(0, 99.9, 100.1, trend_dir=1)
    assert ep.inv > 0 and ep.fills[0].price == 100.1 and ep.fills[0].taker
    pnl = ep.finish(1, 101.0, 101.2)
    assert pnl > 0 and abs(ep.inv) < 1e-9


def test_grid_fills_levels_once_and_rearms():
    ep = PaperEpisode("grid", C, {"levels": 2, "step_bps": 100, "clip_frac": 0.2})
    ep.start(0, 100.0, 100.0)
    assert len(ep.bids) == 2 and len(ep.asks) == 2
    ep.on_touch(1, 97.9, 97.95)  # ask reaches both bid levels (99, 98)
    assert len(ep.fills) == 2 and ep.inv > 0
    ep.on_touch(2, 102.1, 102.15)  # bid reaches both ask levels (101, 102)
    assert len(ep.fills) == 4
    assert ep.finish(3, 100.0, 100.0) > 0


def test_flat_does_nothing():
    ep = PaperEpisode("flat", C, {})
    ep.start(0, 1.0, 1.01)
    ep.on_touch(1, 0.9, 0.91)
    assert ep.finish(2, 1.0, 1.01) == 0.0 and ep.actions == 0


def test_gas_is_charged_per_action():
    c = Costs(taker_bps=0, slip_bps=0, gas_per_action=1.0)
    ep = PaperEpisode("trend", c, {})
    ep.start(0, 100.0, 100.0, trend_dir=1)
    ep.finish(1, 100.0, 100.0)
    assert ep.pnl() == -2.0


def test_a_maker_does_not_requote_on_every_small_touch_move():
    c = Costs(taker_bps=0.0, slip_bps=0.0, gas_per_action=0.001, stake=1000.0)
    ep = PaperEpisode("maker", c, {"spread_bps": 20, "skew_bps": 0, "clip_frac": 0.5})
    ep.start(0, 99.99, 100.01)
    start_actions, quotes = ep.actions, (list(ep.bids), list(ep.asks))
    for i in range(1, 200):                                # the touch wobbles by about 1 bp, 200 times
        wobble = 0.01 * (1 if i % 2 else -1)
        ep.on_touch(i, 99.99 + wobble, 100.01 + wobble)
    assert ep.actions == start_actions and (ep.bids, ep.asks) == quotes
    ep.on_touch(300, 100.49, 100.51)                       # a 50 bp move is worth a requote
    assert ep.actions == start_actions + 2 and ep.bids != quotes[0]
    tight = PaperEpisode("maker", c, {"spread_bps": 20, "skew_bps": 0, "requote_bps": 0.0})
    tight.start(0, 99.99, 100.01)
    for i in range(1, 50):
        tight.on_touch(i, 99.99 + 0.002 * i, 100.01 + 0.002 * i)
    assert tight.actions > start_actions + 20             # a zero tolerance restores requoting on every move
