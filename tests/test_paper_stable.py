"""Stable pair mode of the paper episode: price grid, quotes inside the touch, fills from trade prints with queue position."""

import pytest

from flybot.arms import Costs
from flybot.live.paper import PaperEpisode

C = Costs(taker_bps=0.0, slip_bps=0.0, gas_per_action=0.0, stake=1000.0)
P = {"mode": "inside", "tick": 0.0001, "improve_ticks": 1, "trade_fills": True, "stake": 250.0, "clip_frac": 1.0}


class FakeBook:
    def __init__(self, bids: dict, asks: dict):
        self.bids, self.asks = bids, asks


def ep_started(bb=0.9998, ba=1.0002, **over) -> PaperEpisode:
    ep = PaperEpisode("maker", C, {**P, **over})
    ep.start(0, bb, ba)
    return ep


def test_quotes_sit_one_tick_inside_the_touch():
    ep = ep_started()
    assert ep.bids == [0.9999] and ep.asks == [1.0001]


def test_quotes_join_the_touch_when_there_is_no_room_to_improve():
    ep = ep_started(0.9999, 1.0001)     # a 2 tick spread: improving both sides would lock the book
    assert ep.bids == [0.9999] and ep.asks == [1.0001]


def test_quotes_are_snapped_to_the_grid_bids_down_asks_up():
    ep = ep_started(0.99983, 1.00017)
    assert ep.bids == [0.9999] and ep.asks == [1.0001]


def test_a_sell_print_fills_our_bid_up_to_its_own_size():
    ep = ep_started()
    ep.on_trade(1, 0.9998, 8.8, "sell")         # the market's sell printed at the old best bid; we are better priced
    assert len(ep.fills) == 1 and not ep.fills[0].taker
    assert ep.fills[0].price == 0.9999 and ep.fills[0].side == "buy" and ep.fills[0].qty == pytest.approx(8.8)
    assert ep.inv == pytest.approx(8.8) and ep.bids == [0.9999]
    ep.on_trade(2, 0.9998, 5.0, "sell")
    assert ep.inv == pytest.approx(13.8)        # partial fills accumulate against the same resting clip


def test_a_buy_print_fills_our_ask_and_a_buy_never_fills_our_bid():
    ep = ep_started()
    ep.on_trade(1, 1.0002, 10.0, "buy")
    assert [(f.side, f.price) for f in ep.fills] == [("sell", 1.0001)]
    assert ep.inv == pytest.approx(-10.0)


def test_the_queue_ahead_is_eaten_before_we_fill():
    ep = ep_started(0.9999, 1.0001)             # joined the touch, behind the book's own orders
    ep.on_touch(1, 0.9999, 1.0001, FakeBook({0.9999: 100.0, 0.9998: 500.0}, {1.0001: 100.0}))
    ep.on_trade(2, 0.9999, 60.0, "sell")
    assert not ep.fills                          # 60 of the 100 ahead of us
    ep.on_trade(3, 0.9999, 60.0, "sell")         # 40 left ahead, 20 overflow reaches us
    assert len(ep.fills) == 1 and ep.fills[0].qty == pytest.approx(20.0)


def test_a_print_better_priced_than_us_only_eats_the_queue():
    ep = ep_started(0.9999, 1.0001)
    ep.on_touch(1, 0.9999, 1.0001, FakeBook({1.0000: 30.0, 0.9999: 100.0}, {1.0001: 100.0}))
    ep.on_trade(2, 1.0000, 500.0, "sell")        # sold into a better bid: 130 queued at our price or better, no overflow here
    assert not ep.fills


def test_the_inventory_cap_stops_the_bid_and_keeps_the_ask():
    ep = ep_started()
    ep.on_trade(1, 0.9998, 1000.0, "sell")       # far more than the 250 cap
    assert ep.inv == pytest.approx(250.0) and ep.bids == [] and ep.asks == [1.0001]
    ep.on_minute(60_000, 0.9998, 1.0002)
    assert ep.bids == []                         # still at the cap: no bid is re-placed


def test_a_used_up_clip_is_replaced_on_the_next_minute_with_a_fresh_queue_place():
    ep = ep_started(clip_frac=0.1)               # clip 25
    ep.on_trade(1, 0.9998, 25.0, "sell")
    assert ep.bids == []
    ep.on_minute(60_000, 0.9998, 1.0002)
    assert ep.bids == [0.9999]
    ep.on_trade(61_000, 0.9998, 5.0, "sell")
    assert ep.inv == pytest.approx(30.0)


def test_a_round_trip_earns_the_two_tick_spread():
    ep = ep_started(clip_frac=0.1)
    ep.on_trade(1, 0.9998, 25.0, "sell")         # we buy 25 at 0.9999
    ep.on_trade(2, 1.0002, 25.0, "buy")          # we sell 25 at 1.0001
    assert abs(ep.inv) < 1e-9
    assert ep.finish(3, 0.9998, 1.0002) == pytest.approx(25.0 * 0.0002)


def test_flattening_pays_the_touch_and_charges_gas():
    ep = PaperEpisode("maker", Costs(taker_bps=0.0, slip_bps=0.0, gas_per_action=0.001, stake=1000.0), {**P, "clip_frac": 0.1})
    ep.start(0, 0.9998, 1.0002)
    assert ep.actions == 2                       # one placed quote per side
    ep.on_trade(1, 0.9998, 25.0, "sell")
    pnl = ep.finish(2, 0.9998, 1.0002)           # long 25 bought at 0.9999, sold at the bid 0.9998
    assert pnl == pytest.approx(25.0 * (0.9998 - 0.9999) - ep.actions * 0.001)
    assert ep.actions == 2 + 1 + 1               # the bid was used up, so one quote cancelled, plus one flatten


def test_trade_fills_replace_the_touch_rule():
    ep = ep_started()
    ep.on_touch(1, 0.9995, 0.9996, FakeBook({0.9995: 50.0}, {0.9996: 50.0}))   # the market fell through our bid
    assert not ep.fills                          # fills come only from prints
    assert ep.bids == [0.9995] and ep.asks == [0.9996]     # a one tick spread leaves no room to improve: quotes join the new touch


def test_everything_is_off_by_default():
    ep = PaperEpisode("maker", C, {"spread_bps": 20, "skew_bps": 0, "clip_frac": 0.5})
    ep.start(0, 99.99, 100.01)
    ep.on_trade(1, 99.0, 1000.0, "sell")
    assert not ep.fills                          # no trade_fills: prints are ignored
    ep.on_touch(2, 99.7, 99.75)
    assert len(ep.fills) == 1                    # the touch rule still works as before
    assert ep.tick == 0.0 and not ep.inside
