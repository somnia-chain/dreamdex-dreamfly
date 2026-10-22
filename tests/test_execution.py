import sys
import types

import pytest

from flybot.arms import Costs
from flybot.live.execution import DryRunExecutor, ExecutionError, LiveExecutor, SpotMirror, WalletLedger, build_executor, explain
from flybot.live.paper import PaperEpisode
from flybot.live.settings import Settings, SettingsError

BB, BA = 100.0, 100.2


class Recorder:
    """An executor that remembers calls and can report a wallet balance."""

    dry = False

    def __init__(self, base=None, quote=None):
        self.calls, self.base, self.quote, self._n = [], base, quote, 0

    def place(self, symbol, is_bid, price, qty, post_only):
        self._n += 1
        self.calls.append(("place", "buy" if is_bid else "sell", price, qty, post_only))
        return f"o{self._n}"

    def cancel(self, symbol, order_id):
        self.calls.append(("cancel", order_id))
        return True

    def base_balance(self, symbol):
        return self.base

    def quote_balance(self, symbol):
        return self.quote


def settings(**kw):
    return Settings(**{"live_stake_usdso": 20.0, "min_gap_s": 0.0, **kw})


def mirror(ex, **kw):
    return SpotMirror(ex, "WETH:USDso", 1000.0, settings(**kw), lambda k, r: None, clock=iter(range(10_000)).__next__)


def episode(arm, **params):
    ep = PaperEpisode(arm, Costs(stake=1000.0), params)
    return ep


# ---------- settings ----------
def test_defaults_are_the_safe_ones():
    s = Settings.from_env({})
    assert s.network == "testnet" and s.dry_run and not s.live and not s.has_key
    assert "dry run" in s.summary()


def test_env_is_read_and_the_key_is_only_checked_for_presence():
    s = Settings.from_env({"NETWORK": "MAINNET", "DRY_RUN": "false", "PRIVATE_KEY": "0xabc", "MAINNET_CONFIRM": "yes",
                           "MARKETS": "WETH:USDso, WBTC:USDso", "LIVE_STAKE_USDSO": "50"})
    assert s.network == "mainnet" and s.live and s.has_key and s.live_stake_usdso == 50.0
    assert s.markets == ("WETH:USDso", "WBTC:USDso")
    assert "0xabc" not in repr(s) and "0xabc" not in s.summary()


def test_live_needs_a_key_and_mainnet_needs_confirmation():
    with pytest.raises(SettingsError, match="PRIVATE_KEY"):
        Settings.from_env({"DRY_RUN": "false"})
    with pytest.raises(SettingsError, match="PRIVATE_KEY"):
        Settings.from_env({"DRY_RUN": "false", "PRIVATE_KEY": "0x..."})     # the placeholder a generated .env ships with
    with pytest.raises(SettingsError, match="MAINNET_CONFIRM"):
        Settings.from_env({"DRY_RUN": "false", "PRIVATE_KEY": "0x1", "NETWORK": "mainnet"})
    assert Settings.from_env({"DRY_RUN": "false", "PRIVATE_KEY": "0x1", "NETWORK": "testnet"}).live
    assert Settings.from_env({"NETWORK": "mainnet"}).dry_run  # reading mainnet data is fine in a dry run


def test_a_mainnet_only_market_is_refused_on_testnet():
    with pytest.raises(SettingsError, match="mainnet only"):
        Settings.from_env({"NETWORK": "testnet", "MARKETS": "SOMI:USDso,USDC.e:USDso"})
    assert Settings.from_env({"NETWORK": "mainnet", "MARKETS": "SOMI:USDso,USDC.e:USDso"}).markets == ("SOMI:USDso", "USDC.e:USDso")


def test_bad_values_are_refused():
    for env in ({"NETWORK": "devnet"}, {"DRY_RUN": "maybe"}, {"LIVE_STAKE_USDSO": "abc"}, {"LIVE_STAKE_USDSO": "0"}):
        with pytest.raises(SettingsError):
            Settings.from_env(env)


# ---------- dry run ----------
def test_dry_run_logs_and_sends_nothing():
    rows = []
    ex = DryRunExecutor(lambda k, r: rows.append(r))
    assert ex.place("WETH:USDso", True, 100.0, 0.5, True) == "dry-1"
    assert ex.cancel("WETH:USDso", "dry-1") and ex.base_balance("WETH:USDso") is None
    assert [r["action"] for r in rows] == ["place", "cancel"] and all(r["mode"] == "dry" for r in rows)
    assert isinstance(build_executor(Settings(), lambda k, r: None), DryRunExecutor)


# ---------- mirror ----------
def test_maker_quotes_are_placed_scaled_and_post_only():
    ex, m, ep = Recorder(), mirror(Recorder()), episode("maker", spread_bps=20, skew_bps=10, clip_frac=0.25)
    m = mirror(ex)
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    placed = [c for c in ex.calls if c[0] == "place"]
    assert {p[1] for p in placed} == {"buy", "sell"} and all(p[4] for p in placed)
    mid = (BB + BA) / 2
    assert placed[0][3] == pytest.approx(ep.clip * 20.0 / 1000.0)      # 25% of a 20 USDso stake
    assert placed[0][3] * mid == pytest.approx(20.0 * 0.25, rel=1e-6)


def test_a_small_paper_requote_does_not_churn_live_orders():
    ex = Recorder()
    m, ep = mirror(ex, requote_bps=5.0), episode("maker", spread_bps=20, skew_bps=10, clip_frac=0.25)
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    n = len(ex.calls)
    m.sync(ep, BB, BA)
    ep.on_touch(1, BB + 0.01, BA + 0.01)                              # 1 bp move: inside the tolerance
    m.sync(ep, BB + 0.01, BA + 0.01)
    assert len(ex.calls) == n
    ep.on_touch(2, BB + 1.0, BA + 1.0)                                # 100 bps move: cancel and re-place
    m.sync(ep, BB + 1.0, BA + 1.0)
    kinds = [c[0] for c in ex.calls[n:]]
    assert kinds.count("cancel") == 2 and kinds.count("place") == 2


def test_open_orders_are_capped():
    ex = Recorder()
    m, ep = mirror(ex, max_open_orders=2), episode("grid", step_bps=30, levels=4, clip_frac=0.2)
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    assert sum(1 for c in ex.calls if c[0] == "place") == 2


def test_flat_and_finished_episodes_send_nothing():
    ex = Recorder()
    m, ep = mirror(ex), episode("flat")
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    assert ex.calls == []


def test_trend_entry_is_a_taker_through_the_touch_and_capped_to_the_stake():
    ex = Recorder()
    m, ep = mirror(ex, take_buffer_bps=10.0), episode("trend")
    ep.start(0, BB, BA, trend_dir=1)
    m.on_start(ep, BB, BA)
    (kind, side, px, qty, post) = ex.calls[0]
    assert (kind, side, post) == ("place", "buy", False)
    assert px == pytest.approx(BA * 1.001)
    assert qty * px <= 20.0 * 1.05 + 1e-9


def test_finish_cancels_and_flattens_from_the_wallet_when_it_is_known():
    ex = Wallet(base=1.0, fills=[True])
    m, ep = mirror(ex), episode("maker", spread_bps=20, skew_bps=10, clip_frac=0.25)
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)          # start balance 1.0
    ex.base = 1.15                  # a bid filled for real: 0.15 base, about 15 USDso
    ex.calls.clear()
    m.finish(ep, BB, BA)
    assert sum(1 for c in ex.calls if c[0] == "cancel") == 2
    sells = [c for c in ex.calls if c[0] == "place"]
    assert len(sells) == 1 and sells[0][1] == "sell" and not sells[0][4]
    assert sells[0][3] == pytest.approx(0.15)                      # the whole surplus, not capped like an entry


class Wallet(Recorder):
    """A recorder whose balance only moves when a taker order is marked as filling."""

    def __init__(self, base, fills=()):
        super().__init__(base)
        self.fills = list(fills)

    def place(self, symbol, is_bid, price, qty, post_only):
        oid = super().place(symbol, is_bid, price, qty, post_only)
        if not post_only and self.fills and self.fills.pop(0):
            self.base += qty if is_bid else -qty
        return oid


def test_a_missed_close_is_sent_again_wider_and_then_at_the_next_episode_end():
    rows = []
    ex = Wallet(base=1.0, fills=[False, False, True])              # first two closes miss, the third fills
    m = SpotMirror(ex, "WETH:USDso", 1000.0, settings(take_buffer_bps=10.0), lambda k, r: rows.append(r), clock=iter(range(10_000)).__next__)
    ep = episode("maker")
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    ex.base = 1.2                                                  # a real fill: 0.2 base held
    m.finish(ep, BB, BA)
    closes = [c for c in ex.calls if c[0] == "place" and not c[4]]
    assert [c[1] for c in closes] == ["sell", "sell"]
    assert closes[1][2] == pytest.approx(BB * (1 - 0.005)) and closes[0][2] == pytest.approx(BB * 0.999)
    assert rows and rows[-1]["action"] == "close_left" and rows[-1]["base"] == pytest.approx(0.2)
    ep2 = episode("flat")
    m.on_start(ep2, BB, BA)
    ex.calls.clear()
    m.finish(ep2, BB, BA)                                          # the leftover is still against the baseline
    closes = [c for c in ex.calls if c[0] == "place" and not c[4]]
    assert len(closes) == 1 and closes[0][3] == pytest.approx(0.2) and ex.base == pytest.approx(1.0)
    assert m.base0 == pytest.approx(1.0)


def test_a_failed_cancel_is_retried_and_blocks_a_slot_until_it_gives_up():
    ex = Recorder()
    ex.cancel = lambda symbol, oid: ex.calls.append(("cancel", oid)) or False
    m, ep = mirror(ex, max_open_orders=2), episode("grid", levels=1, step_bps=30)
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)                                         # one bid and one ask resting
    ep.bids, ep.asks = [BB * 0.99], [BA * 1.01]                    # paper moved both quotes
    m.sync(ep, BB, BA)
    assert set(m.stale) == {"o1", "o2"} and not [c for c in ex.calls if c[0] == "place" and c[2] == BB * 0.99]
    m.sync(ep, BB, BA)
    m.sync(ep, BB, BA)                                             # third failure: given up, the slots free again
    assert not m.stale
    assert sum(1 for c in ex.calls if c[0] == "cancel") == 6


def test_nothing_is_placed_once_the_episode_is_closing():
    ex = Recorder(base=1.0)
    m, ep = mirror(ex), episode("maker")
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    m.finish(ep, BB, BA)
    ex.calls.clear()
    m.sync(ep, BB, BA, force=True)
    assert ex.calls == []


def test_finish_ignores_dust_and_uses_the_paper_inventory_in_a_dry_run():
    ex = Recorder(base=1.0)
    m, ep = mirror(ex), episode("maker", spread_bps=20, skew_bps=10, clip_frac=0.25)
    ep.start(0, BB, BA)
    m.on_start(ep, BB, BA)
    ex.base = 1.0 + 1e-6
    ex.calls.clear()
    m.finish(ep, BB, BA)
    assert not [c for c in ex.calls if c[0] == "place"]
    dry = mirror(DryRunExecutor(lambda k, r: None))
    ep2 = episode("maker")
    ep2.start(0, BB, BA)
    dry.on_start(ep2, BB, BA)
    ep2.inv = 0.1                                                    # the paper episode holds inventory
    dry.finish(ep2, BB, BA)                                          # must not raise


# ---------- live executor, against a stand-in for dreamdex_core ----------
def fake_core(fail=False):
    core = types.ModuleType("dreamdex_core")
    core.OrderType = types.SimpleNamespace(POST_ONLY=3, IOC=2)
    core.created = []

    class Pool:
        def __init__(self, symbol):
            self.symbol, self.placed, self.cancelled = symbol, [], []

        @classmethod
        def load(cls, ctx, symbol):
            return cls(symbol)

        def place(self, **kw):
            if fail:
                raise RuntimeError("[SIM_REVERT] nope")
            self.placed.append(kw)
            return types.SimpleNamespace(order_id=77, tx_hash="0xtx", gas_used=123)

        def cancel(self, order_id):
            self.cancelled.append(order_id)
            return "0xcancel"

        def wallet_base(self):
            return 2.5

    core.Pool = Pool
    core.create_chain_context = lambda: core.created.append(1) or "ctx"
    return core


def test_live_executor_places_through_the_core_and_logs(monkeypatch):
    core = fake_core()
    monkeypatch.setitem(sys.modules, "dreamdex_core", core)
    monkeypatch.setattr(LiveExecutor, "_ensure_quote_allowance", lambda self, pool, price, qty: None)   # tested below
    rows = []
    s = Settings(dry_run=False, has_key=True, network="testnet", quote_ttl_min=30)
    ex = LiveExecutor(s, lambda k, r: rows.append(r))
    assert core.created == [1]
    assert ex.place("WETH:USDso", True, 100.0, 0.2, True) == "77"
    assert ex.place("WETH:USDso", False, 99.0, 0.2, False) == "77"
    pool = ex.pools["WETH:USDso"]
    assert pool.placed[0]["order_type"] == 3 and pool.placed[1]["order_type"] == 2
    assert pool.placed[0]["expire_ms"] == 30 * 60_000
    assert ex.cancel("WETH:USDso", "77") and pool.cancelled == [77]
    assert ex.base_balance("WETH:USDso") == 2.5
    assert all(r["mode"] == "live" and r["ok"] for r in rows)


def test_live_executor_survives_a_rejected_order(monkeypatch):
    monkeypatch.setitem(sys.modules, "dreamdex_core", fake_core(fail=True))
    monkeypatch.setattr(LiveExecutor, "_ensure_quote_allowance", lambda self, pool, price, qty: None)
    rows = []
    ex = LiveExecutor(Settings(dry_run=False, has_key=True), lambda k, r: rows.append(r))
    assert ex.place("WETH:USDso", True, 100.0, 0.2, True) is None
    assert rows[-1]["ok"] is False and "SIM_REVERT" in rows[-1]["error"]


def test_live_executor_explains_a_missing_core(monkeypatch):
    monkeypatch.setitem(sys.modules, "dreamdex_core", None)
    with pytest.raises(ExecutionError, match="dreamfly"):
        LiveExecutor(Settings(dry_run=False, has_key=True), lambda k, r: None)


def test_the_daily_loss_stop_only_exists_for_live_orders():
    dry = Settings()
    assert dry.daily_loss_stop() == float("inf")                # paper and dry runs learn through losses
    assert dry.daily_loss_stop(20.0) == 20.0                    # unless a stop is asked for
    live = Settings(dry_run=False, has_key=True)
    assert live.daily_loss_stop() == 50.0                       # real money keeps its stop
    assert live.daily_loss_stop(10.0) == 10.0


# ---------- wallet ledger: the daily stop watches real money ----------
DAY1, DAY2 = 1_790_000_000.0, 1_790_000_000.0 + 86_400   # two UTC days


def test_the_ledger_values_the_wallet_in_usdso_and_resets_at_midnight():
    ex = Recorder(base=0.0, quote=1000.0)
    now = [DAY1]
    led = WalletLedger(ex, clock=lambda: now[0])
    assert led.day_pnl is None                                   # nothing read yet: the stop falls back to paper
    assert led.update("WETH:USDso", BB, BA) == 0.0               # the first reading of the day is the baseline
    ex.quote, ex.base = 960.0, 0.1                               # bought 0.1 for 40, now worth 10 at mid 100.1
    assert round(led.update("WETH:USDso", BB, BA), 2) == -29.99
    ex.quote, ex.base = 950.0, 0.0                               # sold it for 10: the loss is 50, in cash
    assert round(led.update("WETH:USDso", BB, BA), 2) == -50.0
    now[0] = DAY2                                                # a new day starts from today's wallet
    assert led.update("WETH:USDso", BB, BA) == 0.0


def test_the_ledger_counts_every_market_once_against_one_wallet():
    ex = Recorder(base=0.0, quote=1000.0)
    led = WalletLedger(ex, clock=lambda: DAY1)
    led.update("WETH:USDso", BB, BA)
    ex.quote = 980.0                                             # a second market reads the same, lower, USDso balance
    assert led.update("WBTC:USDso", 200.0, 200.4) == -20.0       # the quote move is counted once, not per market
    ex.base = 0.05                                               # WBTC inventory 0.05 at mid 200.2 = 10.01
    assert round(led.update("WBTC:USDso", 200.0, 200.4), 2) == -9.99


def test_an_unreadable_wallet_leaves_the_ledger_unknown():
    led = WalletLedger(Recorder(), clock=lambda: DAY1)           # base and quote both None (dry run, or RPC down)
    assert led.update("WETH:USDso", BB, BA) is None
    assert led.day_pnl is None


def test_finish_reads_the_wallet_into_the_shared_ledger():
    ex = Recorder(base=0.0, quote=500.0)
    led = WalletLedger(ex, clock=lambda: DAY1)
    m = mirror(ex)
    m.ledger = led
    ep = episode("maker", n_levels=1)
    m.on_start(ep, BB, BA)
    ex.quote = 470.0                                             # the day cost 30 USDso by the time the episode ends
    m.finish(ep, BB, BA)
    assert led.day_pnl == -30.0


def test_known_pool_errors_get_a_name_and_unknown_ones_are_left_alone():
    raw = "[SIM_REVERT] placeOrder simulation reverted: ('0xd48c4403', '0xd48c4403')"
    assert explain(raw).startswith("ImmediateOrCancelNoFill") and raw in explain(raw)
    assert "ERC20InsufficientAllowance" in explain("revert 0xfb8f41b2 0000")
    assert explain("something else") == "something else"


class FakeNM:
    def __init__(self):
        self.n, self.resets = 0, 0

    def reserve(self):
        self.n += 1
        return self.n

    def gas_fields(self):
        return {"gasPrice": 1}

    def reset(self):
        self.resets += 1


def allowance_world(allowance, token="0x" + "9c" * 20, status=1, native=False):
    """A pool and chain just real enough for the approval step: one ERC-20, a recorded approve."""
    from types import SimpleNamespace as NS

    sent = []

    class Fn:
        def __init__(self, v):
            self.v = v

        def call(self, *a, **k):
            return self.v

        def estimate_gas(self, *a, **k):
            return 400_000

        def build_transaction(self, tx):
            return {**tx, "data": "approve"}

    class Erc:
        class functions:
            @staticmethod
            def allowance(owner, spender):
                return Fn(allowance)

            @staticmethod
            def approve(spender, amount):
                sent.append(amount)
                return Fn(None)

    def contract(address=None, abi=None):
        return Erc

    eth = NS(contract=contract, send_raw_transaction=lambda raw: b"h", wait_for_transaction_receipt=lambda h, timeout: {"status": status, "gasUsed": 1})
    acct = NS(sign_transaction=lambda tx: NS(raw_transaction=b"raw", gas=tx["gas"]))
    ctx = NS(address="0x" + "aa" * 20, w3=NS(eth=eth), account=acct, net=NS(chain_id=5031))
    pull = ("0x28f34DeFd2b4CB48d9eE6d89f2Be4Bc601694c00" if native else token, 500, 500)
    pool = NS(params=NS(tick_size=100, lot_size=10), quote_decimals=18, base_decimals=18, address="0x" + "11" * 20, nm=FakeNM(),
              _contract=NS(functions=NS(getAutoPullRequirement=lambda *a: Fn(pull))))
    ex = LiveExecutor.__new__(LiveExecutor)
    ex.ctx = ctx
    return ex, pool, sent


def test_a_bid_approves_the_quote_token_with_an_estimated_gas_limit():
    pytest.importorskip("dreamdex_core")
    ex, pool, sent = allowance_world(allowance=0)
    ex._ensure_quote_allowance(pool, 0.2, 10.0)
    assert sent == [500 * 8]                       # the same 8x headroom the core uses
    assert pool.nm.n == 1


def test_no_approval_when_the_allowance_is_enough_or_the_pull_is_native():
    pytest.importorskip("dreamdex_core")
    ex, pool, sent = allowance_world(allowance=10**30)
    ex._ensure_quote_allowance(pool, 0.2, 10.0)
    ex2, pool2, sent2 = allowance_world(allowance=0, native=True)
    ex2._ensure_quote_allowance(pool2, 0.2, 10.0)
    assert sent == [] and sent2 == [] and pool.nm.n == 0 and pool2.nm.n == 0


def test_an_approval_that_reverts_is_reported_not_ignored():
    pytest.importorskip("dreamdex_core")
    ex, pool, _ = allowance_world(allowance=0, status=0)
    with pytest.raises(ExecutionError, match="approving the quote token reverted"):
        ex._ensure_quote_allowance(pool, 0.2, 10.0)
