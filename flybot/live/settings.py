"""Run settings from the environment, in the same shape as the DreamDEX bot kit: NETWORK, DRY_RUN, PRIVATE_KEY.

The defaults are the safe ones: testnet, dry run. In a dry run nothing is sent: the order mirror only writes
what it would have placed. A live run needs a key, and a live run on mainnet also needs MAINNET_CONFIRM=yes.

One network per run. NETWORK=testnet (Shannon) or mainnet runs spot markets and event contracts; NETWORK=hideki
runs perps, the only network that has them. Perps on Shannon or mainnet, and spot or event contracts on Hideki,
are refused with a message that says where each belongs.

The key is never read here beyond checking that it is set, and it is never stored or logged.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

NETWORKS = ("testnet", "hideki", "mainnet")
NETWORK_NAMES = {"testnet": "the Shannon testnet (chain 50312)", "hideki": "the Hideki testnet (chain 50383)",
                 "mainnet": "mainnet (chain 5031)"}
SPOT_NETWORKS = ("testnet", "mainnet")       # spot pools and event contracts live here
PERP_NETWORK = "hideki"                      # perp markets live here, and only here
PERP_TAPE_PREFIX = "hideki_perp_"            # the recorder's tape files for that network
SPOT_DEFAULT = ("SOMI:USDso", "WETH:USDso", "WBTC:USDso")
MAINNET_ONLY_MARKETS = ("USDC.e:USDso",)     # the testnet has no pool for these
PERP_DEFAULT = ("BTC/USDso:USDso", "ETH/USDso:USDso", "SOL/USDso:USDso")


LIVE_DAILY_STOP = 50.0


class SettingsError(ValueError):
    """The environment asks for something unsafe or unreadable."""


def _flag(raw: str | None, default: bool, name: str) -> bool:
    if raw is None or raw.strip() == "":
        return default
    s = raw.strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"{name}={raw!r} is not true or false")


def _number(raw: str | None, default: float, name: str, lo: float, hi: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        v = float(raw)
    except ValueError as e:
        raise SettingsError(f"{name}={raw!r} is not a number") from e
    if not lo <= v <= hi:
        raise SettingsError(f"{name}={v} is outside {lo} to {hi}")
    return v


def _list(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _looks_like_a_key(raw: str | None) -> bool:
    """Set, and not the `0x...` placeholder a generated .env ships with. Only the shape is checked; the value is never
    kept or printed."""
    v = (raw or "").strip()
    return bool(v) and not v.endswith("...") and v != "0x"


@dataclass(frozen=True)
class Settings:
    network: str = "testnet"
    dry_run: bool = True
    has_key: bool = False
    mainnet_confirm: bool = False
    live_stake_usdso: float = 20.0      # notional the live orders are scaled to, per market
    max_open_orders: int = 6            # resting orders per market
    take_buffer_bps: float = 10.0       # a taker is priced this far through the touch (0.1%)
    requote_bps: float = 5.0            # a resting order within this of the wanted price is left alone
    min_gap_s: float = 2.0              # least time between two order syncs on one market
    quote_ttl_min: float = 90.0         # how long a resting order lives on the book
    markets: tuple[str, ...] = SPOT_DEFAULT
    perp_markets: tuple[str, ...] = ()
    ec: bool = False
    concurrent_episodes: int = 3        # paper episodes at once per spot market, one shared brain
    brain: str = "delta"                # how the spot flies learn: "delta" (per-strategy delta rule) or "mb" (original)
    owner_address: str = ""             # perps: the DreamDEX account a linked bot key trades for (blank: the key's own account)
    perp_leverage: float = 2.0          # perps: live notional per market = LIVE_STAKE_USDSO x this

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        network = (e.get("NETWORK") or "testnet").strip().lower()
        s = cls(
            network=network,
            dry_run=_flag(e.get("DRY_RUN"), True, "DRY_RUN"),
            has_key=_looks_like_a_key(e.get("PRIVATE_KEY")),
            mainnet_confirm=_flag(e.get("MAINNET_CONFIRM"), False, "MAINNET_CONFIRM"),
            live_stake_usdso=_number(e.get("LIVE_STAKE_USDSO"), 20.0, "LIVE_STAKE_USDSO", 1.0, 100_000.0),
            max_open_orders=int(_number(e.get("MAX_OPEN_ORDERS"), 6, "MAX_OPEN_ORDERS", 1, 50)),
            take_buffer_bps=_number(e.get("TAKE_BUFFER_BPS"), 10.0, "TAKE_BUFFER_BPS", 0.0, 200.0),
            requote_bps=_number(e.get("REQUOTE_BPS"), 5.0, "REQUOTE_BPS", 0.0, 500.0),
            min_gap_s=_number(e.get("MIN_ORDER_GAP_S"), 2.0, "MIN_ORDER_GAP_S", 0.0, 600.0),
            quote_ttl_min=_number(e.get("QUOTE_TTL_MIN"), 90.0, "QUOTE_TTL_MIN", 1.0, 1440.0),
            markets=_list(e.get("MARKETS"), () if network == PERP_NETWORK else SPOT_DEFAULT),
            perp_markets=_list(e.get("PERP_MARKETS"), ()),
            ec=_flag(e.get("EC_ENABLED"), False, "EC_ENABLED"),
            concurrent_episodes=int(_number(e.get("CONCURRENT_EPISODES"), 3, "CONCURRENT_EPISODES", 1, 8)),
            brain=(e.get("BRAIN") or "delta").strip().lower(),
            owner_address=(e.get("OWNER_ADDRESS") or "").strip(),
            perp_leverage=_number(e.get("PERP_LEVERAGE"), 2.0, "PERP_LEVERAGE", 1.0, 10.0),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if self.network not in NETWORKS:
            raise SettingsError(f'NETWORK={self.network!r} is not one of {", ".join(NETWORKS)}: '
                                'testnet (Shannon) or mainnet for spot and event contracts, hideki for perps')
        if self.brain not in ("delta", "mb"):
            raise SettingsError(f'BRAIN={self.brain!r} is not delta or mb')
        # one network per run: perps live on Hideki only, spot pools and event contracts on Shannon and mainnet only
        where = NETWORK_NAMES[self.network]
        if self.perp_markets and self.network != PERP_NETWORK:
            raise SettingsError(f"PERP_MARKETS is set but NETWORK={self.network} is {where}, which has no perp markets: "
                                f"perps trade on {NETWORK_NAMES[PERP_NETWORK]} only. Run them with NETWORK={PERP_NETWORK} "
                                "in a run of their own, or clear PERP_MARKETS here")
        if self.network == PERP_NETWORK:
            if self.markets:
                raise SettingsError(f"MARKETS={','.join(self.markets)} cannot run on NETWORK={PERP_NETWORK}: {where} has "
                                    "perp markets only. Spot runs with NETWORK=testnet or mainnet; leave MARKETS blank here")
            if self.ec:
                raise SettingsError(f"EC_ENABLED=true cannot run on NETWORK={PERP_NETWORK}: {where} has perp markets only. "
                                    "Event contracts run with NETWORK=testnet or mainnet")
            if not self.perp_markets:
                raise SettingsError(f"NETWORK={PERP_NETWORK} is for perps and PERP_MARKETS is empty: set it to all, or to a "
                                    "list such as BTC/USDso:USDso,ETH/USDso:USDso")
        if self.owner_address and not re.fullmatch(r"0x[0-9a-fA-F]{40}", self.owner_address):
            raise SettingsError("OWNER_ADDRESS must be an address (0x and 40 hex characters), or blank to trade the key's own account")
        if self.network == "testnet":
            bad = [m for m in self.markets if m in MAINNET_ONLY_MARKETS]
            if bad:
                raise SettingsError(f'{", ".join(bad)} exists on mainnet only: set NETWORK=mainnet or take it out of MARKETS')
        if not self.dry_run:
            if not self.has_key:
                raise SettingsError("DRY_RUN=false needs PRIVATE_KEY set in the environment")
            if self.network == "mainnet" and not self.mainnet_confirm:
                raise SettingsError("live orders on mainnet need MAINNET_CONFIRM=yes as well as DRY_RUN=false")

    @property
    def live(self) -> bool:
        return not self.dry_run

    @property
    def perp_live(self) -> bool:
        """Real perp orders: live orders are on and perp markets are configured."""
        return self.live and bool(self.perp_markets)

    @property
    def ec_live(self) -> bool:
        """Real event contract bets: live orders on mainnet with event contracts enabled. The windows exist on
        mainnet only, so on the testnet event contracts stay paper whatever DRY_RUN says."""
        return self.live and self.ec and self.network == "mainnet"

    def daily_loss_stop(self, explicit: float | None = None) -> float:
        """The daily loss, in USDso, after which spot trading is forced flat for the rest of the day.

        It protects real money, so it applies only to live orders. In paper trading and dry runs a loss is just
        a number, and stopping would switch learning off, so there is no stop (infinity) unless one is asked for.
        """
        if explicit is not None:
            return explicit
        return LIVE_DAILY_STOP if self.live else float("inf")

    def summary(self) -> str:
        mode = "LIVE orders" if self.live else "dry run (no orders sent)"
        what = "perps" if self.network == PERP_NETWORK else "spot"
        return f"{self.network} ({what}), {mode}, stake {self.live_stake_usdso:g} USDso per market"
