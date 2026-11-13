"""One fly per perp market, fed by the testnet perp recorder tape (one row per minute)."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np

from ..perp import LONG, PERP_ARMS, PERP_FEATURES, PerpCosts, PerpPaperEpisode, perp_feature_table, rows_to_frame, trend_direction
from ..sim import reward_of
from .base import FlyState
from .bench import ArmBench

KEEP_ROWS = 600


class PerpFly(FlyState):
    def __init__(self, symbol: str, graph, state_dir: Path, costs: PerpCosts, params: dict, seed: int = 0,
                 episode_min: int = 60, daily_loss_stop: float = 50.0, brain: str = "mb", shared: FlyState | None = None):
        super().__init__(symbol, graph, state_dir, seed, PERP_ARMS, PERP_FEATURES, brain)
        # one brain can learn from many markets: then this market keeps only its own books and its own last choice
        self.shared = shared
        if shared is not None:
            self.fly, self.scaler = shared.fly, shared.scaler
        self.choice = None      # (strategy, cells, score) of this market's running episode
        self.symbol, self.costs, self.params = symbol, costs, params
        self.episode_min, self.daily_loss_stop = episode_min, daily_loss_stop
        self.rows: list[dict] = []
        self.frame = None
        self.episode: PerpPaperEpisode | None = None
        self.episode_start = 0
        self.last_row: dict = {}
        self.open_pos = 0.0        # carry position left open by the last episode (base units, signed); lost on restart
        self.mirror = None         # PerpMirror when real (or dry run) perp orders follow this fly's episodes
        self.day_loss = lambda: self.day_pnl    # what the daily stop compares; the runner points it at the wallet when live
        # a strategy that keeps losing here is left out for a while (params "bench": false switches it off)
        self.bench = ArmBench.from_params(params.get("bench"), costs.stake)
        if self.bench is not None and (self.dir / "bench.json").exists():
            try:
                self.bench.load(json.loads((self.dir / "bench.json").read_text()))
            except (ValueError, OSError):
                pass        # an unreadable bench file just means a clean slate

    def on_row(self, row: dict, kill: bool = False) -> dict | None:
        """A new tape row for this market. Returns the finished episode row, if one ended."""
        self.rows.append(row)
        self.rows = self.rows[-KEEP_ROWS:]
        self.frame = rows_to_frame(self.rows)
        if self.frame.empty:
            return None
        cur = self.frame.iloc[-1]
        ts = int(cur["timestamp"])
        self.last_row = {"mark": float(cur["mark"]), "index": None if not np.isfinite(cur["index"]) else float(cur["index"]),
                         "funding_8h": None if not np.isfinite(cur["funding_8h"]) else float(cur["funding_8h"]),
                         "best_bid": None if not np.isfinite(cur["best_bid"]) else float(cur["best_bid"]),
                         "best_ask": None if not np.isfinite(cur["best_ask"]) else float(cur["best_ask"]), "ts": ts}
        done = None
        if len(self.frame) >= LONG // 2:
            self.note_sniff(perp_feature_table(self.frame).iloc[-1], ts)
        if self.episode is not None:
            self.episode.on_row(cur)
            self._mirror("on_row", self.episode)
            if ts - self.episode_start >= self.episode_min * 60_000 or kill:
                done = self.finish(cur)
        if self.episode is None and not kill and len(self.frame) >= LONG // 2:
            self.decide(cur)
        return done

    def decide(self, cur) -> str:
        feats = perp_feature_table(self.frame).iloc[-1]
        kc, scores = self.sniff(feats)
        ts = int(cur["timestamp"])
        held = frozenset(PERP_ARMS.index(a) for a in self.bench.blocked(ts) if a in PERP_ARMS) if self.bench else frozenset()
        k = self.fly.choose(feats, held or None)
        arm = PERP_ARMS[k]
        guard = self.day_loss() <= -self.daily_loss_stop
        if (guard and arm != "flat") or k in held:
            arm, k = "flat", PERP_ARMS.index("flat")
            self.fly.last = (k, self.fly.last[1], float(scores[k]))
        self.choice = self.fly.last
        if self.bench is not None:
            self.bench.note_start(arm, ts)
        trend_dir = trend_direction(self.frame, self.params.get("p_trend", {})) if arm == "p_trend" else 0
        self.episode = PerpPaperEpisode(arm, self.costs, self.params.get(arm, {}), inherit_pos=self.open_pos)
        self.episode.start(cur, trend_dir)
        self.episode_start = int(cur["timestamp"])
        self._mirror("on_start", self.episode)
        self.last_decision = {
            "ts": self.episode_start, "arm": arm, "scores": [round(float(x), 4) for x in scores],
            "kc_active": [int(i) for i in np.flatnonzero(kc > 0)], "kc_frac": round(float((kc > 0).mean()), 4),
            "features": {f: (None if not np.isfinite(feats[f]) else round(float(feats[f]), 6)) for f in PERP_FEATURES},
            "trend_dir": trend_dir, "guard_flat": guard,
        }
        self.log("decisions", self.last_decision)
        return arm

    def _mirror(self, hook: str, ep) -> None:
        """The order mirror follows the paper episode; its failures are logged, never allowed to stop the fly."""
        if self.mirror is None:
            return
        try:
            getattr(self.mirror, hook)(ep)
        except Exception as e:  # noqa: BLE001
            self.log("errors", {"where": f"mirror.{hook}", "error": str(e)[:1500]})

    def finish(self, cur) -> dict:
        ep = self.episode
        pnl = ep.finish(cur)
        self._mirror("finish", ep)
        win = self.frame[self.frame["timestamp"] >= self.episode_start].rename(columns={"mark": "close"})
        if len(win) < 3:
            win = self.frame.tail(self.episode_min).rename(columns={"mark": "close"})
        raw = reward_of(pnl, win, self.costs, clip=np.inf)
        rew = self.scaler(raw)
        if self.choice is not None:
            self.fly.last = self.choice     # a shared brain's last choice may be another market's
        self.fly.update(rew)
        taught = self.teach_others(ep.arm, self.choice[1], win) if self.choice is not None and self.params.get("teach_all") else []
        self.choice = None
        ts = int(cur["timestamp"])
        self.book(ts, pnl)
        row = {"start": self.episode_start, "end": ts, "arm": ep.arm, "pnl": round(pnl, 6), "reward": round(rew, 4),
               "raw_reward": round(raw, 6), "fills": len(ep.fills), "actions": ep.actions, "guarded": ep.guarded,
               "funding": round(ep.funding_paid, 6), "equity": round(self.equity, 6),
               "dopamine": "PAM" if rew > 0 else "PPL1" if rew < 0 else "none", "weights_moved": self.weights_moved()}
        if taught:
            row["taught"] = taught
        self.log("episodes", row)
        for f in ep.fills:
            self.log("fills", {"arm": ep.arm, **f.__dict__})
        self.episode = None
        self.open_pos = ep.pos if ep.keep_open else 0.0     # a paying carry position carries into the next episode
        if self.bench is not None:
            event = self.bench.record(ep.arm, pnl, len(ep.fills) > 0 or ep.actions > 0, ts)
            if event:
                self.log("bench", {"ts": ts, **event})
            (self.dir / "bench.json").write_text(json.dumps(self.bench.to_dict()))
        self.save()
        return row

    def teach_others(self, ran: str, kc: np.ndarray, win) -> list[str]:
        """The strategies this episode did not run learn what they would have made on the same tape rows, each taught
        as if it had been chosen from the same market state (spot's teach_all, on the perp paper engine). Flat is
        worth zero and never taught. Off unless the market's params set `"teach_all": true`."""
        if len(win) < 2:
            return []
        rows = win.rename(columns={"close": "mark"})
        hist = self.frame[self.frame["timestamp"] <= int(rows["timestamp"].iloc[0])]
        taught = []
        for k, arm in enumerate(PERP_ARMS):
            if arm in (ran, "flat"):
                continue
            p = self.params.get(arm, {})
            ep = PerpPaperEpisode(arm, self.costs, p)
            ep.start(rows.iloc[0], trend_direction(hist, p) if arm == "p_trend" else 0)
            for i in range(1, len(rows)):
                ep.on_row(rows.iloc[i])
            pnl = ep.finish(rows.iloc[-1])
            self.fly.last = (k, kc, 0.0)
            self.fly.update(self.scaler(reward_of(pnl, win, self.costs, clip=np.inf)))
            taught.append(arm)
        return taught

    def save(self) -> None:
        """A market on a shared brain keeps only its own books here; the brain itself is saved by its owner."""
        if self.shared is None:
            return super().save()
        meta = {"scaler": {"n": self.scaler.n, "scale": self.scaler.scale}, "equity": self.equity, "day": self.day,
                "day_pnl": self.day_pnl, "n_episodes": self.n_episodes}
        (self.dir / "state.json").write_text(json.dumps(meta))

    def series(self, n: int = 60) -> list[dict]:
        """The last n tape rows as one-minute candles (open = previous mark) for the page's chart."""
        if self.frame is None or len(self.frame) < 2:
            return []
        m = self.frame["mark"].to_numpy()[-(n + 1):]
        t = self.frame["timestamp"].to_numpy()[-(n + 1):]
        return [{"t": int(t[i]), "o": float(m[i - 1]), "h": float(max(m[i - 1], m[i])), "l": float(min(m[i - 1], m[i])),
                 "c": float(m[i]), "v": 0.0} for i in range(1, len(m))]

    def snapshot(self) -> dict:
        ep = self.episode
        return {"symbol": self.symbol, "kind": "perp", **self.brain_snapshot(), "last": self.last_row, "series": self.series(),
                "live": None if ep is None else {"arm": ep.arm, "pos": ep.pos, "cash": round(ep.cash, 6),
                                                  "bids": ep.bids, "asks": ep.asks, "fills": len(ep.fills),
                                                  "unrealised": round(ep.unrealised(ep.last_mark), 6), "guarded": ep.guarded}}
