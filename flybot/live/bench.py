"""The fly's memory of which strategies keep losing on a market, so it stops running them for a while.

The mushroom body learns slowly and keeps a little random exploration forever. On the live paper run that meant one
strategy (the maker) was tried again and again and lost every time: 36 losses in a row across three markets, about
all of the run's paper loss, while the other strategies sat near zero. A bot stuck on a losing strategy is exactly
what the fly is meant to avoid, so this puts a plain limit on it.

An arm is benched when its last `after` episodes that actually traded were all losses and together lost more than
`loss_frac` of the stake. Episodes that sent no orders are ignored (gas for orders that never filled still counts: it is real money). A benched arm is not offered to the
fly for `hours`. When the bench ends the arm gets one trial episode: if that loses again the bench restarts at twice
the length (up to `max_hours`), if it does not lose the arm is clean again. Flat is never benched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HOUR_MS = 3_600_000


@dataclass
class ArmBench:
    after: int = 3                 # losing traded episodes in a row before an arm is benched
    loss_frac: float = 0.001       # ... and together they must lose more than this fraction of the stake
    hours: float = 12.0            # the first bench; each failed trial doubles it
    max_hours: float = 168.0
    stake: float = 1000.0
    recent: dict[str, list[float]] = field(default_factory=dict)
    until: dict[str, int] = field(default_factory=dict)       # arm -> when its bench ends (ms)
    strikes: dict[str, int] = field(default_factory=dict)     # failed trials since the arm was last clean
    trial: set[str] = field(default_factory=set)              # arms on a trial episode right now

    @classmethod
    def from_params(cls, p, stake: float) -> ArmBench | None:
        """p is the market's "bench" setting: absent or true for the defaults, false to switch it off, or a dict."""
        if p is False:
            return None
        p = p if isinstance(p, dict) else {}
        return cls(after=int(p.get("after", 3)), loss_frac=float(p.get("loss_frac", 0.001)), hours=float(p.get("hours", 12.0)),
                   max_hours=float(p.get("max_hours", 168.0)), stake=stake)

    def blocked(self, now_ms: int) -> set[str]:
        """Arms that must not be chosen right now."""
        return {a for a, t in self.until.items() if t > now_ms}

    def note_start(self, arm: str, now_ms: int) -> None:
        """An episode of `arm` is starting. If its bench has run out this is its trial."""
        t = self.until.get(arm)
        if t is not None and t <= now_ms:
            del self.until[arm]
            self.trial.add(arm)

    def record(self, arm: str, pnl: float, traded: bool, now_ms: int) -> dict | None:
        """An episode ended. `traded`: it sent at least one order. Returns an event row when an arm was benched, else None."""
        if arm == "flat" or not traded:
            return None
        if arm in self.trial:
            self.trial.discard(arm)
            if pnl < -self.loss_frac * self.stake / 10:        # the trial lost again
                self.strikes[arm] = self.strikes.get(arm, 0) + 1
                return self._bench(arm, now_ms, f"trial lost {pnl:.3f}")
            self.strikes.pop(arm, None)
            self.recent.pop(arm, None)
            return None
        r = self.recent.setdefault(arm, [])
        r.append(pnl)
        del r[: -self.after]
        if len(r) >= self.after and all(x < 0 for x in r) and sum(r) < -self.loss_frac * self.stake:
            why = f"{self.after} losses in a row, {sum(r):.3f} together"
            self.recent.pop(arm, None)
            return self._bench(arm, now_ms, why)
        return None

    def _bench(self, arm: str, now_ms: int, why: str) -> dict:
        hours = min(self.hours * 2 ** self.strikes.get(arm, 0), self.max_hours)
        self.until[arm] = int(now_ms + hours * HOUR_MS)
        return {"arm": arm, "hours": hours, "until": self.until[arm], "why": why, "strikes": self.strikes.get(arm, 0)}

    def to_dict(self) -> dict:
        return {"recent": self.recent, "until": self.until, "strikes": self.strikes, "trial": sorted(self.trial)}

    def load(self, d: dict) -> None:
        self.recent = {a: [float(x) for x in v] for a, v in d.get("recent", {}).items()}
        self.until = {a: int(t) for a, t in d.get("until", {}).items()}
        self.strikes = {a: int(n) for a, n in d.get("strikes", {}).items()}
        self.trial = set(d.get("trial", []))
