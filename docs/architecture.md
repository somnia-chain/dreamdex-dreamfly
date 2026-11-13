# Architecture

```
 DreamDEX public WebSocket --> book touch ------------------+
 DreamDEX REST candles --> features (8 numbers) --+         |
                                                  v         v
                          +------------- fly (one per market) -------------+
                          | odour > Kenyon cells > output pools > strategy |
                          +----------------------+-------------------------+
                                                 | picks maker, grid, trend or flat
                                                 v
                                   paper episode (simulated fills, P&L)
                                        |                    |
                          reward <------+                    +--> order mirror --> dry-run log
                            |                                                 +--> live orders (spot)
                            v
                  dopamine changes the weights        live.json --> dashboard
```

## Pieces

| Piece | Where | What it does |
| --- | --- | --- |
| Circuit | `flybot/mb/` | The mushroom body subgraph from the Male CNS v1.0 connectome, and a rate model with dopamine-gated depression of the Kenyon cell to output synapses. |
| Features and odour | `flybot/market.py`, `flybot/perp.py`, `flybot/live/ec_live.py` | Turn candles, perp rows or event contract books into a small vector, then into projection neuron rates. |
| Selector | `flybot/selectors.py` | The fly, plus a bandit, a rule switcher and random choice for comparison. |
| Strategies | `flybot/arms.py`, `flybot/perp.py`, `flybot/ec.py` | Hand-written: maker, grid, trend, flat for spot; perp maker, trend, carry, flat; event contract late, sharp, flat. |
| Paper episode | `flybot/live/paper.py` | Runs a strategy against the live touch for an episode and simulates fills. |
| Order mirror | `flybot/live/execution.py` | Copies the episode's intentions into dry-run log lines or real orders. |
| Runner | `flybot/live/runner.py` | Order book and candle feeds, episode boundaries, the reward, saving state. |
| Supervisor | `flybot/live/supervisor.py` | Runs the runner, the dashboard and the recorders as one process. |
| Recorders | `perp/` (Node) | Read-only tapes of perp markets and event contract books, which have no public feed. |
| Dashboard | `site/`, `scripts/serve_site.py` | A three.js scene and a paper terminal, reading `live.json`. |

## One fly per instrument

Every market has its own fly with its own weights. Spot has one per market, perps one per market, and event
contracts one for all windows. They do not share learning. The dashboard builds its tabs from what the runner
reports, so it shows only the markets you run.

## Several episodes at once

Each spot market can run a few paper episodes at the same time (`CONCURRENT_EPISODES`, default 3). They start
staggered through the episode length, and a new one avoids the strategies the others are already running, so the
slots cover different strategies. Every finished episode trains the same weights, remembering the choice it was
made from. Paper episodes do not move the market, so they are independent experiments on the same price path.
Slot 0 is the primary one: it is what the order mirror follows and what the dashboard shows.

## How a decision is made

Once per episode (60 minutes by default) the fly sees the market features, which become an odour. The Kenyon cells
that fire choose an output pool, and each pool stands for one strategy. Between decisions the fly keeps sniffing
the market so the dashboard stays alive, but it only decides once.

At the end of the episode the paper profit, divided by a volatility unit and scaled by a running reward scale,
becomes dopamine. What the dopamine does depends on the brain (`BRAIN`):

- `delta` (default): each strategy keeps a weight per Kenyon cell. Its value for the current odour is the sum of
  the weights of the cells that fire. Dopamine is the error, reward minus that value, and the chosen strategy's
  weights move toward the reward. Flat is worth zero and is never taught. Because weights go up as well as down, a
  strategy can be learned good in one market state and bad in another.
- `mb`: the original rule. Dopamine only weakens the Kenyon cell to output synapses of the chosen strategy's
  pool, and a pool's score is how much that learning changed it. Regimes share many Kenyon cells, so this mostly
  learned each strategy's overall value, not its value in the current state.

Both read the same Kenyon cell code. `scripts/warm_delta.py` replays recorded candles through the `delta` brain,
teaching every strategy on every episode, and the kit copies the result (`config/warm_delta`) into a fresh state
directory. The brains learn in separate weights files (`w_delta.npy`, `w_kc_mbon.npy`) and cannot share them.
Event contracts and perps always use `mb`.

A strategy that keeps losing is benched (`flybot/live/bench.py`): after three losing episodes that traded, it is
not offered for 12 hours, then gets one trial episode.

## Threads

The runner is asynchronous. Order actions are slow, because a live order waits for the chain, so they run on
worker threads and a market's sync is skipped if the previous one is still running. Chain writes share one lock.

## State

Everything the fly learns is in `state/live/<market>/`: the weights (`w_kc_mbon.npy`), the reward scale and
counters (`state.json`), and logs (`decisions`, `episodes`, `fills`, `orders`, `errors` as `.jsonl`). Delete a
market's folder and it starts again from the bundled warm-started brain.
