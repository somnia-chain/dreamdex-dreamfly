# Configuration

Settings come from environment variables. A `.env` file in the working directory is loaded first, and real
environment variables win over it. `dreamfly doctor` prints what was read.

## Network and safety

| Variable | Default | Meaning |
| --- | --- | --- |
| `NETWORK` | `testnet` | One network per run. `testnet` (Shannon, chain 50312) or `mainnet` (chain 5031) run spot markets and event contracts. `hideki` (chain 50383) runs perps, the only network that has them: set `PERP_MARKETS` and leave `MARKETS` blank. `PERP_MARKETS` on testnet or mainnet, and `MARKETS` or `EC_ENABLED` on hideki, are refused at start with a message that says where each belongs. Spot plus perps means two runs, each with its own `.env`. |
| `DRY_RUN` | `true` | `true` logs the orders it would place and sends nothing. `false` places real spot orders. |
| `PRIVATE_KEY` | unset | Needed only when `DRY_RUN=false`. Read by the DreamDEX core client, never logged. |
| `MAINNET_CONFIRM` | `false` | Live orders on mainnet also need `MAINNET_CONFIRM=yes`. |

## What runs

| Variable | Default | Meaning |
| --- | --- | --- |
| `MARKETS` | `SOMI:USDso,WETH:USDso,WBTC:USDso` | Spot markets, comma separated. One fly each. Testnet has SOMI, WETH and WBTC. Mainnet also has `USDC.e:USDso`, the stablecoin pair (see below); it is refused on testnet. |
| `PERP_MARKETS` | empty | Perp markets to trade, with `NETWORK=hideki`: `all` for every market the recorder tape carries (21 on Hideki), or a list such as `BTC/USDso:USDso,ETH/USDso:USDso`. Needs Node. Paper in a dry run; with `DRY_RUN=false` real perp orders follow the episodes (see [live-trading.md](live-trading.md#perps)). |
| `OWNER_ADDRESS` | empty | Perps: your DreamDEX account when `PRIVATE_KEY` is a bot key you linked to it in the app (wallet, Link a bot, Perps). Blank: the key trades its own perps account. |
| `PERP_LEVERAGE` | `2` | Perps: live notional per market = `LIVE_STAKE_USDSO` x this, 1 to 10. The pool still caps every order at what the account's margin backs. |
| `EC_ENABLED` | `false` | Trade the mainnet event contract windows (BTC and ETH Up/Down). Needs Node. Paper in a dry run and on the testnet; with `NETWORK=mainnet` and `DRY_RUN=false` real bets of `LIVE_STAKE_USDSO` each follow the fly (see [live-trading.md](live-trading.md#event-contracts)). |
| `CONCURRENT_EPISODES` | `3` | Paper episodes running at once on each spot market, all training the same brain. They start staggered through the hour and avoid the strategies the others are running, so the fly gets several times more experience per hour. With live orders only the first one places orders. |
| `BRAIN` | `delta` | How the spot flies learn which strategy to pick. `delta`: the same connectome wiring and cell code, but each strategy learns a value for each market state from reward minus what it expected, so it can be good in one state and bad in another. It starts from brains replayed over recorded candles (`config/warm_delta`, made by `scripts/warm_delta.py`). `mb`: the original rule, where connections only ever weaken. The two keep separate weights files (`w_delta.npy`, `w_kc_mbon.npy`) and do not share what they learned. |

The dashboard shows exactly what runs. Leave a market out and it has no tab.

## Per-market strategy settings

`config/arm_params.json` holds each market's strategy parameters. A market can also carry these optional keys:

| Key | Meaning |
| --- | --- |
| `arms` | The strategies this market may use (`maker`, `grid`, `trend`). `flat` is always allowed. Others are never picked. |
| `slots` | Concurrent paper episodes on this market (capped by the number of allowed arms). |
| `slip_bps` | This market's own taker cost in basis points, instead of the default 3. |
| `episode_min` | This market's episode length in minutes, instead of the runner's. A quiet market that keeps resting quotes can run long episodes, since every episode start and end costs order actions (gas). |
| `bench` | The arm bench, on by default. When a strategy's last 3 episodes that sent orders all lost, and together lost more than 0.1% of the stake, the fly stops offering it for 12 hours (the flat strategy is never benched). After the bench it gets one trial episode: a loss doubles the bench (up to 7 days), anything else clears it. Tune with `{"after": 3, "loss_frac": 0.001, "hours": 12, "max_hours": 168}`, or `false` to switch it off. State is kept in `bench.json` next to the market's weights and benches show as `benched` in the live snapshot. |
| `dense_candles` | For a market that trades only a few times an hour: build a full minute-by-minute series by carrying the last close through the quiet minutes, so the fly has enough history to look at. |

`config/launcher.json` lists every one of these settings with a label, default, type and help text, in the shape the
DreamDEX app's bot builder uses, so a step-by-step launcher can generate a DreamFly `.env` from it. A test keeps it in
step with what the kit reads.

A perp market's `p_carry` keys: `min_apr` (default 0.10) is the funding rate, as a yearly rate, above which carry
opens the side funding pays, and `exit_apr` (default 0.03) the rate under which it closes. A paying position is
held across episodes, and any other strategy closes it first. A perp market's `bench` works like the spot one.

The maker's `requote_bps` (default 5, like the live `REQUOTE_BPS`) is how far the wanted quote must move from the
resting one before the paper maker re-places it. Without it every touch change would cost gas for orders a real mirror
never sends.

The maker's own keys (under `maker`) switch on stable pair mode, off by default: `tick` (the price grid, quotes are
rounded onto it), `mode: "inside"` with `improve_ticks` (quote that many ticks inside the best prices instead of around
the last price), `trade_fills` (paper fills come from real trade prints with queue position, not from the touch) and
`stake` (the most it will hold, in quote units).

`USDC.e:USDso` ships configured this way: a maker one tick inside the spread, up to 250 USDso. Measured on 63 days of
real trade prints it earns about 0.13 USDso a day after gas with its 4 hour episodes (hourly episodes lose too much to
gas and flattening), because the pair barely moves and a maker is rarely picked off, but the pair trades only about
4,400 USDso a day. With live orders set `REQUOTE_BPS=0.5` for it, since the default leaves
an order alone while it is within 5 bps of the wanted price, which is wider than this strategy's whole edge.

## Live order limits (spot only)

| Variable | Default | Meaning |
| --- | --- | --- |
| `LIVE_STAKE_USDSO` | `20` | The notional one market can have at work. Order sizes are scaled to it. |
| `MAX_OPEN_ORDERS` | `6` | Most resting orders per market. |
| `TAKE_BUFFER_BPS` | `10` | A taker order is priced this far through the touch so it still fills after the order lands. |
| `REQUOTE_BPS` | `5` | A resting order within this of the wanted price is left alone, to avoid churn and gas. |
| `MIN_ORDER_GAP_S` | `2` | Least seconds between two order syncs on one market. |
| `QUOTE_TTL_MIN` | `90` | How long a resting order lives on the book. |

## Overrides

| Variable | Meaning |
| --- | --- |
| `RPC_URL`, `REST_API_URL`, `WS_URL` | Use your own endpoints instead of the network defaults. |
| `PORT` | Dashboard port. Default 8765. |
| `STATE_DIR` | Where the flies' weights, logs and live state are kept. Default `state/live`. |
| `DASHBOARD_PASSWORD` | If set, the dashboard asks for this password (any user name). |

## Runner flags

`dreamfly run --help` lists them. The useful ones: `--episode-min` (episode length, default 60), `--stake` (the
paper stake the episodes are simulated with, default 1000), `--daily-loss-stop` (a daily loss in USDso that forces
the spot flies flat for the rest of the day; the default is 50 with live orders and none in a dry run, because a
paper loss is only a number and stopping would switch learning off), `--params` (per-market strategy parameters, default `config/arm_params.json`),
`--no-orders` (turn the order mirror off entirely).

## Files

| Path | What it is |
| --- | --- |
| `config/arm_params.json` | Tuned strategy parameters per market. |
| `config/warm_delta/` | Warm-started weights for the `delta` brain, replayed over recorded candles. Copied into the state directory the first time a market runs, and never over what a fly has learned since. |
| `config/warm/` | The same for the original `mb` brain, and for event contracts, which always use it. |
| `state/live/<market>/` | Weights, reward scale, decisions, episodes, fills and `orders.jsonl`. |
| `state/live/live.json` | What the dashboard reads. |
| `state/live/KILL` | Create this file to stop trading: quotes are cancelled and positions closed, and no new episode opens. The runner keeps running, flat. |
