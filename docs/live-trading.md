# Dry run and live trading

The fly always simulates an episode: a paper strategy runs for an hour against the live order book, and its
profit is what the fly learns from. The order mirror sits next to that simulation and copies the strategy's
intentions into orders.

| Mode | Settings | What happens |
| --- | --- | --- |
| Dry run (default) | `DRY_RUN=true` | The mirror writes each order it would place or cancel to `state/live/<market>/orders.jsonl`. Nothing is sent. |
| Live on testnet | `DRY_RUN=false`, `NETWORK=testnet`, `PRIVATE_KEY` | Real orders on the test network. |
| Live on mainnet | as above with `NETWORK=mainnet` and `MAINNET_CONFIRM=yes` | Real orders with real funds. |
| Live perps | `DRY_RUN=false`, `NETWORK=hideki`, `PERP_MARKETS`, `PRIVATE_KEY` | Real perp orders on the Hideki testnet, in a run of their own. |
| Live event contracts | `DRY_RUN=false`, `NETWORK=mainnet`, `MAINNET_CONFIRM=yes`, `EC_ENABLED=true`, `PRIVATE_KEY` | Real bets on the mainnet windows (they exist nowhere else; on testnet event contracts stay paper). |

The runner refuses to start if a live setting is missing, so a half-configured live run cannot happen by accident.
It also keeps to one network per run: perps exist on Hideki only, spot pools and event contracts on Shannon and
mainnet only, and a mix (perps with `NETWORK=testnet` or `mainnet`, spot with `NETWORK=hideki`) is refused with a
message that says where each belongs.

## What the mirror does

For each spot market, while an episode is running:

- **Maker and grid:** the strategy's resting quotes are kept on the book as post-only orders. A quote within
  `REQUOTE_BPS` of the wanted price is left alone, so small moves do not cause a cancel and a new order.
- **Trend:** one taker order at the start, priced `TAKE_BUFFER_BPS` through the touch so it still fills.
- **Flat:** nothing.
- **At the end of the episode:** every resting order is cancelled and whatever the wallet holds beyond what it
  held when the fly was last flat is closed with a taker order. If that order misses because the book moved, it is
  sent once more, further through the touch; anything still left is logged as `close_left` and closed at the next
  episode end. Dust under one USDso is left alone. A cancel that fails is retried; the order expires on its own
  after `QUOTE_TTL_MIN` in any case.

Sizes are scaled to `LIVE_STAKE_USDSO`, so one market never has more than about that much at work, whatever stake
the simulation uses. There are at most `MAX_OPEN_ORDERS` resting orders per market, and no single order is larger
than the stake.

Orders go through the DreamDEX core client (`dreamdex_core`, from the DreamDEX bot kit), which checks tick and lot
size, expiry and price, sizes the funding pull, simulates the order and only then sends it. Chain writes are
sent one at a time so nonces do not collide.

## Perps

With `NETWORK=hideki`, `PERP_MARKETS` set and `DRY_RUN=false` the perp flies send real orders too, on the Hideki
test network, the only one with perp markets (Shannon and mainnet have none, and the runner refuses perps there).
The orders are signed by a small Node sidecar
(`perp/executor.mjs`) built on the same markets SDK the bot kit's `perp-*` strategies use; the runner starts it and
talks to it over a pipe, one request at a time.

Two ways to hold the margin, the same two the kit documents:

- **Linked bot key (recommended):** `PRIVATE_KEY` is a fresh key you linked to your DreamDEX account in the app
  (wallet, **Link a bot**, tick **Perps**) and `OWNER_ADDRESS` is your account. Orders, margin and positions are the
  account's; the key only signs and pays gas (it needs STT). Move USDso into your **Perps account** in the app first
  (**Perps ⇆ Spot**). The key cannot withdraw. A market listed after you linked needs the address added again.
- **The key's own account:** leave `OWNER_ADDRESS` blank. The key then needs USDso in its own Perps balance.
  `scripts/perp_smoke.py --deposit 50` moves USDso from the key's wallet into it.

What the perp mirror does, per market, while an episode runs:

- **Trend and carry:** the paper position is the target. The mirror reads the account's position and sends one
  taker order for the difference, priced 20 bps through the touch. A guard flatten in the simulation, and the close
  at the end of the episode, are the same move with a target of zero. A paying carry position is kept open across
  episodes, as in the simulation, and closed when the rate falls or the next strategy does not want it.
- **Maker:** the two paper quotes are placed as post only orders and re-quoted when the paper quote moves by more
  than `REQUOTE_BPS`. Real fills are the account's own; the paper fills are not chased. The position is closed at
  the end of the episode.
- **Flat, and every episode start:** resting orders the account still has in the market are cancelled, and a
  position the new episode does not own is closed. So a restart cannot leave a forgotten position open.

Sizes: the live margin per market is `LIVE_STAKE_USDSO` and the notional is that times `PERP_LEVERAGE` (default 2),
so the default is 40 USDso of exposure per market. Every order is first sized by the pool itself
(`getMaxPerpOrderSize`), which caps it at what the account's margin can back. A maker quote smaller than the
market's minimum size (BTC's is about 11 USDso) is lifted to that minimum when the per-market notional covers it;
any other order under the minimum is not sent, and the order log says so. With `PERP_MARKETS=all` every market the key may trade is live, so the margin need is the
number of markets times the stake. A linked account's own leverage cap, set in the app, still applies.

The daily loss stop counts the perps account's equity (margin plus unrealised P&L) together with the spot wallet.
Before the first live run: `scripts/perp_smoke.py --arm p_trend --seconds 60` on one market with a throwaway key.

## Event contracts

With `EC_ENABLED=true`, `NETWORK=mainnet` and `DRY_RUN=false` (plus `MAINNET_CONFIRM=yes`) the event contract fly
bets for real. The windows exist on mainnet only, so on the testnet event contracts stay paper whatever `DRY_RUN`
says. The bets are signed by a second small Node sidecar (`perp/ec_executor.mjs`) on the same markets SDK the bot
kit's `ec-*` strategies use, and it follows the kit's hard-won rules: prices and sizes reach the pool as exact
integers on the tick grid, every order carries an expiry no later than the window's, a reverted write is caught
from the receipt, and winnings are claimed because a settled market pays only when asked.

What the mirror does, per window:

- **One bet, taker only.** When the paper fly would send its order (a seller visible at or under its cap in the
  last seconds of the window), the mirror sends one immediate-or-cancel buy of the same side at the same cap for
  `LIVE_STAKE_USDSO`. Nothing rests. If nothing crosses it tries again on the next sample, at most three times.
- **No early exit.** A bought share rides to the window's end. A win pays 1 per share less the venue's settlement
  fee, a loss pays 0, a void pays 0.5.
- **Claiming.** Each settled window is claimed as soon as the tape reports its resolution, retried every two
  minutes until the venue has it finalised, and the venue's own claimable list is swept at start and every few
  hours so a restart never leaves winnings behind.
- **Bets are remembered on disk** (`state/live/EC/ec_bets.json`), so a restart knows which windows are its own.

The wallet ledger counts an open or unclaimed bet at cost, so the daily stop sees a lost bet when the claim
comes back empty and a won one when the payout lands. The stop can only refuse new bets; it cannot close one.
Before the first live run: `scripts/ec_smoke.py` on mainnet with a throwaway key and a few USDso. It waits for a
window in the action band, bets 2 USDso on the book's favourite, waits for the resolution and claims.

## Reading the order log

```json
{"mode": "dry", "action": "place", "id": "dry-1", "symbol": "WETH:USDso", "side": "buy", "price": 3021.4, "qty": 0.0016, "kind": "post_only"}
{"mode": "live", "action": "place", "symbol": "WETH:USDso", "side": "buy", "price": 3021.4, "qty": 0.0016, "kind": "post_only", "ok": true, "id": "812", "tx": "0x...", "gas": 412345}
{"mode": "live", "action": "cancel", "symbol": "WETH:USDso", "id": "812", "ok": false, "error": "..."}
```

A failed cancel usually means the order already filled or expired. A rejected placement is logged with the
reason and the run carries on.

## Before you go live

1. Run in dry run first and read `orders.jsonl` for a day. Do the orders look like something you would place?
2. Run `dreamfly doctor --wallet` to check the address and balances.
3. Go live on testnet. Get test funds at the DreamDEX testnet faucet.
4. Use a fresh wallet with a small balance. The key sits in an environment variable of the process.
5. Keep `LIVE_STAKE_USDSO` small. Watch the first episodes.
6. To stop trading at any time, create the file `state/live/KILL`. Within a minute the runner cancels the resting
   quotes, closes any position and opens no new episode (it keeps running, flat, so the dashboard stays up). Remove the
   file to let it trade again. To stop the process itself, send it SIGTERM.
7. Test the order path on its own first with a throwaway testnet wallet: `scripts/testnet_smoke.py` places, requotes and
   cancels real test orders for one strategy and prints the wallet before and after.

With live orders there is also a daily loss stop: after a 50 USDso loss in a day (change it with
`--daily-loss-stop`) the spot flies are forced flat until the next day. The loss is measured on the wallet, not on
the simulation: USDso held plus each market's inventory at its mid, compared with the first reading of the UTC day,
and all markets count against the one wallet. Gas is not counted, and a deposit or withdrawal during the day moves
the number until the next day. The current value is in the dashboard state as `wallet_day_pnl`. It protects real
money, so in a dry run and in paper trading there is no stop and the fly learns from every episode.

## Limits of this version

- **The reward is simulated.** The fly learns from the simulated episode, not from your real fills. Real fills
  can differ because of queue position, latency and fees.
- **Inventory is read from the wallet.** Filled orders are found by comparing the wallet balance at the start and
  the end of an episode, not by tracking each order.
- **An event contract bet has no early exit.** Once bought it rides to the window's end; the daily stop can only
  block new bets. Real fills happen on the sample the paper order goes out, so the real price can differ from the
  paper fill by a tick or two.
- **Perp fills are the account's own.** A maker episode's real fills differ from its paper fills, so a position
  can build on chain that the simulation never saw. It is closed at the end of every episode.
- **Native SOMI.** On the SOMI market the base asset is the gas token, so gas spent moves the balance the mirror
  reads. Prefer WETH or WBTC for a first live run.
- **One key.** Session keys (operator trading) are not supported.
- **No edge is claimed.** Read [DISCLAIMER.md](../DISCLAIMER.md).
