# Deploy on Railway

The container runs the fly, its order mirror and the dashboard together, so you can watch your fly from anywhere.
Unlike the DreamDEX bot kit workers, it serves a web page. With `MARKETS=WETH:USDso` the page shows that one
market and nothing else.

## Steps

1. Create a new Railway project from this repository. Railway reads `railway.toml` and builds the `Dockerfile`.
2. Open the service, then Variables, then RAW Editor, and paste `.env.railway`. It sets testnet and dry run. Railway's
   health check reads `/healthz`, which answers without the dashboard password.
3. Set `MARKETS` to what you want to run, for example `WETH:USDso`.
4. Add a volume mounted at `/data`. The fly keeps its state in `/data/state` whenever that mount exists, so
   `STATE_DIR` can stay unset. Without a volume the fly's learning is lost on every deploy, and it restarts from
   the bundled warm-started brain.
5. Generate a domain (Settings, Networking). Open it: you see your fly.
6. Set `DASHBOARD_PASSWORD` before you share the link, see below.

## Going live from Railway

Set these as Railway variables. Never put a key in a file or in the repo.

```
DRY_RUN=false
PRIVATE_KEY=0x...
MAINNET_CONFIRM=yes        # only if NETWORK=mainnet
```

Read [live-trading.md](live-trading.md) first. Start with `NETWORK=testnet` and a small `LIVE_STAKE_USDSO`.

## The dashboard is public unless you protect it

The page shows the fly's decisions, paper equity, open quotes and, in live mode, the orders it mirrors. It has no
login by default. Set `DASHBOARD_PASSWORD` and the browser asks for it (any user name). Keep the private key out
of the page: it is never sent to it.

## Perps and event contracts

Add `EC_ENABLED=true` and the container also starts the read-only event contract recorder for the service's
network (testnet, hideki or mainnet); with `DRY_RUN=false` the fly also bets for real on the windows, tUSDC test
money on the test networks and USDso on mainnet (see [live-trading.md](live-trading.md#event-contracts)).
`MARKETS=none` with `EC_ENABLED=true` runs event contracts on their own, with no spot markets. Perps run on the two
test networks: `PERP_MARKETS=all` (or a list) with `NETWORK=hideki` or `testnet`, in the same service as spot and
event contracts or in one of their own. Paper in a dry run; with `DRY_RUN=false` the perp order sidecar trades for
real (see [live-trading.md](live-trading.md#perps) for `OWNER_ADDRESS` and the margin). One service never mixes
networks, and perps on mainnet (which has none) are refused at start.
The Node runtime is already in the image. A redeploy stops the old container with SIGTERM: the fly closes its
positions first, so give it up to two minutes before the platform's hard stop.

## Health and restarts

`railway.toml` checks `/` and restarts the worker on failure. Inside the container the supervisor restarts any
child that dies, and passes SIGTERM to all of them, so on a redeploy the runner cancels its quotes and saves the
weights before it exits.

## Run the image yourself

```bash
docker build -t dreamfly .
docker run --rm -p 8765:8765 -e MARKETS=WETH:USDso dreamfly
```
