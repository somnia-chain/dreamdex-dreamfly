# Getting started

DreamFly runs a small model of a fly's mushroom body. It looks at the market, picks one of a few hand-written
strategies for the next hour, and learns from the result. You can watch it on a dashboard, let it write down the
orders it would place (dry run), or let it place real spot orders.

You do not need to know the DreamDEX protocol. The protocol is documented at https://docs.dreamdex.io.

## What you need

- Python 3.11 or newer.
- Node 20 or newer, only if you want perps or event contracts (they need a recorder).
- For real orders: a funded wallet key, and the DreamDEX core client (installed by the `live` extra).

## Install

```bash
git clone https://github.com/somnia-chain/dreamdex-dreamfly && cd dreamdex-dreamfly
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

Add `,live` inside the brackets (`".[dev,live]"`) if you plan to place real orders.

The circuit data, the dashboard data, the tuned strategy parameters and a warm-started brain for each spot market
are bundled, so there is nothing else to download. If you want to rebuild the circuit from the source files,
`dreamfly extract` explains how (about 1 GB).

## Configure

```bash
cp .env.example .env
```

The defaults are safe: `NETWORK=testnet`, `DRY_RUN=true`. Set `MARKETS` to the spot markets you want, for example
`MARKETS=WETH:USDso`. See [configuration.md](configuration.md) for every setting.

## Check the setup

```bash
dreamfly doctor
```

This is read-only. It prints your settings, checks the bundled data, checks that DreamDEX answers and that your
markets exist on the network you chose.

## Run it

```bash
dreamfly start
```

Open http://localhost:8765. The dashboard shows only the markets you run: with `MARKETS=WETH:USDso` you see one
tab, one fly and one terminal.

What you see:

- The desk: the fly presses a button for each decision.
- The neural replay and brain panels: which Kenyon cells fired and which areas lit up.
- The pool to strategy panel: which strategy the fly picked.
- Zoom to monitor: only the market terminal, with the fly's paper equity and open quotes.

The first decision arrives within a minute. Episodes last 60 minutes by default.

## What next

- Read [live-trading.md](live-trading.md) before you turn `DRY_RUN` off.
- Deploy it as a worker with the dashboard: [railway.md](railway.md).
- How the pieces fit: [architecture.md](architecture.md).
