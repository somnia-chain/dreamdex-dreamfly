# Contributing

Thanks for helping improve DreamFly. This repo is teaching and research material for DreamDEX builders, so the bar
is accuracy and clarity over feature count: a contribution should make the model easier to understand or run
correctly, and anything that touches orders must match the live contract.

## Repository layout

| Path | What it is |
| --- | --- |
| `flybot/mb/` | The mushroom body model and the connectome subgraph loader. |
| `flybot/` | Market features, strategies, selectors, the backtest loop, perp and event contract logic. |
| `flybot/live/` | Recorders, paper execution, the live order mirror, per-instrument flies, the runner. |
| `perp/` | Node recorders for perps and event contracts. |
| `site/` | The dashboard. |
| `scripts/` | Data extraction, tuning and backtest scripts, warm start, container supervisor. |
| `docs/` | Getting started, configuration, live trading, architecture, Railway. |
| `tests/` | Unit tests. |

## Dev setup and checks

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

Before opening a PR:

- `pytest` must pass.
- Match the surrounding code. New user-facing text is English only.
- No em dashes or en dashes anywhere: use a colon, comma or full stop.
- Keep the dashboard working at phone, tablet and desktop widths. Phones show the market terminal only.

## What belongs here

- Improvements to the model, the strategies, the safety checks, the dashboard and the docs.
- Fixes that keep the live path in sync with the DreamDEX contracts and API.

**Not accepted:** volume generation or wash-trade tooling, and anything whose only purpose is to inflate activity.

## Safety rules for order code

- The default must stay `DRY_RUN=true` on `NETWORK=testnet`.
- Live order code must go through `dreamdex_core` and its guards. Do not bypass them.
- A change that touches live orders needs a note saying whether you verified it on testnet, on mainnet, or offline only.
- Never commit secrets, a funded key, or an `.env` file.

## Pull requests

- Keep PRs focused, and describe what changed and why it is correct.
- One logical change per PR where practical. Use a feature branch and merge with `--no-ff`.

## License and legal

This project is MIT-licensed, Copyright (c) 2026 DreamDEX S.A. By contributing, you agree your contributions are
licensed under the same terms. Nothing in this repo is financial advice or a guarantee of results. See
[`DISCLAIMER.md`](DISCLAIMER.md). Credits for the bundled data are in [`NOTICE.md`](NOTICE.md).
