#!/usr/bin/env node
// Read-only perp market recorder. No signer, no writes.
// Every --interval-ms, polls all perp markets and appends one JSON line per
// market to <out>/<network>_perp_YYYYMMDD.jsonl (UTC daily rotation).
//
//   node recorder.mjs [--network hideki|testnet] [--interval-ms 60000] [--duration 0] [--out data/tape]

import { SomniaMarkets, SOMNIA_TESTNET_ADDRESSES, annualizedFundingRate } from "@somnia-chain/markets-sdk";
import { defineChain } from "viem";
import { appendFileSync, mkdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// Two perp test networks. The bundled testnet address set serves both: the perp pool and margin bank come off
// each market row, not from the address set.
const NETWORKS = {
  testnet: {
    rpc: "https://api.infra.testnet.somnia.network",
    ws: "wss://api.infra.testnet.somnia.network/ws",
    indexer: "https://dev.smk.somnia.host/v1/graphql",
    chainId: 50312,
    prefix: "testnet_perp_",
  },
  hideki: {
    rpc: "https://api.hideki2.infra.testnet.somnia.network",
    ws: "wss://api.hideki2.infra.testnet.somnia.network/ws",
    indexer: "https://hideki-dev.smk.somnia.host/v1/graphql",
    chainId: 50383,
    prefix: "hideki_perp_",
  },
};
let NET = NETWORKS.testnet;
const RECREATE_AFTER = 3; // consecutive failed cycles before rebuilding the exchange
const CALL_TIMEOUT_MS = 20_000;

function parseArgs(argv) {
  const here = dirname(fileURLToPath(import.meta.url));
  const o = { intervalMs: 60_000, duration: 0, out: resolve(here, "..", "data", "tape"), network: "hideki" };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === "--interval-ms") o.intervalMs = Number(val());
    else if (a === "--duration") o.duration = Number(val());
    else if (a === "--out") o.out = resolve(val());
    else if (a === "--network") o.network = val();
    else if (a === "-h" || a === "--help") {
      console.log("usage: node recorder.mjs [--network hideki|testnet] [--interval-ms 60000] [--duration 0] [--out data/tape]");
      process.exit(0);
    } else throw new Error(`unknown flag ${a}`);
  }
  if (!NETWORKS[o.network]) throw new Error(`--network must be one of ${Object.keys(NETWORKS).join(", ")}`);
  if (!Number.isFinite(o.intervalMs) || o.intervalMs < 1000) throw new Error("--interval-ms must be >= 1000");
  if (!Number.isFinite(o.duration) || o.duration < 0) throw new Error("--duration must be >= 0");
  return o;
}

function makeExchange() {
  const chain = defineChain({
    id: NET.chainId,
    name: `somnia-${NET.chainId}`,
    nativeCurrency: { name: "Somnia Test Token", symbol: "STT", decimals: 18 },
    rpcUrls: { default: { http: [NET.rpc], webSocket: [NET.ws] } },
  });
  // Read-only: no privateKey. wsRpcUrl is required even for reads.
  return new SomniaMarkets({
    indexerUrl: NET.indexer,
    chain,
    wsRpcUrl: NET.ws,
    addresses: SOMNIA_TESTNET_ADDRESSES,
  });
}

const withTimeout = (p, ms, label) =>
  Promise.race([
    Promise.resolve(p),
    new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timed out after ${ms}ms`)), ms)),
  ]);

const errMsg = (e) => String(e?.shortMessage ?? e?.message ?? e).slice(0, 300);
const wallIso = () => new Date().toISOString();
const num = (raw, dec) => (raw === undefined || raw === null ? null : Number(BigInt(raw)) / 10 ** dec);
const display = (symbol) => `${symbol.split("/")[0]}-PERP`;

function dayFile(out) {
  const d = new Date().toISOString().slice(0, 10).replaceAll("-", "");
  return join(out, `${NET.prefix}${d}.jsonl`);
}

function write(out, rows) {
  if (!rows.length) return;
  try {
    mkdirSync(out, { recursive: true });
    appendFileSync(dayFile(out), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  } catch (e) {
    console.error(`[recorder] write failed: ${errMsg(e)}`);
  }
}

// Live chain mark; null while the pool reports the mark as stale.
async function liveMark(ex, poolAddress) {
  const snap = await withTimeout(ex.client.getPerpHealthSnapshot(poolAddress), CALL_TIMEOUT_MS, "health snapshot");
  if (!snap.priceable) return null;
  const raw = Number(snap.markPrice);
  return raw > 0 ? raw / 1e18 : null;
}

async function pollMarket(ex, m) {
  const info = m.info;
  const wall = wallIso();
  const row = {
    type: "perp",
    symbol: m.symbol,
    display: display(m.symbol),
    wall,
    active: Boolean(m.active),
    mark: null,
    row_mark: num(info.markPrice, 18),
    index: num(info.indexPrice, 18),
    funding_8h: null,
    funding_apr: null,
    oi: num(info.openInterest, Number(info.baseDecimals)),
    best_bid: null,
    best_ask: null,
    funding_ts: info.fundingUpdatedAt ? Number(info.fundingUpdatedAt) : null,
  };
  const problems = [];

  try {
    const rate = BigInt(info.fundingRate ?? 0);
    const windowSec = Number(info.fundingWindowSec);
    if (windowSec > 0) {
      row.funding_8h = (Number(rate) / 1e18) * (28800 / windowSec);
      row.funding_apr = annualizedFundingRate(rate, windowSec);
    }
  } catch (e) {
    problems.push(`funding: ${errMsg(e)}`);
  }
  try {
    row.mark = await liveMark(ex, info.poolAddress);
  } catch (e) {
    problems.push(`mark: ${errMsg(e)}`);
  }
  try {
    const book = await withTimeout(ex.fetchOrderBook(m.symbol, 1), CALL_TIMEOUT_MS, "order book");
    row.best_bid = book.bids?.[0]?.[0] ?? null;
    row.best_ask = book.asks?.[0]?.[0] ?? null;
  } catch (e) {
    problems.push(`book: ${errMsg(e)}`);
  }
  if (problems.length) row.partial_errors = problems;
  return { row, failed: problems.length >= 2 };
}

async function main() {
  const opt = parseArgs(process.argv.slice(2));
  NET = NETWORKS[opt.network];
  let stopped = false;
  for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => { stopped = true; });

  let ex = makeExchange();
  let failStreak = 0;
  const started = Date.now();
  const deadline = opt.duration > 0 ? started + opt.duration * 1000 : Infinity;
  console.log(`[recorder] start network=${opt.network} interval=${opt.intervalMs}ms duration=${opt.duration}s out=${opt.out}`);

  while (!stopped && Date.now() < deadline) {
    const t0 = Date.now();
    let cycleFailed = false;
    try {
      const all = await withTimeout(ex.loadMarkets(true), CALL_TIMEOUT_MS * 2, "loadMarkets");
      const perps = Object.values(all).filter((m) => m.type === "swap");
      if (perps.length === 0) throw new Error("no perp markets returned");
      const rows = [];
      let bad = 0;
      for (const m of perps) {
        if (stopped) break;
        try {
          const r = await pollMarket(ex, m);
          rows.push(r.row);
          if (r.failed) bad++;
        } catch (e) {
          bad++;
          rows.push({ error: errMsg(e), symbol: m.symbol, wall: wallIso() });
        }
      }
      write(opt.out, rows);
      cycleFailed = bad === perps.length;
      console.log(`[recorder] ${wallIso()} wrote ${rows.length} rows (${bad} failed)`);
    } catch (e) {
      cycleFailed = true;
      write(opt.out, [{ error: errMsg(e), wall: wallIso() }]);
      console.error(`[recorder] cycle error: ${errMsg(e)}`);
    }

    failStreak = cycleFailed ? failStreak + 1 : 0;
    if (failStreak >= RECREATE_AFTER) {
      console.error(`[recorder] ${failStreak} failed cycles, recreating exchange`);
      try { await withTimeout(ex.close?.(), 3000, "close"); } catch { /* ignore */ }
      try { ex = makeExchange(); } catch (e) { console.error(`[recorder] recreate failed: ${errMsg(e)}`); }
      failStreak = 0;
    }

    // Interruptible sleep until the next tick (or the deadline).
    const next = Math.min(t0 + opt.intervalMs, deadline);
    while (!stopped && Date.now() < next) await new Promise((r) => setTimeout(r, Math.min(500, next - Date.now())));
  }

  try { await withTimeout(ex.close?.(), 3000, "close"); } catch { /* ignore */ }
  console.log("[recorder] stopped");
  process.exit(0);
}

main().catch((e) => {
  console.error(`[recorder] fatal: ${errMsg(e)}`);
  process.exit(1);
});
