#!/usr/bin/env node
// Read-only event-contract (Up/Down binary market) book recorder, MAINNET. No signer, no writes.
// Tracks active BTC/ETH 5m/15m/1h binary markets. In the last --window-sec before a market's
// expiry it samples both outcome books every --sample-ms, then reads the venue's resolution
// ~30 s after expiry. Rows go to <out>/mainnet_ec_YYYYMMDD.jsonl (UTC daily rotation).
//
// Every book sample also carries the venue oracle's current price, the window's open price, the
// lead in bps, the realised volatility and the zero-drift probability of "close at or above open"
// (the arithmetic the late-taker arm uses). The oracle feed is the SDK's indexed price feed.
//
//   node ec_recorder.mjs [--window-sec 60] [--sample-ms 2500] [--duration 0] [--out data/tape]

import { SomniaMarkets, SOMNIA_MAINNET_ADDRESSES, SOMNIA_MAINNET_PRICE_FEED } from "@somnia-chain/markets-sdk";
import { defineChain } from "viem";
import { appendFileSync, mkdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const RPC = "https://api.infra.mainnet.somnia.network";
const WS = "wss://api.infra.mainnet.somnia.network/ws";
const INDEXER = "https://prd.smk.somnia.host/v1/graphql";
const CHAIN_ID = 5031;
const INTERVALS = new Set(["5m", "15m", "1h"]);
const ASSETS = new Set(["BTC", "ETH"]);
const REFRESH_MS = 10_000;
const RESOLVE_DELAY_MS = 30_000;
const RESOLVE_RETRY_MS = 15_000;
const RESOLVE_TRIES = 8;
const RECREATE_AFTER = 5; // consecutive failed refresh/sample cycles before rebuilding the client
const CALL_TIMEOUT_MS = 20_000;
const DEPTH = 3;
const ORACLE_ASSETS = ["BTC", "ETH"];
const TICK_KEEP_S = 2 * 3600;
const VOL_WINDOW_S = 3600;
const ORACLE_HEARTBEAT_MS = 60_000;

// Pure: realised volatility from irregular samples [{t: seconds, price}] as bps per sqrt(second)
// over the last `windowS` seconds; null when there is not enough to say.
export function realisedVol(samples, windowS = VOL_WINDOW_S, minSamples = 12) {
  if (!Array.isArray(samples) || samples.length < minSamples) return null;
  const tEnd = samples[samples.length - 1].t;
  let sumR2 = 0, sumDt = 0, n = 0;
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    if (b.t < tEnd - windowS) continue;
    const dt = b.t - a.t;
    if (!(dt > 0) || !(a.price > 0) || !(b.price > 0)) continue;
    const r = Math.log(b.price / a.price);
    sumR2 += r * r; sumDt += dt; n++;
  }
  if (n < minSamples - 1 || !(sumDt > 0)) return null;
  return Math.sqrt(sumR2 / sumDt) * 1e4;
}

// Pure: standard normal CDF (Abramowitz and Stegun 7.1.26).
export function normalCdf(z) {
  if (!Number.isFinite(z)) return z > 0 ? 1 : 0;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989423 * Math.exp(-z * z / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return z >= 0 ? 1 - p : p;
}

// Pure: zero-drift P(close >= open) from the lead so far (bps), sigma (bps per sqrt(second)) and
// the seconds left.
export function baselineProb(leadBps, sigma, secondsLeft) {
  if (sigma == null || !(sigma > 0) || !Number.isFinite(leadBps)) return { prob: null, z: null };
  const z = leadBps / (sigma * Math.sqrt(Math.max(secondsLeft, 1)));
  return { prob: normalCdf(z), z };
}

// Pure: the venue stores the open as an integer with an unknown number of decimals; pick the power
// of ten that lands within 20% of the live price.
export function scaleToSpot(raw, spot) {
  if (!(raw > 0) || !(spot > 0)) return null;
  for (let p = 0; p <= 12; p++) {
    const v = raw / 10 ** p;
    if (Math.abs(v - spot) / spot < 0.2) return v;
  }
  return null;
}

function parseArgs(argv) {
  const here = dirname(fileURLToPath(import.meta.url));
  const o = { windowSec: 60, sampleMs: 2500, duration: 0, out: resolve(here, "..", "data", "tape") };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => argv[++i];
    if (a === "--window-sec") o.windowSec = Number(val());
    else if (a === "--sample-ms") o.sampleMs = Number(val());
    else if (a === "--duration") o.duration = Number(val());
    else if (a === "--out") o.out = resolve(val());
    else if (a === "-h" || a === "--help") {
      console.log("usage: node ec_recorder.mjs [--window-sec 60] [--sample-ms 2500] [--duration 0] [--out data/tape]");
      process.exit(0);
    } else throw new Error(`unknown flag ${a}`);
  }
  if (!Number.isFinite(o.windowSec) || o.windowSec <= 0) throw new Error("--window-sec must be > 0");
  if (!Number.isFinite(o.sampleMs) || o.sampleMs < 500) throw new Error("--sample-ms must be >= 500");
  if (!Number.isFinite(o.duration) || o.duration < 0) throw new Error("--duration must be >= 0");
  return o;
}

function makeExchange() {
  const chain = defineChain({
    id: CHAIN_ID,
    name: "somnia",
    nativeCurrency: { name: "Somnia", symbol: "SOMI", decimals: 18 },
    rpcUrls: { default: { http: [RPC], webSocket: [WS] } },
  });
  // Read-only: no privateKey. wsRpcUrl is required even for reads.
  return new SomniaMarkets({ indexerUrl: INDEXER, chain, wsRpcUrl: WS, addresses: SOMNIA_MAINNET_ADDRESSES,
    priceFeed: SOMNIA_MAINNET_PRICE_FEED });
}

// Oracle price ticks per asset from the SDK's live price store, kept for TICK_KEEP_S seconds.
class OracleFeed {
  constructor() {
    this.ticks = Object.fromEntries(ORACLE_ASSETS.map((a) => [a, []]));
    this.handle = null;
    this.unsub = null;
    this.lastPushAt = 0;
  }
  async attach(ex) {
    this.detach();
    this.handle = await withTimeout(ex.client.watchPrices(ORACLE_ASSETS), CALL_TIMEOUT_MS, "watchPrices");
    this.unsub = ex.client.subscribePrices(() => this.pull(ex));
    this.pull(ex);
  }
  detach() {
    try { this.unsub?.(); } catch { /* ignore */ }
    try { this.handle?.stop?.(); } catch { /* ignore */ }
    this.handle = this.unsub = null;
  }
  pull(ex) {
    for (const a of ORACLE_ASSETS) {
      let p = null;
      try { p = ex.client.getLivePrice(a); } catch { continue; }
      if (!p || !(p.price > 0)) continue;
      const buf = this.ticks[a];
      const last = buf[buf.length - 1];
      if (last && last.t >= p.blockTimestamp) continue;
      buf.push({ t: p.blockTimestamp, price: p.price });
      this.lastPushAt = Date.now();
      while (buf.length && p.blockTimestamp - buf[0].t > TICK_KEEP_S) buf.shift();
    }
  }
  last(asset) {
    const buf = this.ticks[asset] ?? [];
    return buf.length ? buf[buf.length - 1] : null;
  }
  sigma(asset) {
    return realisedVol(this.ticks[asset] ?? []);
  }
  // What a book sample needs: oracle price, lead vs the open, sigma, z, model probability of Up.
  view(asset, open, secondsLeft) {
    const tick = this.last(asset);
    const sigma = this.sigma(asset);
    const out = { oracle: tick?.price ?? null, oracle_t: tick?.t ?? null, open: open ?? null,
      lead_bps: null, sigma_bps: sigma, z: null, prob_up: null };
    if (tick && open > 0) {
      out.lead_bps = (tick.price / open - 1) * 1e4;
      const b = baselineProb(out.lead_bps, sigma, secondsLeft);
      out.z = b.z;
      out.prob_up = b.prob;
    }
    return out;
  }
}

const withTimeout = (p, ms, label) =>
  Promise.race([
    Promise.resolve(p),
    new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timed out after ${ms}ms`)), ms)),
  ]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const errMsg = (e) => String(e?.shortMessage ?? e?.message ?? e).slice(0, 300);
const wallIso = () => new Date().toISOString();

// Pure: turn the venue's getMarketResolution() object into { found, truth, voided, open, close }.
// truth: true = Up (YES) won, false = Down (NO) won, null = voided / not found.
export function readResolution(res) {
  const out = { found: false, truth: null, voided: false, open: null, close: null };
  if (!res || typeof res !== "object") return out;
  const num = (a) => {
    const v = a?.numericValue;
    if (v == null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  out.open = num(res.openingAnswer);
  out.close = num(res.closingAnswer ?? res.oracleAnswer);
  const events = Array.isArray(res.events) ? res.events : [];
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (!e) continue;
    if (e.voided === true) return { ...out, found: true, voided: true };
    if (e.winningOutcome === 0 || e.winningOutcome === 1) return { ...out, found: true, truth: e.winningOutcome === 0 };
    const nums = Array.isArray(e.payoutNumerators) ? e.payoutNumerators.map(Number) : null;
    if (nums && nums.length === 2 && nums.some((n) => n > 0)) {
      const total = nums[0] + nums[1];
      if (total > 0 && Math.max(...nums) === total) return { ...out, found: true, truth: nums[0] > nums[1] };
      if (nums[0] === nums[1]) return { ...out, found: true, voided: true };
    }
  }
  return out;
}

function dayFile(out) {
  const d = new Date().toISOString().slice(0, 10).replaceAll("-", "");
  return join(out, `mainnet_ec_${d}.jsonl`);
}

function write(out, row) {
  try {
    mkdirSync(out, { recursive: true });
    appendFileSync(dayFile(out), JSON.stringify(row) + "\n");
  } catch (e) {
    console.error(`[ec_recorder] write failed: ${errMsg(e)}`);
  }
}

const side = (levels) => (levels ?? []).slice(0, DEPTH).map(([p, q]) => [p, q]);

async function main() {
  const opt = parseArgs(process.argv.slice(2));
  let stopped = false;
  for (const s of ["SIGINT", "SIGTERM"]) process.on(s, () => { stopped = true; });

  let ex = makeExchange();
  let failStreak = 0;
  const tracked = new Map(); // marketId -> market record
  const deadline = opt.duration > 0 ? Date.now() + opt.duration * 1000 : Infinity;
  let lastRefresh = 0;
  let refreshing = false;
  let lastHeartbeat = 0;
  let counts = { open: 0, book: 0, res: 0 };
  const oracle = new OracleFeed();
  console.log(`[ec_recorder] start window=${opt.windowSec}s sample=${opt.sampleMs}ms duration=${opt.duration}s out=${opt.out}`);

  const attachOracle = async () => {
    try { await oracle.attach(ex); }
    catch (e) { write(opt.out, { error: `oracle: ${errMsg(e)}`, wall: wallIso() }); console.error(`[ec_recorder] oracle: ${errMsg(e)}`); }
  };

  const noteFail = async (ok) => {
    failStreak = ok ? 0 : failStreak + 1;
    if (failStreak >= RECREATE_AFTER) {
      console.error(`[ec_recorder] ${failStreak} failed cycles, recreating client`);
      oracle.detach();
      try { await withTimeout(ex.close?.(), 3000, "close"); } catch { /* ignore */ }
      try { ex = makeExchange(); } catch (e) { console.error(`[ec_recorder] recreate failed: ${errMsg(e)}`); }
      await attachOracle();
      failStreak = 0;
    }
  };

  // The window's open price from the venue's resolution record, scaled to the live oracle price.
  async function readOpen(t) {
    const spot = oracle.last(t.asset)?.price;
    if (!(spot > 0)) return;
    try {
      const r = readResolution(await withTimeout(ex.client.getMarketResolution(t.marketId), CALL_TIMEOUT_MS, "open"));
      t.open = scaleToSpot(r.open, spot);
      if (t.open != null) write(opt.out, { type: "ec_openprice", marketId: t.marketId, symbol: t.symbol, open: t.open, wall: wallIso() });
    } catch (e) {
      write(opt.out, { error: `open ${t.marketId}: ${errMsg(e)}`, wall: wallIso() });
    }
  }

  function heartbeat() {
    for (const a of ORACLE_ASSETS) {
      const tick = oracle.last(a);
      write(opt.out, { type: "ec_oracle", asset: a, price: tick?.price ?? null, t: tick?.t ?? null,
        sigma_bps: oracle.sigma(a), ticks: oracle.ticks[a].length, silent_ms: Date.now() - oracle.lastPushAt, wall: wallIso() });
    }
  }

  async function refresh() {
    refreshing = true;
    try {
      await withTimeout(ex.loadMarkets(true), CALL_TIMEOUT_MS * 2, "loadMarkets");
      for (const m of Object.values(ex.markets ?? {})) {
        if (m.type !== "binary" || !m.active) continue;
        const i = m.info ?? {};
        if (!INTERVALS.has(i.interval) || !ASSETS.has(i.asset)) continue;
        const marketId = i.marketId ?? m.id;
        if (!marketId || tracked.has(marketId)) continue;
        const up = m.outcomes?.find((o) => o.label === "YES")?.symbol;
        const down = m.outcomes?.find((o) => o.label === "NO")?.symbol;
        if (!up || !down) continue;
        const t = {
          marketId, symbol: m.symbol, asset: i.asset, interval: i.interval, expiry: Number(i.expiry),
          openTs: Number(i.tradingStart ?? i.openTs ?? i.startTime ?? 0) || null,
          up, down, sampling: false, resolveAt: null, resolving: false, open: null, openTries: 0,
        };
        tracked.set(marketId, t);
        counts.open++;
        write(opt.out, { type: "ec_open", marketId, symbol: t.symbol, asset: t.asset, interval: t.interval,
          openTs: t.openTs, expiry: t.expiry, wall: wallIso() });
      }
      for (const t of tracked.values()) {
        if (t.open == null && t.openTries < 20 && t.expiry > Date.now() / 1000) {
          t.openTries++;
          await readOpen(t);
        }
      }
      await noteFail(true);
    } catch (e) {
      write(opt.out, { error: `refresh: ${errMsg(e)}`, wall: wallIso() });
      console.error(`[ec_recorder] refresh error: ${errMsg(e)}`);
      await noteFail(false);
    } finally {
      refreshing = false;
    }
  }

  async function sample(t) {
    t.sampling = true;
    const t0 = Date.now();
    try {
      const [u, d] = await Promise.all([
        withTimeout(ex.fetchOrderBook(t.up, DEPTH), CALL_TIMEOUT_MS, "book up"),
        withTimeout(ex.fetchOrderBook(t.down, DEPTH), CALL_TIMEOUT_MS, "book down"),
      ]);
      const secondsLeft = +(t.expiry - t0 / 1000).toFixed(1);
      write(opt.out, {
        type: "ec_book", marketId: t.marketId, symbol: t.symbol, asset: t.asset, interval: t.interval,
        expiry: t.expiry, secondsLeft,
        up: { bids: side(u?.bids), asks: side(u?.asks) },
        down: { bids: side(d?.bids), asks: side(d?.asks) },
        ...oracle.view(t.asset, t.open, secondsLeft),
        wall: new Date(t0).toISOString(),
      });
      counts.book++;
      await noteFail(true);
    } catch (e) {
      write(opt.out, { error: `book ${t.marketId}: ${errMsg(e)}`, wall: wallIso() });
      await noteFail(false);
    } finally {
      t.sampling = false;
    }
  }

  async function resolveMarket(t) {
    t.resolving = true;
    for (let n = 0; n < RESOLVE_TRIES && !stopped; n++) {
      try {
        const r = readResolution(await withTimeout(ex.client.getMarketResolution(t.marketId), CALL_TIMEOUT_MS, "resolution"));
        if (r.found) {
          write(opt.out, { type: "ec_resolution", marketId: t.marketId, symbol: t.symbol,
            truth: r.voided ? null : (r.truth ? "up" : "down"), voided: r.voided,
            open: r.open, close: r.close, wall: wallIso() });
          counts.res++;
          tracked.delete(t.marketId);
          return;
        }
      } catch (e) {
        write(opt.out, { error: `resolution ${t.marketId}: ${errMsg(e)}`, wall: wallIso() });
      }
      await sleep(RESOLVE_RETRY_MS);
    }
    write(opt.out, { type: "ec_resolution", marketId: t.marketId, symbol: t.symbol, truth: null, timeout: true, wall: wallIso() });
    tracked.delete(t.marketId);
  }

  await attachOracle();
  while (!stopped && Date.now() < deadline) {
    const tick = Date.now();
    if (!refreshing && tick - lastRefresh >= REFRESH_MS) {
      lastRefresh = tick;
      refresh(); // background; never throws
    }
    if (tick - lastHeartbeat >= ORACLE_HEARTBEAT_MS) {
      lastHeartbeat = tick;
      oracle.pull(ex);
      heartbeat();
      if (oracle.lastPushAt && tick - oracle.lastPushAt > 3 * ORACLE_HEARTBEAT_MS) {
        console.error("[ec_recorder] oracle feed silent, re-attaching");
        await attachOracle();
      }
    }
    for (const t of tracked.values()) {
      const left = t.expiry - tick / 1000;
      if (left <= opt.windowSec && left > 0 && !t.sampling) sample(t);
      else if (left <= 0 && !t.resolving && tick / 1000 >= t.expiry + RESOLVE_DELAY_MS / 1000) resolveMarket(t);
    }
    const next = Math.min(tick + opt.sampleMs, deadline);
    while (!stopped && Date.now() < next) await sleep(Math.min(250, next - Date.now()));
  }

  console.log(`[ec_recorder] stopped opens=${counts.open} books=${counts.book} resolutions=${counts.res} pending=${tracked.size}`);
  oracle.detach();
  try { await withTimeout(ex.close?.(), 3000, "close"); } catch { /* ignore */ }
  process.exit(0);
}

main().catch((e) => {
  console.error(`[ec_recorder] fatal: ${errMsg(e)}`);
  process.exit(1);
});
