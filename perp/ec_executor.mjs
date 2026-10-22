// Event contract order sidecar: the one process that signs event contract (binary Up/Down) orders for DreamFly.
//
// The runner (Python) starts it and talks JSON lines over stdin and stdout: one request per line, one reply
// per line, with the same `id`, served one at a time so the signer's nonces stay in order. It uses the
// DreamDEX markets SDK the bot kit's `ec-*` strategies use, and follows the kit's ec-core where the venue
// has sharp edges: prices and sizes go to the pool as exact integers on the tick and lot grid (a float price
// on the 18 decimal mainnet venue lands a few wei off the grid and is rejected), a reverted write does not
// throw so the receipt is checked, an order always carries an expiry no later than the window's, winnings are
// claimed (a settled market pays only when asked), and a voided market pays both sides 0.5.
//
//   node ec_executor.mjs --network mainnet [--dry-run]
//
// Event contract windows exist on mainnet; that is the only network this sidecar accepts. PRIVATE_KEY comes
// from the environment. The fly only ever takes (one immediate-or-cancel buy per window); nothing rests.
//
// Requests ({"id": 1, "op": "...", ...}) and what they return (always with "id" and "ok"):
//   init                      {address, chain, collateral, decimals, markets}
//   list                      {windows: [{market_id, symbol, asset, interval, expiry, seconds_left}]} the live windows
//   book {market_id}          {up: {bids, asks}, down: {bids, asks}} best levels, each leg in its own probability
//   market {market_id}        {status, tradable, expiry, seconds_left, symbol}
//   buy {market_id, side, cap, notional, ttl_s}
//                             side up|down, cap = the most to pay per share in that side's own probability,
//                             notional = USDso to spend; {sent, filled, cost, avg_price, tx, reason, no_fill}
//   position {market_id}      {up, down} shares held
//   claim {market_id}         {claimed, redeemed: [{side, amount}], tx, reason}
//   sweep                     redeems everything the venue lists as claimable for this address: {claimed, markets, est_payout}
//   wallet                    {address, somi, usdso}
//   ping, shutdown
//
// Numbers in and out are human units (probabilities in 0..1, shares, USDso). Nothing here reads a .env file.
import { createInterface } from "node:readline";
import { ORDER_TYPE, SomniaMarkets, SOMNIA_MAINNET_ADDRESSES } from "@somnia-chain/markets-sdk";
import { createPublicClient, defineChain, http, parseAbi } from "viem";

const NETWORKS = {
  mainnet: {
    rpc: "https://api.infra.mainnet.somnia.network",
    ws: "wss://api.infra.mainnet.somnia.network/ws",
    indexer: "https://prd.smk.somnia.host/v1/graphql",
    chainId: 5031,
    addresses: SOMNIA_MAINNET_ADDRESSES,
    native: "SOMI",
  },
};

const CALL_TIMEOUT_MS = 60_000;
const DEFAULT_GRID = 10n ** 15n;        // the mainnet venue's tick and lot (0.001) when the pool does not say
const STATUS_TRADING = 1;               // on-chain MarketStatus: 0 Listed, 1 Trading, 2 Locked, 3 Settling, 4 Resolved, 5 Voided
const ERC20 = parseAbi(["function balanceOf(address) view returns (uint256)"]);

function parseArgs(argv) {
  const o = { network: "mainnet", dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--network") o.network = argv[++i];
    else if (a === "--dry-run") o.dryRun = true;
    else throw new Error(`unknown flag ${a}`);
  }
  if (!NETWORKS[o.network]) throw new Error(`--network ${o.network}: event contract windows exist on mainnet only (--network mainnet, NETWORK=mainnet)`);
  return o;
}

const withTimeout = (p, ms, label) =>
  Promise.race([
    Promise.resolve(p),
    new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timed out after ${ms} ms`)), ms)),
  ]);

const errMsg = (e) => {
  const name = e?.cause?.data?.errorName ?? e?.data?.errorName ?? e?.errorName;
  const msg = String(e?.shortMessage ?? e?.message ?? e).slice(0, 400);
  return name && !msg.includes(name) ? `${name}: ${msg}` : msg;
};

// Human <-> raw. Prices and sizes are built in grid steps (small integers) so a float never reaches the chain
// a few wei off the grid.
const fromRaw = (raw, decimals) => Number(BigInt(raw)) / 10 ** decimals;
function toSteps(human, one, step, mode) {
  const stepsPerOne = Number(one / step);
  const n = Number(human) * stepsPerOne;
  const steps = mode === "round" ? Math.round(n) : Math.floor(n + 1e-9);
  return BigInt(Math.max(0, steps)) * step;
}

class Executor {
  constructor(opts) {
    this.opts = opts;
    this.net = NETWORKS[opts.network];
    this.ex = null;
    this.me = null;
    this.collateral = null;
    this.decimals = 18;
    this.grid = { tick: DEFAULT_GRID, lot: DEFAULT_GRID };
    this.markets = new Map();     // marketId -> {marketId, symbol, asset, interval, expiry, up, down}
    const raw = (process.env.PRIVATE_KEY ?? "").trim();
    this.key = raw && !raw.startsWith("0x") ? `0x${raw}` : raw;
    this.pc = createPublicClient({ chain: this.chain(), transport: http(this.net.rpc) });
  }

  chain() {
    return defineChain({
      id: this.net.chainId,
      name: `somnia-${this.net.chainId}`,
      nativeCurrency: { name: this.net.native, symbol: this.net.native, decimals: 18 },
      rpcUrls: { default: { http: [this.net.rpc], webSocket: [this.net.ws] } },
    });
  }

  async init() {
    if (this.ex) await this.closeExchange();
    this.ex = new SomniaMarkets({
      indexerUrl: this.net.indexer, chain: this.chain(), wsRpcUrl: this.net.ws, addresses: this.net.addresses,
      privateKey: this.opts.dryRun ? undefined : this.key || undefined,
    });
    this.me = this.ex.walletAddress ?? null;
    const n = await this.loadMarkets();
    // the collateral token and its decimals come from a live window's on-chain record
    const first = this.markets.values().next().value;
    if (first) {
      const oc = await this.onchain(first.marketId);
      this.collateral = oc.collateral;
      this.decimals = Number(oc.decimals ?? 18);
      await this.learnGrid(oc.pool);
    }
    return { address: this.me, chain: this.net.chainId, collateral: this.collateral, decimals: this.decimals,
             markets: n, tick: String(this.grid.tick), lot: String(this.grid.lot), dry_run: this.opts.dryRun };
  }

  async learnGrid(pool) {
    // the SDK exposes the pool's book grid on newer deployments; the mainnet default is kept otherwise
    try {
      const p = await withTimeout(this.ex.client.getBinaryBookParams?.(pool), CALL_TIMEOUT_MS, "book params");
      const tick = p?.tickSize ?? p?.tick, lot = p?.lotSize ?? p?.lot;
      if (tick && lot && BigInt(tick) > 0n && BigInt(lot) > 0n) this.grid = { tick: BigInt(tick), lot: BigInt(lot) };
    } catch { /* keep the default */ }
  }

  async loadMarkets() {
    const all = await withTimeout(this.ex.loadMarkets(true), CALL_TIMEOUT_MS * 2, "loadMarkets");
    this.markets.clear();
    for (const m of Object.values(all ?? {})) {
      if (m.type !== "binary" || !m.active) continue;
      const i = m.info ?? {};
      const marketId = i.marketId ?? m.id;
      if (!marketId) continue;
      const up = m.outcomes?.find((o) => o.label === "YES")?.symbol;
      const down = m.outcomes?.find((o) => o.label === "NO")?.symbol;
      this.markets.set(String(marketId).toLowerCase(), {
        marketId, symbol: m.symbol, asset: i.asset ?? null, interval: i.interval ?? null, expiry: Number(i.expiry ?? 0), up, down,
      });
    }
    return this.markets.size;
  }

  async known(marketId) {
    const k = String(marketId).toLowerCase();
    if (!this.markets.has(k)) await this.loadMarkets();
    return this.markets.get(k) ?? { marketId, symbol: null, up: null, down: null, expiry: 0 };
  }

  onchain(marketId) {
    return withTimeout(this.ex.client.getMarketOnchain(marketId), CALL_TIMEOUT_MS, "market onchain");
  }

  requireSigner() {
    if (this.opts.dryRun) return null;
    if (!this.me) throw new Error("no signer: PRIVATE_KEY is not set");
    return this.me;
  }

  async list() {
    await this.loadMarkets();
    const now = Date.now() / 1000;
    return { windows: [...this.markets.values()].map((m) => ({
      market_id: m.marketId, symbol: m.symbol, asset: m.asset, interval: m.interval, expiry: m.expiry,
      seconds_left: m.expiry ? +(m.expiry - now).toFixed(1) : null,
    })).sort((a, b) => (a.expiry ?? 0) - (b.expiry ?? 0)) };
  }

  async book(marketId, depth = 3) {
    // best levels of both legs, each in its own probability, as the recorder writes them
    const m = await this.known(marketId);
    if (!m.up || !m.down) throw new Error(`${marketId} is not a live window here`);
    const [u, d] = await Promise.all([
      withTimeout(this.ex.fetchOrderBook(m.up, depth), CALL_TIMEOUT_MS, "book up"),
      withTimeout(this.ex.fetchOrderBook(m.down, depth), CALL_TIMEOUT_MS, "book down"),
    ]);
    const side = (levels) => (levels ?? []).slice(0, depth).map(([p, q]) => [Number(p), Number(q)]);
    return { symbol: m.symbol, up: { bids: side(u?.bids), asks: side(u?.asks) }, down: { bids: side(d?.bids), asks: side(d?.asks) } };
  }

  async market(marketId) {
    const m = await this.known(marketId);
    const oc = await this.onchain(marketId);
    const expiry = Number(oc.expiry ?? m.expiry ?? 0);
    return { symbol: m.symbol, status: Number(oc.status), tradable: Number(oc.status) === STATUS_TRADING, expiry,
             seconds_left: expiry ? +(expiry - Date.now() / 1000).toFixed(1) : null,
             resolved: Boolean(oc.isResolved), voided: Boolean(oc.isVoided) };
  }

  async balances(oc, who) {
    const [yes, no] = await Promise.all([
      this.ex.client.getOutcomeBalance({ outcomeToken: oc.outcomeToken, account: who, id: oc.yesId }),
      this.ex.client.getOutcomeBalance({ outcomeToken: oc.outcomeToken, account: who, id: oc.noId }),
    ]);
    return { yes, no };
  }

  async position(marketId) {
    const who = this.requireSigner();
    if (!who) return { up: 0, down: 0 };
    const oc = await this.onchain(marketId);
    const b = await this.balances(oc, who);
    const d = Number(oc.decimals ?? this.decimals);
    return { up: fromRaw(b.yes, d), down: fromRaw(b.no, d) };
  }

  async wallet() {
    const who = this.me;
    if (!who) throw new Error("no signer: PRIVATE_KEY is not set");
    const somi = await this.pc.getBalance({ address: who });
    let usdso = null;
    if (this.collateral) {
      const raw = await this.pc.readContract({ address: this.collateral, abi: ERC20, functionName: "balanceOf", args: [who] });
      usdso = fromRaw(raw, this.decimals);
    }
    return { address: who, somi: fromRaw(somi, 18), usdso };
  }

  // One immediate-or-cancel buy of `side` at up to `cap` (that side's own probability), spending up to `notional`.
  async buy({ market_id, side, cap, notional, ttl_s = 60 }) {
    if (!["up", "down"].includes(side)) throw new Error(`side must be up or down, not ${side}`);
    const m = await this.known(market_id);
    const oc = await this.onchain(market_id);
    if (Number(oc.status) !== STATUS_TRADING) return { sent: false, filled: 0, cost: 0, reason: `market status ${oc.status}, not Trading` };
    const d = Number(oc.decimals ?? this.decimals);
    const one = 10n ** BigInt(d);
    const priceOwn = toSteps(cap, one, this.grid.tick, "round");
    if (priceOwn <= 0n || priceOwn >= one) return { sent: false, filled: 0, cost: 0, reason: `cap ${cap} is outside (0, 1) on the tick grid` };
    const shares = Number(notional) / Number(cap);
    const quantity = toSteps(shares, one, this.grid.lot, "floor");
    if (quantity <= 0n) return { sent: false, filled: 0, cost: 0, reason: `${notional} USDso at ${cap} rounds below one lot` };
    const nowSec = Math.floor(Date.now() / 1000);
    const expiresAt = Math.min(nowSec + Math.max(5, Number(ttl_s) || 60), Number(oc.expiry));
    if (expiresAt <= nowSec) return { sent: false, filled: 0, cost: 0, reason: "the window has expired" };
    const priceYes = side === "up" ? priceOwn : one - priceOwn;       // the pool quotes in YES terms
    const need = (priceOwn * quantity) / one;
    if (this.opts.dryRun) {
      return { sent: false, dry_run: true, filled: 0, cost: 0, qty: fromRaw(quantity, d), price: fromRaw(priceOwn, d) };
    }
    const who = this.requireSigner();
    // refuse what the wallet cannot back, before it costs gas (a reverted write does not throw)
    const [gas, have] = await Promise.all([
      this.pc.getBalance({ address: who }),
      this.pc.readContract({ address: oc.collateral, abi: ERC20, functionName: "balanceOf", args: [who] }),
    ]);
    if (gas === 0n) return { sent: false, filled: 0, cost: 0, reason: `out of gas: ${who} holds no ${this.net.native}` };
    if (have < need) return { sent: false, filled: 0, cost: 0, reason: `wallet holds ${fromRaw(have, d).toFixed(2)} USDso, the bet needs ${fromRaw(need, d).toFixed(2)}` };
    let res;
    try {
      res = await withTimeout(this.ex.trader.placeOrder({
        pool: oc.pool, side: side === "up" ? "BUY_YES" : "BUY_NO", price: priceYes, quantity,
        outcomeToken: oc.outcomeToken, yesId: oc.yesId, noId: oc.noId, collateral: oc.collateral,
        orderType: ORDER_TYPE.MARKET, expireTimestampNs: BigInt(expiresAt) * 1_000_000_000n,
      }), CALL_TIMEOUT_MS * 2, "placeOrder");
    } catch (e) {
      const msg = errMsg(e);
      if (msg.includes("ImmediateOrCancelNoFill")) return { sent: true, filled: 0, cost: 0, no_fill: true, reason: "no seller at or under the cap" };
      throw e;
    }
    if (res?.receipt?.status === "reverted") throw new Error(`order reverted on chain (tx ${res.hash})`);
    let filledRaw = 0n, costRaw = 0n;
    for (const f of res.fills ?? []) {
      const q = BigInt(f.quantityFilled ?? 0);
      const pxYes = BigInt(f.fillPrice ?? priceYes);
      const pxOwn = side === "up" ? pxYes : one - pxYes;
      filledRaw += q;
      costRaw += (pxOwn * q) / one;
    }
    const filled = fromRaw(filledRaw, d), cost = fromRaw(costRaw, d);
    return { sent: true, filled, cost, avg_price: filled > 0 ? cost / filled : null, tx: res.hash,
             order_id: res.orderId != null ? String(res.orderId) : null, no_fill: filled === 0,
             reason: filled === 0 ? "nothing crossed" : null, symbol: m.symbol };
  }

  // Redeem what this address holds in one settled window. A void pays both sides 0.5, so both are claimed.
  async claim(marketId) {
    const who = this.requireSigner();
    if (!who) return { claimed: false, dry_run: true, redeemed: [] };
    const oc = await this.onchain(marketId);
    if (!(oc.isResolved || oc.isVoided)) return { claimed: false, redeemed: [], reason: `not settled yet (status ${oc.status})` };
    const held = await this.balances(oc, who);
    const d = Number(oc.decimals ?? this.decimals);
    const claims = oc.isVoided
      ? [[0, held.yes], [1, held.no]].filter(([, amt]) => amt > 0n)
      : [[oc.winningOutcome === 0 ? 0 : 1, oc.winningOutcome === 0 ? held.yes : held.no]].filter(([, amt]) => amt > 0n);
    if (!claims.length) return { claimed: false, redeemed: [], reason: "nothing to claim", held: { up: fromRaw(held.yes, d), down: fromRaw(held.no, d) } };
    const redeemed = [];
    let tx = null;
    for (const [idx, amount] of claims) {
      const res = await withTimeout(this.ex.trader.redeem({ marketId, amount, outcomeIdx: idx }), CALL_TIMEOUT_MS * 2, "redeem");
      if (res?.receipt?.status === "reverted") throw new Error(`redeem reverted on chain (tx ${res.hash})`);
      redeemed.push({ side: idx === 0 ? "up" : "down", amount: fromRaw(amount, d) });
      tx = res.hash;
    }
    return { claimed: true, redeemed, tx, voided: Boolean(oc.isVoided) };
  }

  // Everything the venue lists as claimable for this address, in batches (a restart's catch-up).
  async sweep() {
    const who = this.requireSigner();
    if (!who) return { claimed: 0, markets: 0, dry_run: true };
    const entries = await withTimeout(this.ex.client.getClaimable(who), CALL_TIMEOUT_MS, "claimable");
    const list = (entries ?? []).filter((e) => BigInt(e.amount ?? 0) > 0n)
      .map((e) => ({ marketId: e.marketId, outcomeIdx: e.outcomeIdx, amount: BigInt(e.amount) }));
    let est = 0n;
    for (const e of entries ?? []) est += BigInt(e.estPayout ?? 0);
    const txs = [];
    for (let i = 0; i < list.length; i += 10) {
      const res = await withTimeout(this.ex.trader.redeemMany({ entries: list.slice(i, i + 10) }), CALL_TIMEOUT_MS * 3, "redeemMany");
      if (res?.receipt?.status === "reverted") throw new Error(`redeemMany reverted on chain (tx ${res.hash})`);
      txs.push(res.hash);
    }
    return { claimed: list.length, markets: new Set(list.map((e) => e.marketId)).size, est_payout: fromRaw(est, this.decimals), txs };
  }

  async closeExchange() {
    const ex = this.ex;
    this.ex = null;
    if (!ex) return;
    try { await withTimeout(Promise.resolve(ex.close?.()), 3000, "close"); } catch { /* ignore */ }
  }

  async handle(req) {
    switch (req.op) {
      case "init": return this.init();
      case "markets": return { markets: await this.loadMarkets() };
      case "list": return this.list();
      case "book": return this.book(req.market_id, req.depth);
      case "market": return this.market(req.market_id);
      case "buy": return this.buy(req);
      case "position": return this.position(req.market_id);
      case "claim": return this.claim(req.market_id);
      case "sweep": return this.sweep();
      case "wallet": return this.wallet();
      case "ping": return { pong: true };
      case "shutdown": await this.closeExchange(); setTimeout(() => process.exit(0), 50); return { bye: true };
      default: throw new Error(`unknown op ${req.op}`);
    }
  }
}

const json = (o) => JSON.stringify(o, (_k, v) => (typeof v === "bigint" ? v.toString() : v));

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const exec = new Executor(opts);
  const out = (o) => process.stdout.write(json(o) + "\n");
  let chain = Promise.resolve();
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on("line", (line) => {
    if (!line.trim()) return;
    let req;
    try { req = JSON.parse(line); } catch (e) { out({ id: null, ok: false, error: `bad json: ${errMsg(e)}` }); return; }
    chain = chain.then(async () => {
      try {
        out({ id: req.id, ok: true, ...(await exec.handle(req)) });
      } catch (e) {
        out({ id: req.id, ok: false, error: errMsg(e) });
      }
    });
  });
  rl.on("close", async () => { await chain; await exec.closeExchange(); process.exit(0); });
  process.on("SIGTERM", async () => { await exec.closeExchange(); process.exit(0); });
}

main().catch((e) => { process.stderr.write(`${errMsg(e)}\n`); process.exit(1); });
