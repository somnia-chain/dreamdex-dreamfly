// Spot sidecar for networks the DreamDEX web API and the bot kit's spot core do not cover (Hideki): market data
// and orders through the DreamDEX markets SDK, the same way the perp and event contract sidecars work.
//
// The runner (Python) starts it and talks JSON lines over stdin and stdout: one request per line, one reply per
// line, with the same `id`, served one at a time so the signer's nonces stay in order.
//
//   node spot_executor.mjs --network hideki [--dry-run]
//
// PRIVATE_KEY comes from the environment; without it (or with --dry-run) only the reads work. Symbols are the
// runner's BASE:QUOTE form (WETH:USDso); the SDK's BASE/QUOTE form is used inside.
//
// Requests ({"id": 1, "op": "...", ...}) and what they return (always with "id" and "ok"):
//   init                      {address, chain, markets: [symbol]}
//   book {symbol, depth}      {bids: [[price, qty]], asks: [[price, qty]]}, best first
//   candles {symbol, limit}   {candles: [{timestamp, open, high, low, close, volume}]} 1 minute, oldest first
//   place {symbol, side, price, qty, post_only}
//                             side buy|sell; post_only rests or is refused, otherwise immediate-or-cancel
//                             {order_id, status, filled, tx}
//   cancel {symbol, order_id} {canceled}
//   balances {symbol}         {base, quote} the signer's wallet, in human units
//   wallet                    {address, native}
//   ping, shutdown
import { createInterface } from "node:readline";
import { SomniaMarkets, SOMNIA_HIDEKI_TESTNET_ADDRESSES } from "@somnia-chain/markets-sdk";
import { createPublicClient, defineChain, http } from "viem";

const NETWORKS = {
  hideki: {
    rpc: "https://api.hideki2.infra.testnet.somnia.network",
    ws: "wss://api.hideki2.infra.testnet.somnia.network/ws",
    indexer: "https://hideki-dev.smk.somnia.host/v1/graphql",
    chainId: 50383,
    addresses: SOMNIA_HIDEKI_TESTNET_ADDRESSES,
    native: "STT",
  },
};
const CALL_TIMEOUT_MS = 60_000;

function parseArgs(argv) {
  const o = { network: "hideki", dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--network") o.network = argv[++i];
    else if (a === "--dry-run") o.dryRun = true;
    else throw new Error(`unknown flag ${a}`);
  }
  if (!NETWORKS[o.network]) throw new Error(`--network ${o.network}: this sidecar serves ${Object.keys(NETWORKS).join(", ")}; testnet and mainnet spot go through the bot kit core`);
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
const sdkSymbol = (s) => String(s).replace(":", "/");

class Executor {
  constructor(opts) {
    this.opts = opts;
    this.net = NETWORKS[opts.network];
    this.ex = null;
    this.me = null;
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
    await this.closeExchange();
    this.ex = new SomniaMarkets({
      indexerUrl: this.net.indexer, chain: this.chain(), wsRpcUrl: this.net.ws, addresses: this.net.addresses,
      privateKey: this.opts.dryRun ? undefined : this.key || undefined,
    });
    this.me = this.ex.walletAddress ?? null;
    const all = await withTimeout(this.ex.loadMarkets(true), CALL_TIMEOUT_MS * 2, "loadMarkets");
    const markets = Object.values(all ?? {}).filter((m) => m.type === "spot" && m.active).map((m) => m.symbol.replace("/", ":"));
    return { address: this.me, chain: this.net.chainId, markets, dry_run: this.opts.dryRun };
  }

  market(symbol) {
    const m = this.ex.markets?.[sdkSymbol(symbol)];
    if (!m || m.type !== "spot") throw new Error(`${symbol} is not a spot market here`);
    return m;
  }

  async book(symbol, depth = 5) {
    const ob = await withTimeout(this.ex.fetchOrderBook(sdkSymbol(symbol), depth), CALL_TIMEOUT_MS, "book");
    const side = (levels) => (levels ?? []).slice(0, depth).map(([p, q]) => [Number(p), Number(q)]);
    return { bids: side(ob?.bids), asks: side(ob?.asks) };
  }

  async candles(symbol, limit = 300) {
    const rows = await withTimeout(this.ex.fetchOHLCV(sdkSymbol(symbol), "1m", undefined, limit), CALL_TIMEOUT_MS, "candles");
    return { candles: (rows ?? []).map(([t, o, h, l, c, v]) => ({ timestamp: Number(t), open: Number(o), high: Number(h),
      low: Number(l), close: Number(c), volume: Number(v ?? 0) })) };
  }

  requireSigner() {
    if (this.opts.dryRun) throw new Error("dry run: nothing is signed");
    if (!this.me) throw new Error("no signer: PRIVATE_KEY is not set");
    return this.me;
  }

  async place({ symbol, side, price, qty, post_only }) {
    this.requireSigner();
    if (!["buy", "sell"].includes(side)) throw new Error(`side must be buy or sell, not ${side}`);
    this.market(symbol);
    const o = await withTimeout(this.ex.createOrder(sdkSymbol(symbol), "limit", side, Number(qty), Number(price),
      { timeInForce: post_only ? "PO" : "IOC" }), CALL_TIMEOUT_MS * 2, "createOrder");
    return { order_id: o?.id != null ? String(o.id) : null, status: o?.status ?? null, filled: Number(o?.filled ?? 0),
             tx: o?.info?.hash ?? o?.info?.receipt?.transactionHash ?? null };
  }

  async cancel({ symbol, order_id }) {
    this.requireSigner();
    const r = await withTimeout(this.ex.cancelOrder(String(order_id), sdkSymbol(symbol)), CALL_TIMEOUT_MS * 2, "cancelOrder");
    return { canceled: r?.status === "canceled" };
  }

  async balances(symbol) {
    this.requireSigner();
    const m = this.market(symbol);
    const b = await withTimeout(this.ex.fetchBalance(), CALL_TIMEOUT_MS, "balance");
    const free = (code) => Number(b?.[code]?.free ?? b?.free?.[code] ?? b?.[code]?.total ?? 0);
    return { base: free(m.base), quote: free(m.quote) };
  }

  async wallet() {
    const who = this.requireSigner();
    return { address: who, native: Number(await this.pc.getBalance({ address: who })) / 1e18 };
  }

  async closeExchange() {
    const ex = this.ex;
    this.ex = null;
    if (!ex) return;
    try { await withTimeout(Promise.resolve(ex.close?.()), 3000, "close"); } catch { /* ignore */ }
  }

  async handle(req) {
    if (req.op !== "init" && req.op !== "ping" && req.op !== "shutdown" && !this.ex) await this.init();
    switch (req.op) {
      case "init": return this.init();
      case "book": return this.book(req.symbol, req.depth);
      case "candles": return this.candles(req.symbol, req.limit);
      case "place": return this.place(req);
      case "cancel": return this.cancel(req);
      case "balances": return this.balances(req.symbol);
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
