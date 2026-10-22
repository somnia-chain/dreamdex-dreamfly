// Perp order sidecar: the one process that signs perp orders for DreamFly.
//
// The runner (Python) starts it and talks JSON lines over stdin and stdout: one request per line, one reply
// per line, with the same `id`. Requests are served one at a time, in order, so the signer's nonces stay in
// order too. It uses the DreamDEX markets SDK the bot kit's perp strategies use, and places the way they do:
// the pool sizes every order (`getMaxPerpOrderSize`), a taker pays a slippage buffer past the touch, a
// rejected order arrives as a named revert, and a close is sized from the pool's own close preview.
//
//   node executor.mjs --network hideki [--owner 0x...] [--dry-run]
//
// PRIVATE_KEY comes from the environment. With --owner the key is a trading key linked to that DreamDEX
// account in the app (wallet, Link a bot, Perps): orders, margin and positions are the account's, and the
// writes go through placeOrderFor / cancelOrderFor as the kit's perp-core sends them. Without it the key
// trades its own MarginBank account. Mainnet has no perp markets and is refused.
//
// Requests ({"id": 1, "op": "...", ...}) and what they return (always with "id" and "ok"):
//   init                      address, markets [{symbol, display, base_decimals, tick, lot, min_qty}], linked {symbol: bool}
//   markets                   reloads the market list (same shape as init)
//   mark {symbol}             {mark} the pool's live mark, or null while the market is un-priceable
//   book {symbol}             {bid, ask} best levels (null when a side is empty)
//   preflight {symbol}        {unlocked, locked, equity, withdrawable, status, leverage, linked}
//   equity                    {equity, unlocked} of the account in the MarginBank (USDso)
//   wallet                    {address, stt, usdso} what the signer's wallet holds outside the MarginBank
//   deposit {symbol, amount}  moves USDso from the wallet into the MarginBank (own account only)
//   place {symbol, side, price, qty, kind, slippage_bps}
//                             side long|short, kind post_only|limit|market; {sent, order_id, filled, no_fill, reason}
//   cancel {symbol, order_id} {ok}
//   open_orders {symbol}      {orders: [{id, side, price, remaining}]}
//   position {symbol}         {position: {side, contracts, entry, mark} | null}
//   close {symbol, fraction}  {sent, filled, left} a taker close of the account's position in that market
//   shutdown                  closes the exchange and exits
//
// Numbers in and out are human units (USDso prices, base quantities). Nothing here reads a .env file.

import { createInterface } from "node:readline";
import {
  ORDER_TYPE,
  SomniaMarkets,
  SOMNIA_HIDEKI_TESTNET_ADDRESSES,
  contractErrorsAbi,
  decodePlaceOrder,
} from "@somnia-chain/markets-sdk";
import { createPublicClient, createWalletClient, defineChain, getAddress, http, isAddress, parseAbi } from "viem";
import { privateKeyToAccount } from "viem/accounts";

const NETWORKS = {
  hideki: {
    rpc: "https://api.hideki2.infra.testnet.somnia.network",
    ws: "wss://api.hideki2.infra.testnet.somnia.network/ws",
    indexer: "https://hideki-dev.smk.somnia.host/v1/graphql",
    chainId: 50383,
    addresses: SOMNIA_HIDEKI_TESTNET_ADDRESSES,
  },
};
// Perp markets exist on Hideki only. Spot and event contracts (Shannon testnet, mainnet) never go through here.

const CALL_TIMEOUT_MS = 60_000;
const TAKER_SLIPPAGE_BPS = 20;          // the kit's default for an IOC: pays this far past the touch
const ORDER_GAS = 10_000_000n;          // what the SDK sends a perp order with; Somnia bills gas used
const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000";

// The trading key path, as the kit's perp-core sends it: the SDK has no perp `...For` writes.
const OPERATOR_ABI = [
  ...parseAbi([
    "function placeOrderFor(address owner, bool isBid, uint64 userData, uint256 price, uint256 quantity, uint64 expireTimestampNs, uint8 orderType, uint8 selfMatchingOption, address builder, uint96 builderFeeBpsTimes1k) payable returns (bool success, uint128 orderId)",
    "function cancelOrderFor(address owner, uint128 orderId)",
    "function isOperatorAuthorized(address owner, address operator, bytes4 selector) view returns (bool)",
  ]),
  ...contractErrorsAbi,
];
const SELECTOR = { placeOrderFor: "0x80054449", cancelOrderFor: "0xe37b444b" };

function parseArgs(argv) {
  const o = { network: "hideki", owner: "", dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--network") o.network = argv[++i];
    else if (a === "--owner") o.owner = argv[++i] ?? "";
    else if (a === "--dry-run") o.dryRun = true;
    else throw new Error(`unknown flag ${a}`);
  }
  if (!NETWORKS[o.network]) throw new Error(`--network ${o.network}: perps trade on the Hideki testnet only (--network hideki, NETWORK=hideki); Shannon and mainnet have no perp markets`);
  if (o.owner && !isAddress(o.owner)) throw new Error(`--owner ${o.owner} is not an address`);
  return o;
}

const withTimeout = (p, ms, label) =>
  Promise.race([
    Promise.resolve(p),
    new Promise((_, rej) => setTimeout(() => rej(new Error(`${label} timed out after ${ms} ms`)), ms)),
  ]);

const errMsg = (e) => String(e?.shortMessage ?? e?.message ?? e).slice(0, 400);

// Human <-> raw on the market's grid. toFixed to the grid's places round-trips exactly; parseUnits of a float
// string can land a few wei off the tick and the pool rejects it as InvalidPrice.
function toRaw(human, decimals) {
  const s = Number(human).toFixed(Math.min(decimals, 9));
  const [i = "0", f = ""] = s.split(".");
  const neg = i.startsWith("-");
  const digits = (neg ? i.slice(1) : i) + f.padEnd(decimals, "0").slice(0, decimals);
  const raw = BigInt(digits || "0");
  return neg ? -raw : raw;
}
const fromRaw = (raw, decimals) => Number(BigInt(raw)) / 10 ** decimals;
const alignDown = (v, grid) => (grid > 0n ? (v / grid) * grid : v);
const alignUp = (v, grid) => (grid > 0n ? ((v + grid - 1n) / grid) * grid : v);

class Executor {
  constructor(opts) {
    this.opts = opts;
    this.net = NETWORKS[opts.network];
    this.ex = null;
    this.markets = new Map();     // symbol -> {symbol, display, info}
    this.tradingKey = null;
    this.me = null;               // the account whose orders these are (the owner in trading key mode)
    const raw = (process.env.PRIVATE_KEY ?? "").trim();
    this.key = raw && !raw.startsWith("0x") ? `0x${raw}` : raw;   // a bare 64 hex key is accepted too
  }

  chain() {
    return defineChain({
      id: this.net.chainId,
      name: `somnia-${this.net.chainId}`,
      nativeCurrency: { name: "Somnia Test Token", symbol: "STT", decimals: 18 },
      rpcUrls: { default: { http: [this.net.rpc], webSocket: [this.net.ws] } },
    });
  }

  async init() {
    if (this.ex) await this.closeExchange();
    const base = { indexerUrl: this.net.indexer, chain: this.chain(), wsRpcUrl: this.net.ws, addresses: this.net.addresses };
    if (this.opts.owner) {
      const owner = getAddress(this.opts.owner);
      // The SDK reads as the owner (positions, margin, sizing are the account's); the key only signs.
      this.ex = new SomniaMarkets({ ...base, account: owner });
      this.me = owner;
      if (this.key) {
        const account = privateKeyToAccount(this.key);
        if (account.address.toLowerCase() === owner.toLowerCase()) {
          throw new Error("OWNER_ADDRESS is this key's own address: leave it blank to trade the key's own account");
        }
        const transport = http(this.net.rpc);
        this.tradingKey = {
          owner, operator: account.address, account,
          publicClient: createPublicClient({ chain: this.chain(), transport }),
          walletClient: createWalletClient({ account, chain: this.chain(), transport }),
        };
      }
    } else {
      this.ex = new SomniaMarkets({ ...base, privateKey: this.key || undefined });
      this.me = this.ex.walletAddress ?? null;
    }
    return this.loadMarkets();
  }

  async loadMarkets() {
    const all = await withTimeout(this.ex.loadMarkets(true), CALL_TIMEOUT_MS, "loadMarkets");
    this.markets.clear();
    for (const m of Object.values(all)) {
      if (m.type !== "swap" || !m.active) continue;
      this.markets.set(m.symbol, { symbol: m.symbol, display: `${m.symbol.split("/")[0]}-PERP`, info: m.info });
    }
    const linked = {};
    if (this.tradingKey) {
      for (const m of this.markets.values()) linked[m.symbol] = await this.linkOk(m);
    }
    return {
      address: this.me,
      signer: this.tradingKey ? this.tradingKey.operator : this.me,
      trading_key: Boolean(this.tradingKey),
      markets: [...this.markets.values()].map((m) => ({
        symbol: m.symbol, display: m.display, base_decimals: Number(m.info.baseDecimals),
        quote_decimals: Number(m.info.quoteDecimals), tick: String(m.info.tickSize), lot: String(m.info.lotSize),
        min_qty: String(m.info.minQuantity),
      })),
      linked,
    };
  }

  market(symbol) {
    const m = this.markets.get(symbol);
    if (!m) throw new Error(`${symbol} is not a live perp market here (${[...this.markets.keys()].join(", ") || "none loaded"})`);
    return m;
  }

  async linkOk(m) {
    const k = this.tradingKey;
    const ask = (selector) => k.publicClient.readContract({
      address: m.info.poolAddress, abi: OPERATOR_ABI, functionName: "isOperatorAuthorized", args: [k.owner, k.operator, selector],
    });
    const [place, cancel] = await Promise.all([ask(SELECTOR.placeOrderFor), ask(SELECTOR.cancelOrderFor)]);
    return Boolean(place && cancel);
  }

  async mark(symbol) {
    const m = this.market(symbol);
    try {
      const snap = await withTimeout(this.ex.client.getPerpHealthSnapshot(m.info.poolAddress), CALL_TIMEOUT_MS, "mark");
      if (!snap.priceable) return { mark: null };
      const raw = Number(snap.markPrice);
      return { mark: raw > 0 ? raw / 1e18 : null };
    } catch (e) {
      return { mark: null, error: errMsg(e) };
    }
  }

  async book(symbol) {
    const m = this.market(symbol);
    const b = await withTimeout(this.ex.fetchOrderBook(m.symbol, 1), CALL_TIMEOUT_MS, "book");
    return { bid: b.bids[0]?.[0] ?? null, ask: b.asks[0]?.[0] ?? null };
  }

  requireAccount() {
    if (!this.me) throw new Error("no account: PRIVATE_KEY is not set");
    return this.me;
  }

  async marginOf(m) {
    const a = await withTimeout(this.ex.client.getMarginAccount(m.info.marginBank, this.requireAccount()), CALL_TIMEOUT_MS, "margin");
    const q = Number(m.info.quoteDecimals);
    return {
      unlocked: fromRaw(a.unlockedCollateralBalance, q), locked: fromRaw(a.lockedCollateral, q), equity: fromRaw(a.equity, q),
      withdrawable: fromRaw(a.withdrawable, q), status: String(a.marginStatus),
    };
  }

  async preflight(symbol) {
    const m = this.market(symbol);
    const margin = await this.marginOf(m);
    let leverage = null;
    try {
      leverage = await this.ex.client.getPerpMaxLeverage({ marginBank: m.info.marginBank, account: this.requireAccount(), pool: m.info.poolAddress });
    } catch { /* an unset lever reads as 0 on most deployments; a read failure is not a reason to stop */ }
    return { ...margin, leverage, linked: this.tradingKey ? await this.linkOk(m) : true };
  }

  async equity() {
    const first = this.markets.values().next().value;
    if (!first) throw new Error("no markets loaded");
    const margin = await this.marginOf(first);
    return { equity: margin.equity, unlocked: margin.unlocked };
  }

  async wallet() {
    // what the SIGNER holds in its wallet: gas (STT) and the collateral token outside the MarginBank
    const first = this.markets.values().next().value;
    if (!first) throw new Error("no markets loaded");
    const who = this.tradingKey ? this.tradingKey.operator : this.requireAccount();
    const pc = this.tradingKey ? this.tradingKey.publicClient : createPublicClient({ chain: this.chain(), transport: http(this.net.rpc) });
    const stt = await pc.getBalance({ address: who });
    const erc = parseAbi(["function balanceOf(address) view returns (uint256)"]);
    const usdso = await pc.readContract({ address: first.info.quoteToken, abi: erc, functionName: "balanceOf", args: [who] });
    return { address: who, stt: fromRaw(stt, 18), usdso: fromRaw(usdso, Number(first.info.quoteDecimals)) };
  }

  async deposit(symbol, amount) {
    const m = this.market(symbol);
    if (this.tradingKey) throw new Error("a trading key cannot move funds: deposit from the account in the app (Perps <-> Spot)");
    if (this.opts.dryRun) return { sent: false, dry_run: true };
    const res = await withTimeout(this.ex.trader.depositMargin({
      marginBank: m.info.marginBank, pool: m.info.poolAddress, amount: toRaw(amount, Number(m.info.quoteDecimals)), autoApprove: true,
    }), CALL_TIMEOUT_MS * 3, "deposit");
    if (res?.receipt?.status === "reverted") throw new Error(`deposit reverted on chain (tx ${res.hash})`);
    return { sent: true, tx: res.hash };
  }

  async sized(m, side, priceRaw, wantedRaw, liftTo) {
    const max = await withTimeout(this.ex.client.getMaxPerpOrderSize({
      pool: m.info.poolAddress, marginBank: m.info.marginBank, account: this.requireAccount(), isBid: side === "long",
      price: priceRaw, autoPull: false,
    }), CALL_TIMEOUT_MS, "size");
    if (!max.priceable) return { ok: false, reason: "market is un-priceable right now (stale mark)" };
    if (!max.placeable) return { ok: false, reason: `nothing can be placed: ${max.limitedBy ?? "unknown limit"}` };
    const capped = wantedRaw < max.maxQuantity ? wantedRaw : max.maxQuantity;
    const lot = BigInt(m.info.lotSize), min = BigInt(m.info.minQuantity);
    let q = alignDown(capped, lot);
    if (q < min && liftTo && wantedRaw > 0n) {
      // a quote smaller than the market minimum (a small stake on BTC) is lifted to the minimum when the caller's
      // notional cap covers it and the margin allows; otherwise it is refused below
      const lifted = alignUp(min, lot);
      const notional = fromRaw(lifted, Number(m.info.baseDecimals)) * fromRaw(priceRaw, 18);
      if (notional <= liftTo && lifted <= max.maxQuantity) return { ok: true, quantity: lifted, limited_by: "market minimum" };
    }
    if (q < min || q <= 0n) {
      return { ok: false, reason: `size rounds below the market minimum (${fromRaw(min, Number(m.info.baseDecimals))}); limited by ${max.limitedBy ?? "the request"}` };
    }
    return { ok: true, quantity: q, limited_by: max.limitedBy ?? null };
  }

  async sendFor(pool, functionName, args) {
    const k = this.tradingKey;
    const { request } = await k.publicClient.simulateContract({ address: pool, abi: OPERATOR_ABI, functionName, args, account: k.account, gas: ORDER_GAS });
    const hash = await k.walletClient.writeContract(request);
    const receipt = await k.publicClient.waitForTransactionReceipt({ hash });
    if (receipt.status !== "success") throw new Error(`${functionName} reverted on chain (tx ${hash})`);
    return { hash, receipt };
  }

  async place({ symbol, side, price, qty, kind = "limit", slippage_bps, reduce = false, lift_to = 0 }) {
    const m = this.market(symbol);
    if (!["long", "short"].includes(side)) throw new Error(`side must be long or short, not ${side}`);
    const orderType = { post_only: ORDER_TYPE.POST_ONLY, limit: ORDER_TYPE.LIMIT, market: ORDER_TYPE.MARKET }[kind];
    if (orderType === undefined) throw new Error(`kind must be post_only, limit or market, not ${kind}`);
    const taker = orderType === ORDER_TYPE.MARKET;
    const slip = BigInt(Math.max(0, Math.round(slippage_bps ?? (taker ? TAKER_SLIPPAGE_BPS : 0))));
    let priceRaw = toRaw(price, 18);
    if (slip > 0n) priceRaw = side === "long" ? (priceRaw * (10_000n + slip)) / 10_000n : (priceRaw * (10_000n - slip)) / 10_000n;
    const tick = BigInt(m.info.tickSize);
    // a bid is rounded down and an ask up, so the aligned price never crosses further than the buffer allowed
    priceRaw = side === "long" ? alignDown(priceRaw, tick) : alignUp(priceRaw, tick);
    if (priceRaw <= 0n) throw new Error(`price ${price} is not positive on the tick grid`);
    const wanted = toRaw(qty, Number(m.info.baseDecimals));
    // a reducing order is sent at the size the close preview gave: the pool accepts a reduce even when the
    // account has no free margin left, which is exactly when the close matters most
    const s = reduce
      ? { ok: alignDown(wanted, BigInt(m.info.lotSize)) >= BigInt(m.info.minQuantity), quantity: alignDown(wanted, BigInt(m.info.lotSize)), limited_by: "reduce", reason: "size below the market minimum" }
      : await this.sized(m, side, priceRaw, wanted, Number(lift_to) || 0);
    if (!s.ok) return { sent: false, filled: 0, reason: s.reason };
    const human = fromRaw(s.quantity, Number(m.info.baseDecimals));
    if (this.opts.dryRun) return { sent: false, dry_run: true, filled: 0, qty: human, price: fromRaw(priceRaw, 18) };
    let res;
    try {
      const order = { pool: m.info.poolAddress, isBid: side === "long", price: priceRaw, quantity: s.quantity, orderType };
      if (this.tradingKey) {
        const { hash, receipt } = await withTimeout(this.sendFor(m.info.poolAddress, "placeOrderFor", [
          this.tradingKey.owner, order.isBid, 0n, order.price, order.quantity, farFutureNs(), orderType, 0, ZERO_ADDRESS, 0n,
        ]), CALL_TIMEOUT_MS * 2, "placeOrderFor");
        const { orderId, fills } = decodePlaceOrder(receipt, m.info.poolAddress);
        res = { hash, orderId, fills };
      } else {
        res = await withTimeout(this.ex.trader.placePerpOrder(order), CALL_TIMEOUT_MS * 2, "placePerpOrder");
      }
    } catch (e) {
      const msg = errMsg(e);
      // the two ordinary outcomes of a moving book: nothing to cross, or a maker that would have taken
      if (msg.includes("ImmediateOrCancelNoFill")) return { sent: true, filled: 0, no_fill: true, reason: "nothing crossed" };
      if (msg.includes("PostOnlyWouldCross")) return { sent: false, filled: 0, no_fill: true, reason: "post only would cross" };
      throw e;
    }
    if (res?.receipt?.status === "reverted") throw new Error(`order reverted on chain (tx ${res.hash})`);
    const filled = (res.fills ?? []).reduce((sum, f) => sum + BigInt(f.quantityFilled ?? 0), 0n);
    return {
      sent: true, order_id: res.orderId === undefined ? null : String(res.orderId), filled: fromRaw(filled, Number(m.info.baseDecimals)),
      qty: human, price: fromRaw(priceRaw, 18), tx: res.hash ?? null, limited_by: s.limited_by,
    };
  }

  async cancel({ symbol, order_id }) {
    const m = this.market(symbol);
    if (this.opts.dryRun) return { ok: true, dry_run: true };
    const id = BigInt(order_id);
    try {
      if (this.tradingKey) await withTimeout(this.sendFor(m.info.poolAddress, "cancelOrderFor", [this.tradingKey.owner, id]), CALL_TIMEOUT_MS * 2, "cancelOrderFor");
      else await withTimeout(this.ex.trader.cancelOrder({ pool: m.info.poolAddress, orderId: id }), CALL_TIMEOUT_MS * 2, "cancelOrder");
      return { ok: true };
    } catch (e) {
      // an order that filled or expired between the read and the cancel is gone either way
      return { ok: false, error: errMsg(e) };
    }
  }

  async openOrders(symbol) {
    const m = this.market(symbol);
    this.requireAccount();
    const rows = await withTimeout(this.ex.fetchOpenOrders(m.symbol), CALL_TIMEOUT_MS, "open orders");
    return { orders: rows.map((o) => ({ id: String(o.id), side: o.side, price: o.price ?? null, remaining: o.remaining })) };
  }

  async position(symbol) {
    const m = this.market(symbol);
    this.requireAccount();
    const all = await withTimeout(this.ex.fetchPositions([m.symbol]), CALL_TIMEOUT_MS, "positions");
    const p = all.find((x) => x.symbol === m.symbol);
    return { position: p ? { side: p.side, contracts: p.contracts, entry: p.entryPrice, mark: p.markPrice ?? null } : null };
  }

  async close({ symbol, fraction = 1 }) {
    const m = this.market(symbol);
    const account = this.requireAccount();
    const preview = await withTimeout(this.ex.client.previewPerpClosePnl({ pool: m.info.poolAddress, marginBank: m.info.marginBank, account }), CALL_TIMEOUT_MS, "close preview");
    if (!preview.priceable) return { sent: false, filled: 0, left: null, reason: "market is un-priceable: the close preview has no size" };
    const whole = preview.closedQuantity;
    if (whole <= 0n) return { sent: false, filled: 0, left: 0, reason: "flat" };
    const frac = Math.min(1, Math.max(0, Number(fraction)));
    const wanted = frac === 1 ? whole : alignDown((whole * BigInt(Math.round(frac * 1000))) / 1000n, BigInt(m.info.lotSize));
    if (wanted < BigInt(m.info.minQuantity) || wanted <= 0n) return { sent: false, filled: 0, left: fromRaw(whole, Number(m.info.baseDecimals)), reason: "below the market minimum" };
    const { position } = await this.position(symbol);
    if (!position) return { sent: false, filled: 0, left: 0, reason: "flat" };
    const side = position.side === "long" ? "short" : "long";
    const b = await this.book(symbol);
    const touch = side === "long" ? b.ask : b.bid;
    if (touch === null) return { sent: false, filled: 0, left: position.contracts, reason: "no counterparty resting on the book" };
    const r = await this.place({ symbol, side, price: touch, qty: fromRaw(wanted, Number(m.info.baseDecimals)), kind: "market", reduce: true });
    const after = this.opts.dryRun ? position : (await this.position(symbol)).position;
    return { ...r, left: after ? after.contracts : 0 };
  }

  async closeExchange() {
    try {
      await Promise.race([Promise.resolve(this.ex?.close?.()), new Promise((r) => setTimeout(r, 3000))]);
    } catch { /* a close that fails is not worth failing a run over */ }
    this.ex = null;
  }

  async handle(req) {
    switch (req.op) {
      case "init": return this.init();
      case "markets": return this.loadMarkets();
      case "mark": return this.mark(req.symbol);
      case "book": return this.book(req.symbol);
      case "preflight": return this.preflight(req.symbol);
      case "equity": return this.equity();
      case "wallet": return this.wallet();
      case "deposit": return this.deposit(req.symbol, req.amount);
      case "place": return this.place(req);
      case "cancel": return this.cancel(req);
      case "open_orders": return this.openOrders(req.symbol);
      case "position": return this.position(req.symbol);
      case "close": return this.close(req);
      case "ping": return { pong: true };
      case "shutdown": await this.closeExchange(); setTimeout(() => process.exit(0), 50); return { bye: true };
      default: throw new Error(`unknown op ${req.op}`);
    }
  }
}

function farFutureNs() {
  return BigInt(Math.floor(Date.now() / 1000) + 50 * 365 * 24 * 3600) * 1_000_000_000n;
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
