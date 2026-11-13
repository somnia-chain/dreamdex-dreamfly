/* DreamFly monitor: a paper-trading terminal drawn on a canvas (used as a CanvasTexture).
   Layout follows the DreamDEX trading page (market header with stats, chart left, order book middle,
   order ticket right, orders below) using the app's default "Voltron" palette. It is labelled "paper".
   The logo geometry is the app's own (dreamdex-logo.js); this is built for DreamDEX. Font: Geist (the app's UI face), falling back to system sans. */
(function (root) {
  'use strict';

  var C = {
    page: '#0b0e12', surface: '#14191f', raised: '#1e252d', overlay: '#2a333d',
    heading: '#f2f5f8', body: '#c9d1d9', muted: '#a2adb8', subtle: '#909ca8',
    accent: '#6e6eed', accentText: '#a39cff', action: '#5b55c8', buy: '#00fa89', sell: '#dc0b4a',
    line: 'rgba(242,245,248,0.08)', frame: 'rgba(242,245,248,0.18)'
  };
  var ARM_COL = { maker: '#8a8aff', grid: '#c9a6ff', trend: '#ffcf6b', flat: '#a2adb8',
    p_maker: '#8a8aff', p_trend: '#ffcf6b', p_carry: '#7fe0d6', ec_late: '#ffcf6b', ec_sharp: '#c9a6ff' };
  var SANS = "Geist, 'Segoe UI', system-ui, sans-serif";
  var MONO = "'Geist Mono', ui-monospace, 'SF Mono', Menlo, monospace";

  function fmtPx(p) {
    if (p == null || isNaN(p)) return '-';
    return Number(p).toFixed(p >= 1000 ? 2 : p >= 2 ? 3 : 5);   // a stable pair around 1.0000 moves in 0.0001 steps
  }
  function fmtMoney(n, sign) {
    if (n == null || isNaN(n)) return '-';
    var s = Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2);
    return sign && n > 0 ? '+' + s : s;
  }
  function rr(x, X, y, Y, w, h, r) {
    X.beginPath();
    X.moveTo(x + r, y); X.arcTo(x + w, y, x + w, y + h, r); X.arcTo(x + w, y + h, x, y + h, r);
    X.arcTo(x, y + h, x, y, r); X.arcTo(x, y, x + w, y, r); X.closePath();
  }

  function create(canvas, opts) {
    var compact = !!(opts && opts.compact);
    var X = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;

    var F = 1.55; // canvas is 1600 wide and shown ~40% size on the monitor, so type is set large
    function text(s, x, y, size, color, weight, align, mono) {
      size = Math.round(size * F);
      X.font = (weight || 400) + ' ' + size + 'px ' + (mono ? MONO : SANS);
      X.fillStyle = color; X.textAlign = align || 'left'; X.textBaseline = 'middle';
      X.fillText(s, x, y);
    }
    /* the {d} dreamdex lockup and the {d} monogram, from the app's own geometry (dreamdex-logo.js) */
    var LG = window.DREAMDEX_LOGO, LOGO = null;
    if (LG && window.Path2D) LOGO = { mark: LG.mark.map(function (d) { return new Path2D(d); }), word: LG.word.map(function (d) { return new Path2D(d); }) };
    function logo(x, y, h, full) {
      if (!LOGO) return 0;
      var sc = h / LG.viewBox[1];
      X.save(); X.translate(x, y); X.scale(sc, sc); X.fillStyle = C.heading;
      LOGO.mark.forEach(function (p) { X.fill(p); });
      if (full) { X.translate(LG.wordAt[0], LG.wordAt[1]); LOGO.word.forEach(function (p) { X.fill(p); }); }
      X.restore();
      return (full ? LG.viewBox[0] : LG.markViewBox[0]) * sc;
    }
    function panel(x, y, w, h) {
      X.fillStyle = C.surface; rr(x, X, y, 0, w, h, 4); X.fill();
      X.strokeStyle = C.line; X.lineWidth = 1; X.stroke();
    }

    function drawChart(x, y, w, h, candles, src) {
      panel(x, y, w, h);
      text('1m', x + 18, y + 22, 14, C.accentText, 600);
      text(src === 'api' ? 'last 60 candles' : src === 'tape' ? 'mark, one row a minute from the recorder' : 'built from this page\'s touch samples', x + 74, y + 22, 13, C.muted, 400);
      var cx = x + 14, cy = y + 42, cw = w - 118, chh = h - 42 - 62, vh = 46;
      if (!candles.length) { text('waiting for candles', x + w / 2, y + h / 2, 14, C.muted, 400, 'center'); return; }
      var lo = Infinity, hi = -Infinity, vmax = 0;
      candles.forEach(function (c) { lo = Math.min(lo, c.l); hi = Math.max(hi, c.h); vmax = Math.max(vmax, c.v); });
      if (hi === lo) { hi += hi * 1e-4 || 1e-6; lo -= lo * 1e-4 || 1e-6; }
      var pad = (hi - lo) * 0.08; hi += pad; lo -= pad;
      var Y = function (p) { return cy + chh - (chh * (p - lo)) / (hi - lo); };
      X.strokeStyle = C.line; X.lineWidth = 1;
      for (var g = 0; g <= 4; g++) {
        var gy = Math.round(cy + (chh * g) / 4) + 0.5;
        X.beginPath(); X.moveTo(cx, gy); X.lineTo(cx + cw, gy); X.stroke();
        text(fmtPx(hi - ((hi - lo) * g) / 4), cx + cw + 8, gy, 11, C.muted, 400, 'left', true);
      }
      var n = candles.length, step = cw / n, bw = Math.max(2, step * 0.62);
      candles.forEach(function (c, i) {
        var mx = cx + step * (i + 0.5), up = c.c >= c.o, col = up ? C.buy : C.sell;
        X.strokeStyle = col; X.fillStyle = col; X.lineWidth = 1.4;
        X.beginPath(); X.moveTo(mx, Y(c.h)); X.lineTo(mx, Y(c.l)); X.stroke();
        var top = Y(Math.max(c.o, c.c)), bh = Math.max(2, Math.abs(Y(c.o) - Y(c.c)));
        X.fillRect(mx - bw / 2, top, bw, bh);
        if (vmax > 0) {
          X.globalAlpha = 0.35;
          var vv = (vh * c.v) / vmax; X.fillRect(mx - bw / 2, cy + chh + 12 + vh - vv, bw, vv);
          X.globalAlpha = 1;
        }
      });
      var last = candles[n - 1];
      var oc = last.c >= last.o ? C.buy : C.sell;
      text('O ' + fmtPx(last.o) + '  H ' + fmtPx(last.h) + '  L ' + fmtPx(last.l) + '  C ' + fmtPx(last.c), x + w * 0.42, y + 22, 11, oc, 400, 'left', true);
      var ly = Math.round(Y(last.c)) + 0.5;
      X.setLineDash([4, 4]); X.strokeStyle = C.frame; X.beginPath(); X.moveTo(cx, ly); X.lineTo(cx + cw, ly); X.stroke(); X.setLineDash([]);
      X.fillStyle = last.c >= last.o ? C.buy : C.sell; rr(cx + cw + 4, X, ly - 14, 0, 106, 28, 5); X.fill();
      text(fmtPx(last.c), cx + cw + 10, ly, 11, '#000', 600, 'left', true);
    }

    function drawBook(x, y, w, h, s) {
      panel(x, y, w, h);
      text('Order book', x + 16, y + 24, 13, C.heading, 500);
      X.fillStyle = C.accent; X.fillRect(x + 16, y + 40, X.measureText('Order book').width, 2);
      text('Trades', x + 16 + X.measureText('Order book').width + 32, y + 24, 13, C.muted, 400);
      var base = String(s.symbol || '').split(/[:\/]/)[0];
      var hy = y + 60;
      text('Price', x + 16, hy, 10, C.muted, 400);
      text('Size (' + base + ')', x + w * 0.62, hy, 10, C.muted, 400, 'right');
      text('Total (' + base + ')', x + w - 14, hy, 10, C.muted, 400, 'right');
      var tc = s.touch, rows = 7, rh = (h - 74 - 56) / (rows * 2), mid = y + 70 + rows * rh;
      if (!tc) { text('no touch yet', x + w / 2, y + h / 2, 16, C.muted, 400, 'center'); return; }
      var spread = Math.max(tc[1] - tc[0], tc[1] * 1e-5), tick = spread;
      var own = function (list) {
        var m = {};
        (list || []).forEach(function (q) {
          var p = typeof q === 'number' ? q : Array.isArray(q) ? q[0] : q.price != null ? q.price : q.p;
          var z = typeof q === 'number' ? null : Array.isArray(q) ? q[1] : q.size != null ? q.size : q.q;
          m[Number(p).toFixed(8)] = z == null ? 0 : z;
        });
        return m;
      };
      var ownA = own(s.asks), ownB = own(s.bids);
      function row(price, side, idx, best) {
        var ry = side === 'ask' ? mid - (idx + 1) * rh : mid + 50 + idx * rh;
        var col = side === 'ask' ? C.sell : C.buy;
        var key = Number(price).toFixed(8), o = (side === 'ask' ? ownA : ownB)[key];
        var depth = o != null ? 0.7 : 0.12 + 0.05 * (rows - idx) / rows;
        X.globalAlpha = best ? 0.28 : 0.13;
        X.fillStyle = col; rr(x + w - 12 - (w - 24) * depth, X, ry + 2, 0, (w - 24) * depth, rh - 4, 3); X.fill();
        X.globalAlpha = 1;
        text(fmtPx(price), x + 16, ry + rh / 2, 13, col, best ? 600 : 400, 'left', true);
        if (o != null) {
          text(o ? Number(o).toPrecision(3) : 'paper', x + w * 0.62, ry + rh / 2, 13, C.heading, 500, 'right', true);
          rr(x + w - 78, X, ry + rh / 2 - 11, 0, 64, 22, 11); X.fillStyle = C.accent; X.fill();
          text('paper', x + w - 46, ry + rh / 2, 9, '#fff', 600, 'center');
        }
      }
      for (var i = 0; i < rows; i++) { row(tc[1] + tick * i, 'ask', i, i === 0); }
      for (var j = 0; j < rows; j++) { row(tc[0] - tick * j, 'bid', j, j === 0); }
      var m = (tc[0] + tc[1]) / 2;
      X.fillStyle = C.raised; X.fillRect(x + 8, mid + 4, w - 16, 42);
      text(fmtPx(m), x + 16, mid + 25, 14, C.heading, 600, 'left', true);
      text(((tc[1] - tc[0]) / m * 1e4).toFixed(1) + ' bps', x + w - 14, mid + 25, 9, C.muted, 400, 'right', true);
    }

    function drawForm(x, y, w, h, s) {
      panel(x, y, w, h);
      var pad = 16, iw = w - pad * 2, short = h < 420, tt = short ? 0 : 46;   // the tablet and phone panel is short: no order type tabs
      if (!short) {   // Market | Limit | Pro, as in the app
        rr(x + pad, X, y + 12, 0, iw, 36, 8); X.fillStyle = C.raised; X.fill();
        rr(x + pad + 4, X, y + 16, 0, iw / 3 - 6, 28, 6); X.fillStyle = C.overlay; X.fill();
        ['Market', 'Limit', 'Pro'].forEach(function (t, i) { text(t, x + pad + iw / 6 + i * (iw / 3), y + 30, 11, i === 0 ? C.accentText : C.body, i === 0 ? 600 : 400, 'center'); });
      }
      // buy / sell: the active side is a tinted pill with coloured text
      rr(x + pad, X, y + 16 + tt, 0, iw, 44, 8); X.fillStyle = C.raised; X.fill();
      rr(x + pad + 4, X, y + 20 + tt, 0, iw / 2 - 6, 36, 6); X.fillStyle = 'rgba(0,250,137,0.22)'; X.fill();
      text('Buy', x + pad + iw / 4, y + 38 + tt, 12, C.buy, 600, 'center');
      text('Sell', x + pad + (iw * 3) / 4, y + 38 + tt, 12, C.body, 500, 'center');

      text('strategy', x + pad, y + 86 + tt, 10, C.muted, 400);
      var pitch = short ? 46 : 52, bhArm = short ? 38 : 44;   // the tablet and phone panel is short: tighter rows
      var arms = s.arms, yy = y + 104 + tt, sel = s.arm;
      var sc = s.scores || [0, 0, 0, 0], mn = Math.min.apply(null, sc), mxs = Math.max.apply(null, sc);
      arms.forEach(function (a, i) {
        var on = a === sel, bx = x + pad + (i % 2) * (iw / 2 + 3), by = yy + Math.floor(i / 2) * pitch, bw = iw / 2 - 3;
        rr(bx, X, by, 0, bw, bhArm, 22);
        X.fillStyle = on ? C.accent : C.raised; X.fill();
        if (on) { X.strokeStyle = C.accentText; X.lineWidth = 1.5; X.stroke(); }
        var frac = mxs > mn ? (sc[i] - mn) / (mxs - mn) : 0;
        text(a, bx + 14, by + bhArm / 2, 11, on ? '#fff' : C.body, on ? 600 : 400);
        X.fillStyle = on ? 'rgba(255,255,255,.55)' : ARM_COL[a] || C.muted;
        X.fillRect(bx + bw - 40, by + bhArm / 2 - 2, 26 * Math.max(0.08, frac), 4);
      });

      var ey = yy + 2 * pitch + 20;
      text('equity', x + pad, ey, 10, C.muted, 400);
      text(fmtMoney(s.equity), x + pad, ey + 34, 17, C.heading, 600, 'left', true);
      text('day p&l', x + w - pad, ey, 10, C.muted, 400, 'right');
      var pc = s.dayPnl > 0 ? C.buy : s.dayPnl < 0 ? C.sell : C.body;
      text(fmtMoney(s.dayPnl, true), x + w - pad, ey + 34, 17, pc, 600, 'right', true);
      // equity by episode
      var by2 = y + h - (short ? 56 : 68);   // the paper button below is anchored to the panel bottom
      var sy = ey + (short ? 48 : 66), sh = Math.max(14, Math.min(60, by2 - 10 - sy));   // the chart ends above it, however short the panel is
      if (s.eqSeries && s.eqSeries.length > 1) {
        var ys = s.eqSeries, lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
        if (hi === lo) { hi += 1e-9; lo -= 1e-9; }
        X.strokeStyle = ys[ys.length - 1] >= ys[0] ? C.buy : C.sell; X.lineWidth = 1.6; X.beginPath();
        ys.forEach(function (v, i) {
          var px = x + pad + (iw * i) / (ys.length - 1), py = sy + sh - (sh * (v - lo)) / (hi - lo);
          i ? X.lineTo(px, py) : X.moveTo(px, py);
        });
        X.stroke();
      } else text('equity trace fills as episodes end', x + pad, sy + 26, 9, C.muted, 400);

      // primary action: paper mode, disabled look
      rr(x + pad, X, by2, 0, iw, short ? 44 : 52, 12); X.fillStyle = 'rgba(91,85,200,0.55)'; X.fill();
      text('paper: simulated fills', x + w / 2, by2 + (short ? 22 : 26), 11, '#fff', 600, 'center');
    }

    function drawOrders(x, y, w, h, s) {
      panel(x, y, w, h);
      var tabs = ['Open orders', 'Fills', 'Inventory'], tx = x + 18;
      tabs.forEach(function (t, i) {
        text(t, tx, y + 26, 11, i === 0 ? C.heading : C.muted, i === 0 ? 600 : 400);
        if (i === 0) { X.fillStyle = C.accent; X.fillRect(tx, y + 42, X.measureText(t).width, 3); }
        tx += X.measureText(t).width + 32;
      });
      var cols = [x + 18, x + w * 0.22, x + w * 0.42, x + w * 0.62, x + w * 0.82];
      ['SIDE', 'PRICE', 'STRATEGY', 'FILLS', 'INVENTORY'].forEach(function (t, i) { text(t, cols[i], y + 70, 9, C.muted, 400); });
      var qs = [];
      (s.asks || []).slice(0, 2).forEach(function (q) { qs.push(['sell', q]); });
      (s.bids || []).slice(0, 2).forEach(function (q) { qs.push(['buy', q]); });
      if (!qs.length) text('no resting paper quotes: the ' + (s.arm || 'flat') + ' arm is holding back', x + 18, y + 108, 11, C.muted, 400);
      qs.slice(0, 3).forEach(function (r, i) {
        var q = r[1], p = typeof q === 'number' ? q : Array.isArray(q) ? q[0] : q.price != null ? q.price : q.p;
        var ry = y + 104 + i * 30;
        text(r[0], cols[0], ry, 10, r[0] === 'buy' ? C.buy : C.sell, 500, 'left', true);
        text(fmtPx(p), cols[1], ry, 10, C.body, 400, 'left', true);
        text(s.arm || '-', cols[2], ry, 10, C.accentText, 400, 'left');
        text(String(s.fills == null ? '-' : s.fills), cols[3], ry, 10, C.body, 400, 'left', true);
        text(s.inv == null ? '-' : String(+Number(s.inv).toPrecision(5)), cols[4], ry, 10, C.body, 400, 'left', true);
      });
    }

    /* Event contracts: the window being watched (last 60 s), open paper positions, recent settlements. */
    function drawEC(x, y, w, h, s) {
      panel(x, y, w, h);
      var e = s.ec, L = e.last || {}, pad = 18;
      text('up / down windows', x + pad, y + 24, 14, C.heading, 600);
      text((e.tracked || 0) + ' tracked, books read in the last 60 s', x + w - 14, y + 24, 9, C.muted, 400, 'right');
      var ty = y + 64;
      if (!L.symbol) { text('waiting for a window to enter its last 25 s', x + pad, ty + 20, 12, C.muted, 400); }
      else {
        text(String(L.symbol).split('/')[0], x + pad, ty, 13, C.accentText, 600);
        text((L.interval || '') + '  ' + (L.secondsLeft != null ? Math.max(0, L.secondsLeft).toFixed(1) + ' s left' : ''), x + w - 14, ty, 11, C.muted, 400, 'right', true);
        var cells = [
          ['book P(up)', L.prob_up_book], ['model P(up)', L.prob_up_model], ['z', L.z], ['lead bps', L.lead_bps]
        ];
        var cw = (w - pad * 2) / cells.length;
        cells.forEach(function (c, i) {
          var v = c[1];
          text(c[0], x + pad + i * cw, ty + 34, 9, C.muted, 400);
          text(v == null || isNaN(v) ? '-' : Number(v).toFixed(c[0] === 'lead bps' ? 1 : 2), x + pad + i * cw, ty + 62, 17,
            c[0].indexOf('P(') >= 0 && v != null ? (v >= 0.5 ? C.buy : C.sell) : C.heading, 600, 'left', true);
        });
      }
      var oy = ty + 100;
      text('open paper positions', x + pad, oy, 10, C.muted, 400);
      var open = (e.open || []).filter(function (o) { return o.side; });
      if (!open.length) text('none: the fly is watching, not buying', x + pad, oy + 26, 11, C.muted, 400);
      open.slice(0, 3).forEach(function (o, i) {
        var ry = oy + 26 + i * 26;
        text(String(o.symbol).split('/')[0], x + pad, ry, 10, C.body, 500, 'left', true);
        text(o.arm, x + w * 0.4, ry, 10, C.accentText, 400);
        text(o.side, x + w * 0.62, ry, 10, o.side === 'up' ? C.buy : C.sell, 600, 'left', true);
        text(fmtPx(o.price) + ' x ' + Number(o.units).toFixed(1), x + w - 14, ry, 10, C.body, 400, 'right', true);
      });
      var ry0 = oy + 26 + Math.max(1, Math.min(3, open.length)) * 26 + 16;
      text('recent settlements', x + pad, ry0, 10, C.muted, 400);
      var rec = (e.recent || []).slice().reverse(), maxRows = Math.max(0, Math.floor((y + h - ry0 - 30) / 26));
      if (!rec.length) text('none settled yet', x + pad, ry0 + 26, 11, C.muted, 400);
      rec.slice(0, maxRows).forEach(function (r, i) {
        var ry = ry0 + 26 + i * 26, pc = r.pnl > 0 ? C.buy : r.pnl < 0 ? C.sell : C.muted;
        text(String(r.symbol).split('/')[0], x + pad, ry, 10, C.body, 500, 'left', true);
        text(r.arm + (r.side ? ' ' + r.side : ''), x + w * 0.4, ry, 10, C.accentText, 400);
        text('closed ' + (r.truth || 'void'), x + w * 0.62, ry, 10, C.muted, 400);
        text(fmtMoney(r.pnl, true), x + w - 14, ry, 10, pc, 600, 'right', true);
      });
    }

    function draw(s) {
      X.fillStyle = C.page; X.fillRect(0, 0, W, H);
      // header: a nav row (market tabs, paper pill) and, as in the app, a market row with the pair and its stats
      var navH = compact ? 0 : 46, banH = compact ? 0 : 34, mkH = 62, hh = navH + banH + mkH;
      X.fillStyle = C.surface; X.fillRect(0, 0, W, hh);
      X.fillStyle = C.line; X.fillRect(0, hh - 1, W, 1);
      function pill(px, py, label, w2) {
        rr(px, X, py, 0, w2, 32, 16); X.fillStyle = 'rgba(110,110,237,0.25)'; X.fill();
        text(label, px + w2 / 2, py + 16, 11, C.accentText, 600, 'center');
      }
      if (!compact) {
        var lw0 = logo(22, 8, 30, true), tx = 22 + lw0 + 44;
        s.symbols.forEach(function (m) {
          var on = m === s.symbol, name = m === 'EC' ? 'up/down' : m.split(/[:\/]/)[0] + (m.indexOf('/') > 0 ? ' perp' : '');
          text(name, tx, navH / 2, 11, on ? C.heading : C.muted, on ? 600 : 400);
          var tw = X.measureText(name).width;
          if (on) { X.fillStyle = C.accent; X.fillRect(tx, navH - 3, tw, 3); }
          tx += tw + 34;
        });
        pill(W - 132, 7, 'paper', 96);
        text(s.network || '', W - 148, navH / 2, 10, C.muted, 400, 'right');
        X.fillStyle = C.line; X.fillRect(0, navH - 1, W, 1);
        // announcement banner, as on the app
        X.fillStyle = '#7373e6'; X.fillRect(0, navH, W, banH);
        text('Paper trading on dreamDEX: a fruit fly picks the strategy. Nothing is sent to the exchange.', W / 2, navH + banH / 2, 12, '#fff', 500, 'center');
      }
      var label = s.kind === 'perp' ? s.symbol.split('/')[0] + '-PERP' : s.kind === 'ec' ? 'BTC / ETH up or down' : s.symbol.replace(':', ' / ');
      var my = navH + banH + mkH / 2;
      text(label, 34, my, 15, C.heading, 600);
      var lw = X.measureText(label).width;
      rr(18, X, navH + banH + 10, 0, lw + 32, mkH - 20, 8); X.strokeStyle = C.frame; X.lineWidth = 1; X.stroke();
      if (compact) { pill(W - 112, navH + banH + 15, 'paper', 96); }
      var cs = s.candles || [];
      if (s.kind !== 'ec' && cs.length) {
        var f0 = cs[0], l0 = cs[cs.length - 1], hi = -Infinity, lo = Infinity, vol = 0;
        cs.forEach(function (c) { hi = Math.max(hi, c.h); lo = Math.min(lo, c.l); vol += c.v || 0; });
        var chg = f0.o ? (l0.c - f0.o) / f0.o * 100 : 0, per = cs.length >= 55 ? '1H' : cs.length + 'M';
        var mid = s.touch ? (s.touch[0] + s.touch[1]) / 2 : l0.c;
        var sx = 18 + lw + 32 + 34;
        var stats = [['MID', fmtPx(mid), C.heading], [per + ' CHANGE', (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%', chg >= 0 ? C.buy : C.sell],
          [per + ' HIGH', fmtPx(hi), C.buy], [per + ' LOW', fmtPx(lo), C.sell], [per + ' VOL', vol >= 1000 ? (vol / 1000).toFixed(1) + 'K' : vol.toFixed(1), C.heading]];
        if (compact) stats = stats.slice(0, 3);   // the tablet and phone canvas is narrow: three stats fit without touching
        var sp = Math.min(150, (W - sx - (compact ? 130 : 30)) / stats.length);
        stats.forEach(function (st, i) {
          text(st[0], sx + i * sp, my - 11, 8.5, C.muted, 400);
          text(st[1], sx + i * sp, my + 9, 11, st[2], 500, 'left', true);
        });
      }
      var footH = compact ? 0 : 36, g = 8, top = hh + g, bh = H - top - g - footH;
      if (!compact) {   // footer, as on the app
        X.fillStyle = C.line; X.fillRect(0, H - footH, W, 1);
        var fw = logo(22, H - footH + 8, 20, false);
        text('\u00a9 2026 - DreamDEX, S.A.', 22 + fw + 12, H - footH / 2, 10, C.muted, 400);
        var links = ['Support', 'Privacy policy', 'Terms'], rx = W - 22;
        links.forEach(function (t) {
          text(t, rx, H - footH / 2, 10, C.muted, 400, 'right');
          var tw = X.measureText(t).width;
          X.fillStyle = C.muted; X.fillRect(rx - tw, H - footH / 2 + 10, tw, 1);
          rx -= tw + 28;
        });
      }
      if (compact) {
        var chH = 300, mH = 372, bwc = Math.round(W * 0.47), fwc = W - bwc - g * 3, oy = top + chH + g + mH + g;
        if (s.ec) drawEC(g, top, W - g * 2, chH + g + mH, s);
        else {
          drawChart(g, top, W - g * 2, chH, s.candles, s.candleSrc);
          drawBook(g, top + chH + g, bwc, mH, s);
        }
        if (s.ec) drawForm(g, oy, W - g * 2, H - oy - g, s);
        else { drawForm(g * 2 + bwc, top + chH + g, fwc, mH, s); drawOrders(g, oy, W - g * 2, H - oy - g, s); }
        return;
      }
      var lowH = Math.round(bh * 0.24), upH = bh - lowH - g;
      var cw = Math.round(W * 0.56), bw = Math.round(W * 0.21), fw = W - cw - bw - g * 4;
      if (s.ec) drawEC(g, top, cw + bw + g, bh, s);
      else {
        drawChart(g, top, cw, upH, s.candles, s.candleSrc);
        drawBook(g * 2 + cw, top, bw, upH, s);
        drawOrders(g, top + upH + g, cw + bw + g, lowH, s);
      }
      drawForm(g * 3 + cw + bw, top, fw, bh, s);
    }
    return { draw: draw };
  }

  root.FlyTerminal = { create: create };
})(window);
