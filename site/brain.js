/* flybot research view (brain.html): 3D mushroom-body circuit + paper trading desk. No build step.
   The 3D renderer lives in brain-core.js (shared with the picture-in-picture on index.html). */
(function () {
  'use strict';

  var ARMS = ['maker', 'grid', 'trend', 'flat'];
  var ARM_COL = { maker: '#6fd0ff', grid: '#c9a6ff', trend: '#ffcf6b', flat: '#8b96ab' };
  var LEGEND = [
    ['Kenyon cells', '#5f86c9'], ['projection neurons', '#26d9c4'], ['output neurons', '#ffe9c7'],
    ['PAM (reward)', '#3ee56f'], ['PPL1 (punish)', '#ff4a3d'], ['APL / DPM', '#a674ff']
  ];
  var POLL_MS = 5000;

  var $ = function (id) { return document.getElementById(id); };

  /* ---------------------------------------------------------------- 3D */
  var brain = null, brainReady = false;

  /* ---------------------------------------------------------------- desk */
  var live = null, selected = null, episodes = {}, lastTs = {}, lastEp = {};

  function fmtPx(p) {
    if (p == null || isNaN(p)) return '-';
    return Number(p).toFixed(p >= 1000 ? 2 : p >= 1 ? 3 : 5);
  }
  function fmtMoney(n, sign) {
    if (n == null || isNaN(n)) return '-';
    var s = Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2);
    return sign && n > 0 ? '+' + s : s;
  }
  function fmtQuote(q) {
    if (q == null) return '';
    if (typeof q === 'number') return fmtPx(q);
    if (Array.isArray(q)) return fmtPx(q[0]) + (q[1] != null ? ' × ' + Number(q[1]).toPrecision(3) : '');
    var p = q.price != null ? q.price : q.p, z = q.size != null ? q.size : q.q;
    return fmtPx(p) + (z != null ? ' × ' + Number(z).toPrecision(3) : '');
  }
  function quotesText(a) { return a && a.length ? a.map(fmtQuote).join('\n') : 'none'; }
  function hhmm(ms) {
    if (!ms) return '';
    var d = new Date(ms);
    return ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2);
  }

  function buildTabs() {
    var tabs = $('tabs'), names = Object.keys(live.markets);
    if (tabs.dataset.sig === names.join()) return;
    tabs.dataset.sig = names.join();
    tabs.textContent = '';
    names.forEach(function (m) {
      var b = document.createElement('button');
      b.className = 'tab'; b.type = 'button'; b.setAttribute('role', 'tab');
      b.textContent = m.split(':')[0];
      b.dataset.m = m;
      b.onclick = function () { select(m); };
      tabs.appendChild(b);
    });
  }
  function select(m) {
    selected = m;
    Array.prototype.forEach.call($('tabs').children, function (b) {
      b.setAttribute('aria-selected', b.dataset.m === m ? 'true' : 'false');
    });
    fetchEpisodes(m).then(function () { renderDesk(); triggerEffects(true); });
    renderDesk();
    triggerEffects(true);
  }

  function armIndex(name) { return Math.max(0, ARMS.indexOf(name)); }

  function renderDesk() {
    if (!live || !selected) return;
    var mk = live.markets[selected]; if (!mk) return;
    var dec = mk.decision || {}, lv = mk.live, tc = (live.touch || {})[selected];
    $('net').textContent = live.network || '';
    $('kill').hidden = !live.kill;
    $('bid').textContent = tc ? fmtPx(tc[0]) : '-';
    $('ask').textContent = tc ? fmtPx(tc[1]) : '-';

    var arm = dec.arm || (lv && lv.arm) || '-';
    var an = $('arm'); an.textContent = arm; an.style.color = ARM_COL[arm] || 'inherit';
    var sub = [];
    if (dec.kc_frac != null) sub.push((dec.kc_frac * 100).toFixed(1) + '% of Kenyon cells fired');
    if (dec.guard_flat) sub.push('guard: flat');
    $('armsub').textContent = sub.join(', ');

    var sc = dec.scores || [0, 0, 0, 0], mn = Math.min.apply(null, sc), mx = Math.max.apply(null, sc);
    var box = $('scores'); box.textContent = '';
    ARMS.forEach(function (a, i) {
      var row = document.createElement('div'); row.className = 'row' + (a === arm ? ' sel' : '');
      var l = document.createElement('span'); l.className = 'lab'; l.textContent = a;
      var tr = document.createElement('div'); tr.className = 'track';
      var fl = document.createElement('div'); fl.className = 'fill';
      var w = mx > mn ? 4 + 96 * (sc[i] - mn) / (mx - mn) : 4;
      fl.style.width = w + '%'; fl.style.background = ARM_COL[a];
      tr.appendChild(fl);
      var v = document.createElement('span'); v.className = 'v'; v.textContent = (sc[i] || 0).toFixed(3);
      row.appendChild(l); row.appendChild(tr); row.appendChild(v);
      box.appendChild(row);
    });

    $('inv').textContent = lv ? String(+Number(lv.inv).toPrecision(5)) : '-';
    $('fills').textContent = lv ? lv.fills : '-';
    $('bids').textContent = lv ? quotesText(lv.bids) : '-';
    $('asks').textContent = lv ? quotesText(lv.asks) : '-';
    $('bids').style.whiteSpace = $('asks').style.whiteSpace = 'pre-line';

    $('eq').textContent = fmtMoney(mk.equity);
    var pe = $('pnl'); pe.textContent = fmtMoney(mk.day_pnl, true);
    pe.className = 'v big ' + (mk.day_pnl > 0 ? 'pos' : mk.day_pnl < 0 ? 'neg' : '');

    renderEpisodes();
  }

  function renderEpisodes() {
    var eps = episodes[selected] || [];
    drawSpark(eps);
    var ul = $('dopa'); ul.textContent = '';
    eps.slice(-10).reverse().forEach(function (e) {
      var li = document.createElement('li');
      var tg = document.createElement('span'); tg.className = 'tag ' + (e.dopamine || 'none'); tg.textContent = e.dopamine || 'none';
      var t = document.createElement('span'); t.className = 't'; t.textContent = hhmm(e.end || e.wall) + '  ' + (e.arm || '');
      var p = document.createElement('span'); p.className = 'p ' + (e.pnl > 0 ? 'pos' : e.pnl < 0 ? 'neg' : '');
      p.textContent = fmtMoney(e.pnl, true);
      li.appendChild(tg); li.appendChild(t); li.appendChild(p); ul.appendChild(li);
    });
    $('dopaempty').hidden = eps.length > 0;
    var last = eps[eps.length - 1], ch = $('changed');
    if (!last) { ch.textContent = 'Waiting for the first episode to end.'; return; }
    var wm = last.weights_moved;
    var moved = wm && typeof wm === 'object' ? Object.keys(wm).length : Number(wm) || 0;
    var d = last.dopamine || 'none';
    ch.textContent = d === 'none' || !moved
      ? 'Last episode (' + last.arm + '): no dopamine signal, no weights moved.'
      : 'Last episode (' + last.arm + '): ' + d + ' dopamine moved ' + moved +
        (typeof wm === 'object' ? ' weight groups' : ' synaptic weights') + '.';
  }

  function drawSpark(eps) {
    var c = $('spark'), dpr = window.devicePixelRatio || 1;
    var w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr; c.height = h * dpr;
    var x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, w, h);
    var ys = eps.map(function (e) { return e.equity; }).filter(function (v) { return typeof v === 'number'; });
    $('sparkempty').hidden = ys.length > 0; c.hidden = ys.length === 0;
    if (!ys.length) return;
    var mn = Math.min.apply(null, ys), mx = Math.max.apply(null, ys), pad = 6;
    if (mx === mn) { mx += 1e-9; mn -= 1e-9; }
    var X = function (i) { return ys.length > 1 ? pad + (w - 2 * pad) * i / (ys.length - 1) : w / 2; };
    var Y = function (v) { return h - pad - (h - 2 * pad) * (v - mn) / (mx - mn); };
    var up = ys[ys.length - 1] >= ys[0], col = up ? '#3ee56f' : '#ff4a3d';
    x.strokeStyle = '#1a2233'; x.beginPath(); x.moveTo(0, Y(ys[0])); x.lineTo(w, Y(ys[0])); x.stroke();
    x.strokeStyle = col; x.lineWidth = 1.6; x.beginPath();
    ys.forEach(function (v, i) { i ? x.lineTo(X(i), Y(v)) : x.moveTo(X(i), Y(v)); });
    x.stroke();
    x.fillStyle = col; x.beginPath(); x.arc(X(ys.length - 1), Y(ys[ys.length - 1]), 3, 0, 6.3); x.fill();
  }

  function fetchEpisodes(m) {
    return fetch('/episodes/' + m.replace(/[:]/g, '_') + '.jsonl', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.text() : ''; })
      .catch(function () { return ''; })
      .then(function (txt) {
        var out = [];
        txt.split('\n').forEach(function (ln) {
          if (!ln.trim()) return;
          try { out.push(JSON.parse(ln)); } catch (e) { /* skip partial line */ }
        });
        episodes[m] = out;
      });
  }

  function triggerEffects(force) {
    if (!brainReady || !live || !selected) return;
    var mk = live.markets[selected]; if (!mk) return;
    var dec = mk.decision;
    if (dec && (force || lastTs[selected] !== dec.ts)) {
      lastTs[selected] = dec.ts;
      brain.flashKC(dec.kc_active || []);
      var pools = mk.pools || [], names = mk.arms || ARMS;
      brain.flashPool(pools[Math.max(0, names.indexOf(dec.arm))] || []);
    }
    var eps = episodes[selected] || [], last = eps[eps.length - 1];
    if (last) {
      var seen = lastEp[selected];
      if (seen !== undefined && seen !== last.wall && last.dopamine !== 'none') brain.flashDopamine(last.dopamine);
      lastEp[selected] = last.wall;
    } else if (lastEp[selected] === undefined) lastEp[selected] = null;
  }

  function poll() {
    fetch('/live.json', { cache: 'no-store' })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) {
        live = d;
        buildTabs();
        var names = Object.keys(d.markets);
        var first = !selected;
        if (!selected || !d.markets[selected]) selected = names[0];
        Array.prototype.forEach.call($('tabs').children, function (b) {
          b.setAttribute('aria-selected', b.dataset.m === selected ? 'true' : 'false');
        });
        return fetchEpisodes(selected).then(function () {
          renderDesk();
          triggerEffects(first);
        });
      })
      .catch(function () { live = live || null; setAge(true); });
  }
  function setAge(failed) {
    var dot = $('dot'), age = $('age');
    if (!live) { age.textContent = failed ? 'no data yet' : 'connecting'; dot.className = 'dot'; return; }
    var s = Math.max(0, Math.round((Date.now() - live.wall) / 1000));
    dot.className = 'dot ' + (s > 150 ? 'stale' : 'live');
    age.textContent = s > 150 ? 'stale, ' + s + 's old' : 'live, ' + s + 's ago';
  }

  /* ---------------------------------------------------------------- boot */
  (function legend() {
    var l = $('legend');
    LEGEND.forEach(function (it) {
      var s = document.createElement('span'), i = document.createElement('i');
      i.style.color = it[1]; s.appendChild(i); s.appendChild(document.createTextNode(it[0])); l.appendChild(s);
    });
  })();

  try { brain = FlyBrain.create($('gl'), { spin: 0, view: 'front', wires: true, interactive: false, wheel: false }); } catch (e) { $('loading').textContent = 'WebGL is not available in this browser.'; console.error(e); return; }
  // both files before the reveal: the cell bodies recentre the camera, so the circuit alone would jump
  Promise.all([
    brain.load('data/brain.json'),
    brain.loadSomas('data/somas.json', { size: 1.6, dim: 0.09 }).catch(function (e) { console.warn('somas', e); })
  ]).then(function () {
    brain.finishRegions();   // draws the mirror image of the hemisphere
    brainReady = true;
    $('loading').classList.add('done');
    triggerEffects(true);
  }).catch(function (e) { $('loading').textContent = 'Could not load brain data.'; console.error(e); });
  poll();
  setInterval(poll, POLL_MS);
  setInterval(function () { setAge(false); }, 1000);
  window.addEventListener('resize', function () { if (live) drawSpark(episodes[selected] || []); });
})();
