/* flybot brain renderer: merged LineSegments per neuron group + flash/floor animation.
   Shared by brain.html (research view) and index.html (picture-in-picture).
   Usage: var b = FlyBrain.create(canvas, {pixelRatio, kcScale, autoRotate, interactive});
          b.load('data/brain.json').then(...);  b.kick(...); b.flashKC(list); ... */
(function (root) {
  'use strict';

  var BASE = [
    [0.002, 0.018, 0.017],// PN teal (dimmed: 150 dense cells saturated to white)
    [0.006, 0.010, 0.022],// KC dim blue-grey
    [0.03, 0.025, 0.017], // MBON warm white (dimmed: 49 dense cells saturated to white)
    [0.006, 0.035, 0.015],  // PAM green (dimmed: 158 dense cells saturated to white)
    [0.08, 0.015, 0.012],  // PPL1 red
    [0.25, 0.15, 0.45],   // APL violet
    [0.25, 0.15, 0.45]    // DPM violet
  ];
  var FLASH = {
    kc: [0.05, 0.14, 0.36], mbon: [0.5, 0.36, 0.16], pam: [0.15, 0.6, 0.25], ppl1: [0.6, 0.15, 0.12]
  };
  var DECAY = 1.6; // per second; ~3 s to visually vanish

  function create(canvas, opts) {
    opts = opts || {};
    var kcScale = opts.kcScale == null ? 0.2 : opts.kcScale; // tones down the Kenyon-cell lobes
    var pixelRatio = opts.pixelRatio || Math.min(window.devicePixelRatio || 1, 2);
    var reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: opts.antialias !== false });
    renderer.setPixelRatio(pixelRatio);
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(opts.background == null ? 0x04060c : opts.background);
    var camera = new THREE.PerspectiveCamera(42, 1, 1, 6000);
    var groups = [], ready = false, running = true, lastT = performance.now();
    var api = {};

    function resize() {
      var p = canvas.parentElement, w = p.clientWidth, h = p.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    new ResizeObserver(resize).observe(canvas.parentElement);

    // minimal orbit control (drag / wheel / pinch); OrbitControls is not on cdnjs
    var orbit = { theta: 0.6, phi: 1.25, dist: 500, minD: 80, maxD: 1400, lastTouch: -1e9 };
    if (opts.view === 'front') {
      // anterior view, as in the classic whole-brain pictures: optic lobes left and right, dorsal up.
      // EM coordinates run y downward (ventral) and z toward the back, so look along +z with y flipped.
      orbit.theta = Math.PI; orbit.phi = Math.PI / 2;
      camera.up.set(0, -1, 0);
    }
    var spin = opts.spin == null ? 0.12 : opts.spin;
    function zoom(f) { orbit.dist = Math.min(orbit.maxD, Math.max(orbit.minD, orbit.dist * f)); }
    if (opts.interactive !== false) {
      var pts = {}, pinch = 0;
      canvas.addEventListener('pointerdown', function (e) {
        canvas.setPointerCapture(e.pointerId);
        pts[e.pointerId] = [e.clientX, e.clientY];
        orbit.lastTouch = performance.now();
      });
      canvas.addEventListener('pointerup', function (e) { delete pts[e.pointerId]; pinch = 0; });
      canvas.addEventListener('pointercancel', function (e) { delete pts[e.pointerId]; pinch = 0; });
      canvas.addEventListener('pointermove', function (e) {
        var prev = pts[e.pointerId];
        if (!prev) return;
        var ids = Object.keys(pts);
        orbit.lastTouch = performance.now();
        if (ids.length === 2) {
          pts[e.pointerId] = [e.clientX, e.clientY];
          var a = pts[ids[0]], b = pts[ids[1]];
          var d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (pinch) zoom(pinch / d);
          pinch = d;
          return;
        }
        orbit.theta -= (e.clientX - prev[0]) * 0.006;
        orbit.phi = Math.min(3.0, Math.max(0.15, orbit.phi - (e.clientY - prev[1]) * 0.006));
        pts[e.pointerId] = [e.clientX, e.clientY];
      });
      if (opts.wheel !== false) {
        canvas.addEventListener('wheel', function (e) {
          e.preventDefault();
          orbit.lastTouch = performance.now();
          zoom(Math.exp(e.deltaY * 0.001));
        }, { passive: false });
      }
    }
    function placeCamera() {
      var s = Math.sin(orbit.phi);
      camera.position.set(orbit.dist * s * Math.sin(orbit.theta), orbit.dist * Math.cos(orbit.phi),
        orbit.dist * s * Math.cos(orbit.theta));
      camera.lookAt(0, 0, 0);
    }

    /* Kenyon cells are grouped into about forty regions of the lobes and calyx. A sniff or decision lights the
       regions by the share of their cells that fire, as soft glows; single cells are not shown. */
    var SPREAD = 3.6, lineLayers = [];
    var reg = { pos: null, n: 0, of: {}, size: [], E: null, floor: null, col: null, pts: null };
    function buildRegions(list) {
      var c = list.map(function (n) {
        var sx = 0, sy = 0, sz = 0, m = 0;
        n.seg.forEach(function (sg) { for (var k = 0; k + 2 < sg.length; k += 3) { sx += sg[k]; sy += sg[k + 1]; sz += sg[k + 2]; m++; } });
        return m ? [sx / m, sy / m, sz / m] : null;
      });
      var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
      c.forEach(function (p) { if (p) for (var a = 0; a < 3; a++) { if (p[a] < mn[a]) mn[a] = p[a]; if (p[a] > mx[a]) mx[a] = p[a]; } });
      var N = 4, cell = {};
      list.forEach(function (n, i) {
        var p = c[i]; if (!p) return;
        var key = [0, 1, 2].map(function (a) { return Math.min(N - 1, Math.floor((p[a] - mn[a]) / (mx[a] - mn[a] + 1e-9) * N)); }).join(',');
        (cell[key] || (cell[key] = { sum: [0, 0, 0], m: 0, ids: [] }));
        for (var a = 0; a < 3; a++) cell[key].sum[a] += p[a];
        cell[key].m++; cell[key].ids.push(n.i);
      });
      var keys = Object.keys(cell).filter(function (k) { return cell[k].m >= 6; });
      reg.n = keys.length; reg.pos = new Float32Array(reg.n * 3); reg.of = {}; reg.size = [];
      keys.forEach(function (k, r) {
        for (var a = 0; a < 3; a++) reg.pos[r * 3 + a] = cell[k].sum[a] / cell[k].m;
        cell[k].ids.forEach(function (id) { reg.of[id] = r; });
        reg.size.push(cell[k].m);
      });
      var cm = [0, 0, 0];   // spread the glows out from their centre (a schematic map of the areas, not exact positions) so they read as separate areas
      for (var q0 = 0; q0 < reg.n; q0++) for (var a0 = 0; a0 < 3; a0++) cm[a0] += reg.pos[q0 * 3 + a0] / reg.n;
      for (var q1 = 0; q1 < reg.n; q1++) for (var a1 = 0; a1 < 3; a1++) reg.pos[q1 * 3 + a1] = cm[a1] + (reg.pos[q1 * 3 + a1] - cm[a1]) * SPREAD;
      reg.cm = cm; reg.nkc = reg.n;
      var xs = [], lo = Infinity, hi = -Infinity;
      for (var r0 = 0; r0 < reg.n; r0++) { var x0 = reg.pos[r0 * 3]; xs.push(x0); if (x0 < lo) lo = x0; if (x0 > hi) hi = x0; }
      reg.t = xs.map(function (x) { return (x - lo) / (hi - lo + 1e-9); });   // 0..1 along the lobes, for the sweep
      reg.rgb = new Float32Array(reg.n * 3);
      reg.t.forEach(function (t, r) {   // blue -> violet -> cyan across the regions so neighbours are told apart
        var c = t < 0.5 ? [0.2 + 0.5 * t * 2, 0.5 - 0.15 * t * 2, 1.0] : [0.7 - 0.5 * (t - 0.5) * 2, 0.35 + 0.55 * (t - 0.5) * 2, 1.0 - 0.1 * (t - 0.5) * 2];
        for (var a = 0; a < 3; a++) reg.rgb[r * 3 + a] = c[a];
      });
      reg.E = new Float32Array(reg.n); reg.floor = new Float32Array(reg.n); reg.col = new Float32Array(reg.n * 3);
    }
    function glowTexture() {
      var cv = document.createElement('canvas'); cv.width = cv.height = 64;
      var x = cv.getContext('2d'), g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
      g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.3, 'rgba(255,255,255,0.35)'); g.addColorStop(1, 'rgba(255,255,255,0)');
      x.fillStyle = g; x.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(cv);
    }
    function centroidOf(list) {
      var sm = [0, 0, 0], m = 0;
      list.forEach(function (n) { n.seg.forEach(function (sg) { for (var k = 0; k + 2 < sg.length; k += 3) { sm[0] += sg[k]; sm[1] += sg[k + 1]; sm[2] += sg[k + 2]; m++; } }); });
      return m ? [sm[0] / m, sm[1] / m, sm[2] / m] : null;
    }
    /* three more glows: the output pool (orange), the reward dopamine cells (green), the punishment ones (red) */
    function addExtras(byG) {
      var defs = [['mbon', 2, [1.0, 0.62, 0.2]], ['pam', 3, [0.3, 1.0, 0.45]], ['ppl1', 4, [1.0, 0.3, 0.25]]];
      var pos = Array.prototype.slice.call(reg.pos), rgb = Array.prototype.slice.call(reg.rgb), t = reg.t.slice();
      reg.extra = {};
      defs.forEach(function (d) {
        var c = centroidOf(byG[d[1]] || []); if (!c) return;
        reg.extra[d[0]] = pos.length / 3;
        for (var a = 0; a < 3; a++) pos.push(reg.cm[a] + (c[a] - reg.cm[a]) * SPREAD);
        rgb.push(d[2][0], d[2][1], d[2][2]); t.push(0.5); reg.size.push(1);
      });
      reg.n = pos.length / 3; reg.pos = new Float32Array(pos); reg.rgb = new Float32Array(rgb); reg.t = t;
      reg.E = new Float32Array(reg.n); reg.floor = new Float32Array(reg.n);
    }
    function lightExtra(name, e, fl) {
      var i = reg.extra && reg.extra[name]; if (i === undefined) return;
      reg.E[i] = Math.max(reg.E[i], e); reg.floor[i] = fl;
    }
    /* called once both files are in: the model is one hemisphere, so the glows are mirrored across the brain's
       midline for display (a fly has two mushroom bodies, mirror images of each other) */
    function finishRegions(midX) {
      if (!reg.n || reg.pts) return;
      var n = reg.n, mirror = midX != null, total = mirror ? n * 2 : n, pos = new Float32Array(total * 3);
      pos.set(reg.pos);
      if (mirror) for (var r = 0; r < n; r++) { pos[(n + r) * 3] = 2 * midX - reg.pos[r * 3]; pos[(n + r) * 3 + 1] = reg.pos[r * 3 + 1]; pos[(n + r) * 3 + 2] = reg.pos[r * 3 + 2]; }
      reg.mirror = mirror; reg.col = new Float32Array(total * 3);
      var geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      reg.colAttr = new THREE.BufferAttribute(reg.col, 3); geo.setAttribute('color', reg.colAttr);
      reg.pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 30, sizeAttenuation: false, map: glowTexture(), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false }));
      reg.pts.renderOrder = 6; scene.add(reg.pts);
    }
    var REG_COL = [0.22, 0.5, 1.0];
    function paintRegions() {
      if (!reg.colAttr) return;
      var n = reg.n;
      for (var r = 0; r < n; r++) for (var a = 0; a < 3; a++) {
        var v = reg.rgb[r * 3 + a] * 0.9 * Math.min(1, reg.E[r]);
        reg.col[r * 3 + a] = v; if (reg.mirror) reg.col[(n + r) * 3 + a] = v;
      }
      reg.colAttr.needsUpdate = true;
    }
    function lightRegions(kc, strength, floorK) {
      if (!reg.n) return;
      var cnt = new Float32Array(reg.nkc);
      kc.forEach(function (id) { var r = reg.of[id]; if (r !== undefined) cnt[r]++; });
      var fr = [], top = 1e-6;
      for (var r = 0; r < reg.nkc; r++) { fr[r] = cnt[r] / reg.size[r]; if (fr[r] > top) top = fr[r]; }
      for (var r2 = 0; r2 < reg.nkc; r2++) {
        var frac = Math.pow(fr[r2] / top, 0.8);   // relative to the busiest region, so the pattern is always readable
        reg.floor[r2] = frac * floorK;
        reg.E[r2] = Math.max(reg.E[r2], frac * strength);
      }
    }

    function build(data) {
      var byG = [];
      for (var g = 0; g < data.groups.length; g++) byG.push([]);
      data.neurons.forEach(function (n) { byG[n.g].push(n); });
      var maxR = 1, kmn = [Infinity, Infinity, Infinity], kmx = [-Infinity, -Infinity, -Infinity];
      byG.forEach(function (list, g) {
        if (g === 1) { buildRegions(list); list = list.filter(function (n) { return n.i % 3 === 0; }); }  // a third of the wires: context, not detail
        var nv = 0;
        list.forEach(function (n) { n.seg.forEach(function (s) { nv += Math.max(0, s.length / 3 - 1) * 2; }); });
        var pos = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
        var v = 0, ranges = [];
        var base = (BASE[g] || BASE[1]).slice();
        if (g === 1) base = base.map(function (x) { return x * kcScale; });
        list.forEach(function (n) {
          var start = v;
          n.seg.forEach(function (s) {
            for (var k = 0; k + 5 < s.length; k += 3) {
              for (var e = 0; e < 2; e++) {
                var x = s[k + e * 3], y = s[k + e * 3 + 1], z = s[k + e * 3 + 2];
                pos[v * 3] = x; pos[v * 3 + 1] = y; pos[v * 3 + 2] = z;
                col[v * 3] = base[0]; col[v * 3 + 1] = base[1]; col[v * 3 + 2] = base[2];
                if (g === 1) { if (x < kmn[0]) kmn[0] = x; if (x > kmx[0]) kmx[0] = x; if (y < kmn[1]) kmn[1] = y; if (y > kmx[1]) kmx[1] = y; if (z < kmn[2]) kmn[2] = z; if (z > kmx[2]) kmx[2] = z; }
                var r = Math.sqrt(x * x + y * y + z * z);
                if (r > maxR) maxR = r;
                v++;
              }
            }
          });
          ranges[n.i] = [start, v - start];
        });
        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        var colAttr = new THREE.BufferAttribute(col, 3);
        geo.setAttribute('color', colAttr);
        var mat = new THREE.LineBasicMaterial({
          vertexColors: true, transparent: true, opacity: 1,
          blending: THREE.AdditiveBlending, depthWrite: false  // opaque dark lines would blot out the structures behind them
        });
        var lines = new THREE.LineSegments(geo, mat); lineLayers.push(lines); lines.visible = !!opts.wires; lines.renderOrder = g === 1 ? 5 : 1;  // wires are not drawn: the panel shows areas  // firing cells draw on top of the output and dopamine cells
        scene.add(lines);

        groups[g] = {
          col: colAttr, ranges: ranges, base: base,
          E: new Float32Array(ranges.length), floor: new Float32Array(ranges.length),
          flash: FLASH.kc, hot: new Set(), dirty: false
        };
      });
      groups[1].flash = FLASH.kc; groups[2].flash = FLASH.mbon;
      groups[3].flash = FLASH.pam; groups[4].flash = FLASH.ppl1;
      if (!orbit.fitted) { orbit.dist = maxR * (opts.distScale || 2.35); orbit.minD = maxR * 0.5; orbit.maxD = maxR * 6; }
      addExtras(byG);
      mb = { c: [(kmn[0] + kmx[0]) / 2, (kmn[1] + kmx[1]) / 2, (kmn[2] + kmx[2]) / 2],
             r: 0.5 * Math.sqrt(Math.pow(kmx[0] - kmn[0], 2) + Math.pow(kmx[1] - kmn[1], 2) + Math.pow(kmx[2] - kmn[2], 2)) };
      ready = true;
    }

    function paint(g, slot) {
      var G = groups[g], r = G.ranges[slot];
      if (!r) return;
      var e = G.E[slot], f = G.flash, b = G.base, a = G.col.array;
      var R = b[0] + f[0] * e, Gc = b[1] + f[1] * e, B = b[2] + f[2] * e;
      for (var v = r[0], end = r[0] + r[1]; v < end; v++) {
        a[v * 3] = R; a[v * 3 + 1] = Gc; a[v * 3 + 2] = B;
      }
      G.dirty = true;
    }
    function kick(g, slots, e0, colour) {
      var G = groups[g];
      if (!G) return;
      if (colour) G.flash = colour;
      slots.forEach(function (s) {
        if (!G.ranges[s]) return;
        G.E[s] = Math.max(G.E[s], e0);
        G.hot.add(s);
        paint(g, s);
      });
    }
    function setFloor(g, slots, val) {
      var G = groups[g], keep = new Set(slots);
      for (var s = 0; s < G.floor.length; s++) {
        if (G.floor[s] > 0 && !keep.has(s)) { G.floor[s] = 0; G.hot.add(s); }
      }
      slots.forEach(function (s) { if (G.ranges[s]) { G.floor[s] = val; G.hot.add(s); } });
    }
    function stepFlashes(dt) {
      var k = Math.exp(-DECAY * dt);
      if (reg.n) { for (var r = 0; r < reg.n; r++) reg.E[r] = reg.floor[r] + (reg.E[r] - reg.floor[r]) * k; paintRegions(); }
      groups.forEach(function (G, g) {
        if (!G || !G.hot.size) return;
        G.hot.forEach(function (s) {
          var f = G.floor[s], e = f + (G.E[s] - f) * k;
          if (e - f < 0.01) { e = f; if (!f) G.hot.delete(s); }
          G.E[s] = e;
          paint(g, s);
        });
        if (G.dirty) { G.col.needsUpdate = true; G.dirty = false; }
      });
    }

    var active = [], shimmerAt = 0, mb = null;
    function frame(t) {
      requestAnimationFrame(frame);
      var dt = Math.min(0.1, (t - lastT) / 1000); lastT = t;
      if (!ready || !running || document.hidden) return;
      if (!reduceMotion && spin > 0 && t - orbit.lastTouch > 3000) orbit.theta += dt * spin;
      if (!reduceMotion && reg.n && t - shimmerAt > 120) {
        shimmerAt = t;
        var wave = (t / 2600) % 1;   // a slow sweep along the lobes so different areas pulse in turn
        for (var q = 0; q < reg.nkc; q++) {
          if (reg.floor[q] < 0.02) continue;
          var near = Math.abs(reg.t[q] - wave); near = Math.min(near, 1 - near);
          if (near < 0.12) reg.E[q] = Math.min(1, reg.E[q] + 0.5 * Math.min(1, reg.floor[q] * 3));
          else if (Math.random() < 0.15) reg.E[q] = Math.min(1, reg.E[q] + 0.35 * Math.min(1, reg.floor[q] * 3));
        }
      }
      stepFlashes(dt);
      placeCamera();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(frame);

    function all(g) {
      var out = [];
      for (var i = 0; i < groups[g].ranges.length; i++) if (groups[g].ranges[i]) out.push(i);
      return out;
    }

    api.load = function (url, onProgress) {
      return fetchJSON(url, onProgress).then(function (d) { build(d); });
    };
    /* Whole-brain context: every cell body as a dim point, optic lobes amber and central brain blue,
       so the lit mushroom body reads as a circuit inside the recognisable fly-brain silhouette. */
    api.loadSomas = function (url, o) {
      o = o || {};
      return fetchJSON(url, o.onProgress).then(function (d) {
        var n = d.n, pos = new Float32Array(d.xyz), col = new Float32Array(n * 3);
        var ol = o.optic || [0.95, 0.62, 0.22], cb = o.central || [0.30, 0.52, 0.95], dim = o.dim == null ? 0.16 : o.dim;
        var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
        for (var i = 0; i < n; i++) {
          var c = d.kind[i] ? ol : cb;
          col[i * 3] = c[0] * dim; col[i * 3 + 1] = c[1] * dim; col[i * 3 + 2] = c[2] * dim;
          for (var a = 0; a < 3; a++) { var v = pos[i * 3 + a]; if (v < mn[a]) mn[a] = v; if (v > mx[a]) mx[a] = v; }
        }
        // centre the view on the whole brain, not on the mushroom body the data was centred on
        var mid = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
        api.somaMid = mid;
        scene.position.set(-mid[0], -mid[1], -mid[2]);
        var maxR = 0.5 * Math.sqrt(Math.pow(mx[0] - mn[0], 2) + Math.pow(mx[1] - mn[1], 2) + Math.pow(mx[2] - mn[2], 2));
        var geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
        var mat = new THREE.PointsMaterial({
          size: o.size || 1.6, sizeAttenuation: true, vertexColors: true, transparent: true,
          opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false
        });
        scene.add(new THREE.Points(geo, mat));
        if (o.fit !== false) { orbit.dist = maxR * (o.distScale || 2.6); orbit.minD = maxR * 0.3; orbit.maxD = maxR * 5; orbit.fitted = true; }
      });
    };
    /* frame the mushroom body itself (Kenyon cells and lobes), so the firing is big enough to see; the cell
       bodies stay behind it as context */
    api.focusMB = function (k) {
      if (!mb) return;
      scene.position.set(-mb.c[0], -mb.c[1], -mb.c[2]);
      orbit.dist = mb.r * (k || 2.4); orbit.minD = mb.r * 0.6; orbit.fitted = true;
    };
    api.isReady = function () { return ready; };
    api.setRunning = function (v) { running = v; };
    api.groups = groups;
    api.kick = kick;
    api.setFloor = setFloor;
    api.FLASH = FLASH;
    // wires mode (the research page) draws every cell and flashes cells; the picture-in-picture shows areas only
    var WIRE_KC = [0.05, 0.12, 0.2];
    api.flashKC = function (kc) {
      if (!ready) return; active = kc; lightRegions(kc, 1.0, 0.55);
      if (opts.wires) { setFloor(1, kc, 0.22 * kcScale); kick(1, kc, 1.4 * kcScale, WIRE_KC); }
    };
    /* a sniff between decisions: the regions the market lights right now flash and keep a glow until the next
       sniff, and a slow shimmer runs over them so the brain is never dark */
    api.sniffKC = function (kc) {
      if (!ready) return; active = kc; lightRegions(kc, 0.9, 0.3);
      if (opts.wires) kick(1, kc, 1.2 * kcScale, WIRE_KC);
    };
    api.flashPool = function (slots) {
      if (!ready) return; lightExtra('mbon', 1.0, 0.3);
      if (opts.wires) { setFloor(2, slots, 0.1); kick(2, slots, 0.55, FLASH.mbon); }
    };
    api.flashDopamine = function (kind) {
      if (!ready) return;
      if (kind === 'PAM') lightExtra('pam', 1.0, 0); else if (kind === 'PPL1') lightExtra('ppl1', 1.0, 0);
      var g = kind === 'PAM' ? 3 : kind === 'PPL1' ? 4 : 0;
      if (opts.wires && g) kick(g, all(g), 1.6, g === 3 ? FLASH.pam : FLASH.ppl1);
    };
    /* the data is one hemisphere; the other is drawn as its mirror image across the brain's midline. The mirror
       shares the colour buffer, so it flashes together with the original. */
    function mirrorWires(midX) {
      if (midX == null) return;
      lineLayers.slice().forEach(function (L) {
        var p = L.geometry.attributes.position.array, q = new Float32Array(p.length);
        for (var i = 0; i < p.length; i += 3) { q[i] = 2 * midX - p[i]; q[i + 1] = p[i + 1]; q[i + 2] = p[i + 2]; }
        var g2 = new THREE.BufferGeometry();
        g2.setAttribute('position', new THREE.BufferAttribute(q, 3));
        g2.setAttribute('color', L.geometry.attributes.color);
        var M = new THREE.LineSegments(g2, L.material); M.renderOrder = L.renderOrder; scene.add(M);
      });
    }
    api.finishRegions = function () {
      var mx = api.somaMid ? api.somaMid[0] : null;
      if (opts.wires) mirrorWires(mx); else finishRegions(mx);
    };
    return api;
  }

  /* JSON fetch that reports progress as a fraction. The server sends X-Raw-Length (the size before
     gzip) so the bar tracks the bytes actually arriving; without it, or without streams, it resolves
     the plain way. The browser cache is allowed: the server marks these files revalidate-by-ETag. */
  function fetchJSON(url, onProgress) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      var total = +r.headers.get('X-Raw-Length') || (!r.headers.get('Content-Encoding') && +r.headers.get('Content-Length')) || 0;
      if (!onProgress || !r.body || !total) return r.json();
      var reader = r.body.getReader(), chunks = [], got = 0;
      return (function pump() {
        return reader.read().then(function (s) {
          if (s.done) {
            var buf = new Uint8Array(got), o = 0;
            chunks.forEach(function (c) { buf.set(c, o); o += c.length; });
            onProgress(1);
            return JSON.parse(new TextDecoder().decode(buf));
          }
          chunks.push(s.value); got += s.value.length; onProgress(Math.min(0.999, got / total));
          return pump();
        });
      })();
    });
  }

  root.FlyBrain = { create: create, fetchJSON: fetchJSON };
})(window);
