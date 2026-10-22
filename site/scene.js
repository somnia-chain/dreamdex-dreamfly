/* DreamFly main page: a low-poly fly at a desk in space, trading on a paper terminal.
   Procedural three.js scene + live data from /live.json. No build step. */
(function () {
  'use strict';

  var ARMS = ['maker', 'grid', 'trend', 'flat'];
  var ARM_COL = { maker: '#8a8aff', grid: '#c9a6ff', trend: '#ffcf6b', flat: '#999999' };
  var POLL_MS = 5000, TERM_MS = 1000, CANDLE_MS = 15000;
  var N_KC = 2044, RASTER_ROWS = 40;
  var $ = function (id) { return document.getElementById(id); };
  var reduceMotion = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ------------------------------------------------------------ state */
  var live = null, selected = null;
  var episodes = {}, lastEpWall = {}, lastDecTs = {};
  var decHist = {};          // sym -> [{ts, kc:[...]}] last RASTER_ROWS decisions seen by this page
  var touchHist = {};        // sym -> [[t, bid, ask]]
  var candles = {};          // sym -> {list, src, at}
  var dopamine = { kind: 'none', t: -1e9 };
  var tapAt = -1e9, typeAt = -1e9, sniffAt = -1e9;

  /* ------------------------------------------------------------ helpers */
  function hms(ms) {
    if (!ms) return '--:--:--';
    var d = new Date(ms), p = function (n) { return ('0' + n).slice(-2); };
    return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
  }
  function num(x) { return Number(x); }

  /* ------------------------------------------------------------ 3D scene */
  var canvas = $('scene');
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(36, 1, 0.1, 200);

  var ACC = 0x6e6eed, ACC_L = 0x8a8aff, ACC_G = 0xa3a3ff;
  function std(color, extra) {
    var m = { color: color, flatShading: true, roughness: 0.75, metalness: 0.1 };
    for (var k in extra) m[k] = extra[k];
    return new THREE.MeshStandardMaterial(m);
  }
  function box(w, h, d, mat, x, y, z, parent) {
    var m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z); (parent || scene).add(m); return m;
  }
  function edges(mesh, color, op) {
    var l = new THREE.LineSegments(new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: op == null ? 0.9 : op }));
    mesh.add(l); return l;
  }

  scene.add(new THREE.HemisphereLight(0x8088b8, 0x10101c, 0.55));
  var key = new THREE.DirectionalLight(0xdfe0ff, 0.9); key.position.set(-3, 5, 4); scene.add(key);
  var screenLight = new THREE.PointLight(ACC_L, 1.4, 9, 1.6); screenLight.position.set(0.5, 1.5, -0.4); scene.add(screenLight);
  var dopaLight = new THREE.PointLight(0x00fa89, 0, 7, 1.5); dopaLight.position.set(0, 1.6, 1.6); scene.add(dopaLight);
  var deskGlow = new THREE.PointLight(ACC, 1.2, 6, 1.5); deskGlow.position.set(0, -0.6, 1); scene.add(deskGlow);

  // --- space: stars + a low-poly planet + faint floor grid
  (function space() {
    var n = 1400, pos = new Float32Array(n * 3), col = new Float32Array(n * 3), c = new THREE.Color();
    for (var i = 0; i < n; i++) {
      var r = 40 + Math.random() * 50, th = Math.random() * 6.283, ph = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(ph) * Math.cos(th); pos[i * 3 + 1] = r * Math.cos(ph) * 0.8 + 6; pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      c.setHex(Math.random() < 0.25 ? ACC_G : 0xf5f5f5).multiplyScalar(0.35 + Math.random() * 0.65);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    var stars = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.16, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.9 }));
    stars.name = 'stars'; scene.add(stars);
    var planet = new THREE.Mesh(new THREE.IcosahedronGeometry(5, 1), std(0x2a2a5c, { emissive: 0x12123a, emissiveIntensity: 0.6 }));
    planet.position.set(-30, -0.5, -20); planet.scale.setScalar(0.6); planet.name = 'planet'; scene.add(planet);  // sits in the empty space under the title
    var ring = new THREE.Mesh(new THREE.RingGeometry(7, 9.5, 48), new THREE.MeshBasicMaterial({ color: ACC_L, transparent: true, opacity: 0.12, side: THREE.DoubleSide }));
    ring.position.copy(planet.position); ring.rotation.set(1.25, 0.2, 0.3); ring.scale.setScalar(0.6); scene.add(ring);
    var grid = new THREE.GridHelper(60, 40, ACC, ACC); grid.position.y = -2.4;
    grid.material.transparent = true; grid.material.opacity = 0.14; scene.add(grid);
  })();

  // --- desk
  var deskMat = std(0x16162a, { roughness: 0.9 });
  var desk = box(7.2, 0.2, 4.2, deskMat, 0, -0.1, 0.3);
  edges(desk, ACC_L, 0.95);
  [[-3.3, -1.7], [3.3, -1.7], [-3.3, 2.1], [3.3, 2.1]].forEach(function (p) {
    var lg = box(0.18, 2.2, 0.18, deskMat, p[0], -1.3, p[1]); edges(lg, ACC, 0.5);
  });
  // glowing front edge strip (edge-lit desk)
  box(7.2, 0.035, 0.035, new THREE.MeshBasicMaterial({ color: ACC_G }), 0, -0.02, 2.4);
  box(0.035, 0.035, 4.2, new THREE.MeshBasicMaterial({ color: ACC_L }), -3.6, -0.02, 0.3);
  box(0.035, 0.035, 4.2, new THREE.MeshBasicMaterial({ color: ACC_L }), 3.6, -0.02, 0.3);

  // --- monitor: curved screen showing the paper terminal
  var TW = 1600, TH = 900;
  var termCanvas = document.createElement('canvas'); termCanvas.width = TW; termCanvas.height = TH;
  var termTex = new THREE.CanvasTexture(termCanvas); termTex.minFilter = THREE.LinearFilter; termTex.anisotropy = 4;
  var terminal = FlyTerminal.create(termCanvas);
  (function monitor() {
    var g = new THREE.Group(); g.position.set(0, 0, -1.45); g.name = 'monitor'; scene.add(g);
    var SW = 4.3, SH = SW * TH / TW, CURVE = 0.03;
    function curved(w, h, zoff) {
      // a gently curved panel: edges come toward the viewer, same curve for screen and bezel so the
      // bezel always frames the picture whatever the camera angle
      var geo = new THREE.PlaneGeometry(w, h, 40, 1), p = geo.attributes.position;
      for (var i = 0; i < p.count; i++) { var x = p.getX(i); p.setZ(i, x * x * CURVE + zoff); }
      geo.computeVertexNormals();
      return geo;
    }
    var scr = new THREE.Mesh(curved(SW, SH, 0), new THREE.MeshBasicMaterial({ map: termTex }));
    scr.position.set(0, 2.0, 0.02); g.add(scr);
    var bezel = new THREE.Mesh(curved(SW + 0.18, SH + 0.18, -0.02), std(0x0c0c16, { roughness: 0.5 }));
    bezel.position.set(0, 2.0, 0.0); g.add(bezel);
    var rim = new THREE.LineSegments(new THREE.EdgesGeometry(bezel.geometry, 30),
      new THREE.LineBasicMaterial({ color: ACC_L, transparent: true, opacity: 0.8 }));
    rim.position.copy(bezel.position); g.add(rim);
    var back = box(SW + 0.18, SH + 0.18, 0.1, std(0x0c0c16, { roughness: 0.6 }), 0, 2.0, -0.08, g);
    box(0.28, 0.8, 0.16, std(0x1a1a30), 0, 0.7, -0.12, g);
    var base = box(1.5, 0.06, 0.6, std(0x1a1a30), 0, 0.03, -0.12, g); edges(base, ACC, 0.7);
    // status led
    var ledX = SW / 2 - 0.1;
    box(0.05, 0.02, 0.02, new THREE.MeshBasicMaterial({ color: 0x00fa89 }), ledX, 2.0 - SH / 2 - 0.05, ledX * ledX * CURVE + 0.03, g);
    screenLight.position.set(0, 1.9, 0.2);
  })();

  // --- mug (the keyboard was replaced by the two stonks buttons under the fly's front feet)
  (function mug() {
    var g = new THREE.Group(); g.position.set(2.7, 0, 0.3); g.scale.setScalar(1.4); scene.add(g);
    var body = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.17, 0.36, 10), std(0x8a8aff, { roughness: 0.5 }));
    body.position.y = 0.18; g.add(body); edges(body, 0xffffff, 0.25);
    var cof = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.01, 10), std(0x1a0d08));
    cof.position.y = 0.34; g.add(cof);
    var h = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.028, 5, 8, 3.6), std(0x8a8aff));
    h.position.set(0.22, 0.19, 0); h.rotation.z = -1.8; g.add(h);
  })();

  // --- the fly
  var fly = new THREE.Group(); fly.position.set(-0.1, 0, 0.95); fly.scale.setScalar(1.25); fly.rotation.y = -0.5; scene.add(fly);
  var bodyMat = std(0x8a94ad, { roughness: 0.65 }), darkMat = std(0x454d63);
  var eyeMat = new THREE.MeshStandardMaterial({ color: 0xc8102e, emissive: 0x9a0a20, emissiveIntensity: 0.9, flatShading: true, roughness: 0.4 });
  function ell(rx, ry, rz, mat, x, y, z, seg, parent) {
    var m = new THREE.Mesh(new THREE.SphereGeometry(1, seg || 8, (seg || 8) - 2), mat);
    m.scale.set(rx, ry, rz); m.position.set(x, y, z); (parent || fly).add(m); return m;
  }
  var thorax = ell(0.3, 0.26, 0.36, bodyMat, 0, 0.62, 0);
  var abdomen = ell(0.25, 0.23, 0.48, bodyMat, 0, 0.55, 0.62, 8); abdomen.rotation.x = -0.18;
  [0.4, 0.62, 0.84].forEach(function (z, i) { // dark abdomen bands
    var b = ell(0.255 - i * 0.02, 0.235 - i * 0.02, 0.05, darkMat, 0, 0.55 - (z - 0.62) * 0.18 * -1, z, 8); b.rotation.x = -0.18;
  });
  var head = new THREE.Group(); head.position.set(0, 0.66, -0.4); fly.add(head);
  ell(0.2, 0.19, 0.19, bodyMat, 0, 0, 0, 8, head);
  var eyeL = ell(0.14, 0.15, 0.14, eyeMat, -0.17, 0.02, -0.06, 7, head);
  var eyeR = ell(0.14, 0.15, 0.14, eyeMat, 0.17, 0.02, -0.06, 7, head);
  ell(0.05, 0.05, 0.1, darkMat, 0, -0.1, -0.17, 5, head); // proboscis
  // wings: two translucent planes hinged on the thorax
  var wingMat = new THREE.MeshStandardMaterial({ color: 0xd6dbff, emissive: 0x5a60c8, emissiveIntensity: 0.5, transparent: true, opacity: 0.6, side: THREE.DoubleSide, roughness: 0.2, flatShading: true, depthWrite: false });
  var wings = [-1, 1].map(function (s) {
    var piv = new THREE.Group(); piv.position.set(s * 0.14, 0.86, 0.05); fly.add(piv);
    var geo = new THREE.PlaneGeometry(0.3, 0.85, 1, 3); geo.rotateX(-Math.PI / 2); geo.translate(s * 0.2, 0, 0.5);
    var p = geo.attributes.position; for (var i = 0; i < p.count; i++) p.setY(i, Math.abs(p.getX(i)) * 0.15);
    piv.add(new THREE.Mesh(geo, wingMat)); piv.userData.s = s; return piv;
  });
  // legs: hip -> knee -> foot, two cylinders each
  var legMat = std(0x2e3752, { roughness: 0.6 });
  var LEGS = [
    { h: [0.12, 0.42, -0.24], f: [0.26, 0.09, -0.68], front: true },
    { h: [0.16, 0.42, 0.0], f: [0.62, 0.0, 0.0] },
    { h: [0.14, 0.42, 0.24], f: [0.55, 0.0, 0.62] }
  ];
  var up0 = new THREE.Vector3(0, 1, 0);
  var legs = [];
  // two big buttons under the front feet: NO STONKS (left) and a side button (right) whose label follows
  // the kind: STONKS on spot, LONG / SHORT STONKS on perps, UP / DOWN STONKS on event contracts
  var btnGroup = new THREE.Group(); btnGroup.rotation.y = 0.5; fly.add(btnGroup);  // cancels the fly's turn so text faces the camera
  var plate = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.06, 0.62), std(0x101020)); plate.position.set(0, 0.03, -0.68); fly.add(plate); edges(plate, ACC_L, 0.6);
  function capTex() {
    var cv = document.createElement('canvas'); cv.width = 256; cv.height = 256;
    var t = new THREE.CanvasTexture(cv); t.anisotropy = 4; return { cv: cv, tex: t };
  }
  function drawCap(c, lines, bg, fg) {
    var x = c.cv.getContext('2d'); x.clearRect(0, 0, 256, 256);
    x.fillStyle = bg; x.beginPath(); x.arc(128, 128, 128, 0, 6.2832); x.fill();
    x.strokeStyle = fg; x.lineWidth = 10; x.beginPath(); x.arc(128, 128, 116, 0, 6.2832); x.stroke();
    x.fillStyle = fg; x.textAlign = 'center'; x.textBaseline = 'middle';
    var n = lines.length, fs = n > 1 ? 44 : 46;
    x.font = '900 ' + fs + 'px system-ui, -apple-system, Segoe UI, sans-serif';
    lines.forEach(function (ln, i) { x.fillText(ln, 128, 128 + (i - (n - 1) / 2) * (fs + 6)); });
    c.tex.needsUpdate = true;
  }
  var BTN_LIFT = 0.1;
  var BTN_COL = [['#7a1f2b', '#ffd7dc', 0xff4a5f], ['#155c3a', '#d8ffe8', 0x35e08a]];
  var caps = [0, 1].map(function (i) {
    var c = capTex(); drawCap(c, i ? ['STONKS'] : ['NO', 'STONKS'], BTN_COL[i][0], BTN_COL[i][1]);
    var top = new THREE.MeshStandardMaterial({ map: c.tex, transparent: true, emissive: BTN_COL[i][2], emissiveIntensity: 0, roughness: 0.5 });
    var side = new THREE.MeshStandardMaterial({ color: parseInt(BTN_COL[i][0].slice(1), 16), roughness: 0.6, emissive: BTN_COL[i][2], emissiveIntensity: 0 });
    var m = new THREE.Group();
    var body = new THREE.Mesh(new THREE.CylinderGeometry(0.23, 0.23, 0.06 + BTN_LIFT, 40), side); body.position.y = BTN_LIFT / 2; m.add(body);  // taller button, same base
    var face = new THREE.Mesh(new THREE.CircleGeometry(0.23, 40), top); face.rotation.x = -Math.PI / 2; face.position.y = 0.032 + BTN_LIFT; m.add(face);
    var x = (i ? 1 : -1) * 0.26, z = -0.68;
    var w = new THREE.Vector3(x, 0.09, z).applyAxisAngle(up0, -0.5);  // place in the rotated group's frame
    m.position.set(w.x, 0.09, w.z); btnGroup.add(m);
    return { mesh: m, c: c, top: top, side: side, y: 0.09 };
  });
  var goLines = ['STONKS'], pressIdx = 0;
  function setGoLabel(lines) {
    if (lines.join('|') === goLines.join('|')) return;
    goLines = lines; drawCap(caps[1].c, lines, BTN_COL[1][0], BTN_COL[1][1]);
  }
  [-1, 1].forEach(function (s) {
    LEGS.forEach(function (L, li) {
      var seg = [0, 1].map(function () {
        var m = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.016, 1, 5), legMat); fly.add(m); return m;
      });
      legs.push({ s: s, L: L, seg: seg, li: li });
    });
  });
  var up = new THREE.Vector3(0, 1, 0), _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3();
  function limb(m, a, b) {
    _d.subVectors(b, a); var len = _d.length();
    m.position.copy(a).addScaledVector(_d, 0.5); m.scale.set(1, len, 1);
    m.quaternion.setFromUnitVectors(up, _d.normalize());
  }
  var hip = new THREE.Vector3(), knee = new THREE.Vector3(), foot = new THREE.Vector3();
  function poseLegs(t) {
    var sinceTap = t - tapAt;
    legs.forEach(function (lg, i) {
      var s = lg.s, L = lg.L;
      hip.set(s * L.h[0], L.h[1], L.h[2]);
      foot.set(s * L.f[0], L.f[1], L.f[2]);
      if (L.front) {
        // the front feet hover over the two buttons and twitch after a sniff; a decision slams one button:
        // wind up, strike, hold, release. Idle twitches never reach the cap.
        var idx = s > 0 ? 1 : 0, cap = caps[idx], h, depress = 0;
        var burst = Math.max(0, 1 - (t - typeAt) / 5);
        var gate = burst > 0 || ((t * 0.6) % 1) < 0.6 ? 1 : 0;
        var ph = (((t * (4.5 + 2 * burst) + (idx ? 0 : 0.5)) % 1) + 1) % 1;
        var T = 0.035, tap = reduceMotion ? 0 : Math.sin(Math.PI * ph) * gate;  // T: foot height when it rests on the cap
        h = T + 0.03 * (1 - gate) + tap * 0.13;
        depress = gate && !reduceMotion ? Math.max(0, Math.min(1, (T + 0.02 - h) / 0.02)) * 0.35 : 0;
        if (idx === pressIdx && sinceTap < 1.0) {
          var u = sinceTap;
          h = u < 0.18 ? 0.12 + u / 0.18 * 0.2 : u < 0.3 ? 0.32 - (u - 0.18) / 0.12 * 0.35 : u < 0.7 ? -0.03 : -0.03 + (u - 0.7) / 0.3 * 0.15;
          depress = Math.max(depress, Math.max(0, Math.min(1, (0.06 - h) / 0.09)));
        }
        foot.y += h + BTN_LIFT;
        cap.mesh.position.y = cap.y - 0.06 * depress;
        var glow = idx === pressIdx ? Math.max(0, 1 - Math.max(0, sinceTap - 0.2) / 1.6) * (sinceTap > 0.2 ? 1 : 0) : 0;
        var g2 = Math.max(glow * 0.9, depress * 0.5); cap.top.emissiveIntensity = g2; cap.side.emissiveIntensity = g2;
      } else if (!reduceMotion) foot.z += Math.sin(t * 1.1 + i) * 0.012;
      knee.copy(hip).lerp(foot, 0.5); knee.x += s * 0.16; knee.y += 0.3 - (L.front ? 0.06 : 0);
      limb(lg.seg[0], hip, knee); limb(lg.seg[1], knee, foot);
    });
  }
  // fly on the desk: subtle glow under it
  var pad = new THREE.Mesh(new THREE.CircleGeometry(0.9, 24), new THREE.MeshBasicMaterial({ color: ACC, transparent: true, opacity: 0.14 }));
  pad.rotation.x = -Math.PI / 2; pad.position.set(0, 0.005, 0.2); fly.add(pad);

  /* --- camera framing (desktop: 3/4 view, phones: close on the monitor) */
  function resize() {
    var p = canvas.parentElement, w = p.clientWidth, h = p.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    var wide = window.innerWidth > 860;
    camera.fov = wide ? 34 : 44;
    camera.userData.wide = wide;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas.parentElement); resize();
  var mouse = { x: 0, y: 0 };
  window.addEventListener('pointermove', function (e) { mouse.x = e.clientX / innerWidth - 0.5; mouse.y = e.clientY / innerHeight - 0.5; });

  var CAM_VIEWS = [[5.46, 3.78, 9.0], [6.72, 3.45, 8.24], [3.94, 3.38, 9.85]];  // centre, right, left
  var zoomK = 0, zoomTarget = 0;
  var _p = new THREE.Vector3(), _l = new THREE.Vector3();
  /* camera tuning: open the page with ?cam to get a panel of camera numbers (drag the scene to orbit, wheel to
     dolly, or type values). The address bar keeps the numbers, so a link like ?cam=2,3,8.4,0.2,1.25,-0.6,34
     reproduces exactly that view for anyone, including a test browser. Without ?cam nothing changes. */
  var camOv = null;
  (function camTune() {
    var q = new URLSearchParams(location.search);
    if (!q.has('cam') || (window.matchMedia && window.matchMedia('(max-width: 700px)').matches)) return;  // phones show the market only
    var v = (q.get('cam') || '').split(',').map(Number);
    var ok = v.length >= 6 && v.every(function (x) { return isFinite(x); });
    camOv = ok ? { p: v.slice(0, 3), l: v.slice(3, 6), fov: v[6] || 34 } : { p: [2.0, 3.0, 8.4], l: [0.2, 1.25, -0.6], fov: 34 };
    var box = document.createElement('div');
    box.style.cssText = 'position:fixed;bottom:12px;left:50%;transform:translateX(-50%);z-index:50;background:rgba(8,8,20,.92);border:1px solid #6e6eed;padding:8px 10px;font:12px/1.3 ui-monospace,monospace;color:#d8d8ff;display:flex;flex-wrap:wrap;gap:6px;align-items:center;max-width:92vw';
    var names = ['pos x', 'pos y', 'pos z', 'look x', 'look y', 'look z', 'fov'], inputs = [];
    function vals() { return camOv.p.concat(camOv.l, [camOv.fov]); }
    var out = document.createElement('input'); out.readOnly = true; out.style.cssText = 'width:260px;background:#0c0c1c;color:#fff;border:1px solid #444;font:inherit';
    function sync() {
      var a = vals().map(function (x) { return Math.round(x * 100) / 100; });
      inputs.forEach(function (el, i) { if (document.activeElement !== el) el.value = a[i]; });
      out.value = '?cam=' + a.join(',');
      try { history.replaceState(null, '', location.pathname + '?cam=' + a.join(',')); } catch (e) { }
    }
    names.forEach(function (n, i) {
      var l = document.createElement('label'); l.textContent = n + ' ';
      var el = document.createElement('input'); el.type = 'number'; el.step = i === 6 ? 1 : 0.1; el.style.cssText = 'width:64px;background:#0c0c1c;color:#fff;border:1px solid #444;font:inherit';
      el.oninput = function () {
        var x = parseFloat(el.value); if (!isFinite(x)) return;
        if (i < 3) camOv.p[i] = x; else if (i < 6) camOv.l[i - 3] = x; else camOv.fov = Math.max(10, Math.min(90, x));
        sync();
      };
      l.appendChild(el); box.appendChild(l); inputs.push(el);
    });
    var cp = document.createElement('button'); cp.textContent = 'copy'; cp.style.cssText = 'font:inherit;background:#6e6eed;color:#fff;border:0;padding:3px 8px;cursor:pointer';
    cp.onclick = function () { try { navigator.clipboard.writeText(location.origin + location.pathname + out.value); cp.textContent = 'copied'; setTimeout(function () { cp.textContent = 'copy'; }, 1200); } catch (e) { out.select(); } };
    box.appendChild(out); box.appendChild(cp); document.body.appendChild(box); sync();
    var drag = null;
    canvas.addEventListener('pointerdown', function (e) { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointerup', function () { drag = null; });
    canvas.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag = { x: e.clientX, y: e.clientY };
      var d = new THREE.Vector3(camOv.p[0] - camOv.l[0], camOv.p[1] - camOv.l[1], camOv.p[2] - camOv.l[2]);
      var sph = new THREE.Spherical().setFromVector3(d);
      sph.theta -= dx * 0.005; sph.phi = Math.max(0.05, Math.min(Math.PI - 0.05, sph.phi - dy * 0.005));
      d.setFromSpherical(sph); camOv.p = [camOv.l[0] + d.x, camOv.l[1] + d.y, camOv.l[2] + d.z]; sync();
    });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      var f = Math.exp(e.deltaY * 0.001);
      camOv.p = [0, 1, 2].map(function (i) { return camOv.l[i] + (camOv.p[i] - camOv.l[i]) * f; }); sync();
    }, { passive: false });
  })();
  function placeCamera(t) {
    if (camOv) {
      camera.position.set(camOv.p[0], camOv.p[1], camOv.p[2]); camera.lookAt(camOv.l[0], camOv.l[1], camOv.l[2]);
      if (camera.fov !== camOv.fov) { camera.fov = camOv.fov; camera.updateProjectionMatrix(); }
      return;
    }
    var wide = camera.userData.wide, ox = reduceMotion ? 0 : mouse.x * 0.5, oy = reduceMotion ? 0 : mouse.y * 0.25;
    var sway = reduceMotion ? 0 : Math.sin(t * 0.18) * 0.25;
    zoomK += (zoomTarget - zoomK) * 0.08;
    wingMat.opacity = 0.6 - 0.54 * zoomK;  // up close the wings would fill the corner of the screen
    if (wide) {
      // the camera drifts between three chosen views (centre, right, left) on a 30 s cycle, all looking at the same point
      var w = reduceMotion ? 0 : Math.sin(t * 0.21), a = CAM_VIEWS[0], b = w >= 0 ? CAM_VIEWS[1] : CAM_VIEWS[2], k = Math.abs(w);
      _p.set(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k).lerp(_l.set(0, 2.0, 3.6), zoomK);
      camera.position.copy(_p);
      camera.lookAt(-0.2 + 0.2 * zoomK, 0.9 + 1.1 * zoomK, -0.6 - 0.8 * zoomK);
    } else {
      camera.clearViewOffset();
      camera.position.set(1.7 + ox * 0.3 + sway * 0.4, 2.3, 6.4);
      camera.lookAt(0, 1.0, 0);
    }
  }

  /* --- per-frame animation */
  var clock0 = performance.now(), lastFrame = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    var t = (now - clock0) / 1000;
    if (!reduceMotion) {
      var br = Math.sin(t * 1.9);
      thorax.scale.y = 0.26 + br * 0.008; abdomen.scale.y = 0.23 + br * 0.012;
      var sn = t - sniffAt, twitch = sn < 0.9 ? Math.sin(sn / 0.9 * Math.PI) * Math.sin(sn * 40) * 0.08 : 0; // antennae twitch on a sniff
      head.rotation.x = Math.sin(t * 1.3) * 0.05 - (sn < 0.9 ? Math.sin(sn / 0.9 * Math.PI) * 0.12 : 0);
      head.rotation.y = Math.sin(t * 0.7) * 0.09 + twitch;
      head.position.y = 0.66 + Math.sin(t * 1.3) * 0.012;
      wings.forEach(function (w) {
        var flutter = Math.sin(t * 34 + w.userData.s) * 0.035 * (0.4 + 0.6 * Math.max(0, Math.sin(t * 0.5)));
        w.rotation.z = w.userData.s * (0.5 + flutter + Math.sin(t * 1.9) * 0.02);
      });
      var sp = scene.getObjectByName('stars'); sp.rotation.y = t * 0.004;
      scene.getObjectByName('planet').rotation.y = t * 0.02;
    } else wings.forEach(function (w) { w.rotation.z = w.userData.s * 0.5; });
    poseLegs(t);
    // dopamine pulse: light wash + eye glow
    var dt = t - dopamine.t, k = Math.max(0, 1 - dt / 1.6);
    dopaLight.intensity = 5 * k * k;
    dopaLight.color.setHex(dopamine.kind === 'PAM' ? 0x00fa89 : 0xdc0b4a);
    eyeMat.emissiveIntensity = 0.9 + (dopamine.kind === 'PAM' ? 1.4 : 0.6) * k;
    screenLight.intensity = 1.3 + Math.sin(t * 0.9) * 0.1;
    placeCamera(t);
    renderer.render(scene, camera);
    if (!firstFrame) { firstFrame = true; bootStage('scene ready', 0.55); idle(loadBrain); if (bootLive) bootDone(); }
  }
  function nowT() { return (performance.now() - clock0) / 1000; }

  /* ------------------------------------------------------------ terminal feed */
  function pushTouch(sym, tc) {
    if (!tc) return;
    var h = touchHist[sym] || (touchHist[sym] = []);
    h.push([Date.now(), num(tc[0]), num(tc[1])]);
    if (h.length > 800) h.shift();
  }
  function candlesFromTouch(sym) {
    var h = touchHist[sym] || [], out = [], B = 30000, cur = null;
    h.forEach(function (s) {
      var b = Math.floor(s[0] / B), m = (s[1] + s[2]) / 2;
      if (!cur || cur.b !== b) { cur = { b: b, o: m, h: m, l: m, c: m, v: 0 }; out.push(cur); }
      cur.h = Math.max(cur.h, m); cur.l = Math.min(cur.l, m); cur.c = m; cur.v += 1;
    });
    return out.slice(-60);
  }
  function fetchCandles(sym) {
    if (!live || !live.markets || !live.markets[sym]) return; // only spot has public REST candles
    var c = candles[sym] || (candles[sym] = { list: [], src: 'touch', at: 0 });
    if (Date.now() - c.at < CANDLE_MS) return;
    c.at = Date.now();
    var api = live.network === 'testnet' ? 'https://stg.api.dreamdex.io' : 'https://api.dreamdex.io';
    fetch(api + '/v0/markets/' + encodeURIComponent(sym) + '/candles?interval=1m&limit=60')
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) {
        var l = (d.candles || []).map(function (k) { return { o: num(k.open), h: num(k.high), l: num(k.low), c: num(k.close), v: num(k.volume) }; });
        if (l.length) { c.list = l; c.src = 'api'; }
      })
      .catch(function () { c.src = 'touch'; });
  }

  /* --- real-time touch straight from the public DreamDEX WebSocket (same feed the runner trades on).
     The runner's snapshot only refreshes once a minute; this keeps the monitor's bid/ask live. */
  var wsTouch = {}, wsBooks = {}, wsSock = null, wsBackoff = 1000, wsPing = null;
  function wsBest(sym) {
    var b = wsBooks[sym]; if (!b) return null;
    var bb = -Infinity, ba = Infinity, k;
    for (k in b.bids) if (+k > bb) bb = +k;
    for (k in b.asks) if (+k < ba) ba = +k;
    return (isFinite(bb) && isFinite(ba) && ba > bb) ? [bb, ba] : null;
  }
  function wsApply(msg) {
    var sym = msg.symbol, b = wsBooks[sym] || (wsBooks[sym] = { bids: {}, asks: {} });
    if (msg.type === 'snapshot') { b.bids = {}; b.asks = {}; }
    ['bids', 'asks'].forEach(function (side) {
      (msg[side] || []).forEach(function (l) {
        var q = num(l.quantity);
        if (q > 0) b[side][l.price] = q; else delete b[side][l.price];
      });
    });
    var t = wsBest(sym);
    if (t) { wsTouch[sym] = t; pushTouch(sym, t); }
  }
  function wsConnect() {
    if (!live || !live.network) return;
    var url = live.network === 'testnet' ? 'wss://stg.api.dreamdex.io/v0/ws/public' : 'wss://api.dreamdex.io/v0/ws/public';
    try { wsSock = new WebSocket(url); } catch (e) { return; }
    wsSock.onopen = function () {
      wsBackoff = 1000;
      wsSock.send(JSON.stringify({ operation: 'subscribe', channel: 'orderbook', params: { symbols: Object.keys(live.markets), depth: 10 } }));
      wsPing = setInterval(function () { if (wsSock && wsSock.readyState === 1) wsSock.send(JSON.stringify({ operation: 'ping' })); }, 30000);
    };
    wsSock.onmessage = function (ev) {
      var msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.channel === 'orderbook' && (msg.type === 'snapshot' || msg.type === 'update')) wsApply(msg);
    };
    wsSock.onclose = wsSock.onerror = function () {
      clearInterval(wsPing); wsSock = null; wsBooks = {};  // cold start on reconnect: never merge
      setTimeout(wsConnect, wsBackoff); wsBackoff = Math.min(wsBackoff * 2, 30000);
    };
  }

  var mterm = $('mterm'), mterminal = FlyTerminal.create(mterm, { compact: true });
  function drawTerminal() {
    if (!live || !selected || !live.all[selected]) return;
    var mk = live.all[selected], dec = mk.decision || {}, lv = mk.live || {}, kind = mk.kind || 'spot';
    var c = candles[selected], useApi = c && c.src === 'api' && c.list.length;
    var eps = episodes[selected] || [];
    var list = kind === 'perp' ? (mk.series || []) : useApi ? c.list : candlesFromTouch(selected);
    var touch = kind === 'perp' ? (mk.last && mk.last.best_bid != null ? [mk.last.best_bid, mk.last.best_ask] : null)
      : wsTouch[selected] || (live.touch || {})[selected];
    var st = {
      symbol: selected, symbols: Object.keys(live.all), kind: kind,
      network: kind === 'perp' ? 'testnet perps' : kind === 'ec' ? 'mainnet up/down' : live.network,
      touch: touch, candles: list, candleSrc: kind === 'perp' ? 'tape' : useApi ? 'api' : 'touch',
      arms: mk.arms || ARMS, arm: dec.arm || lv.arm, scores: dec.scores,
      bids: lv.bids, asks: lv.asks, inv: lv.inv != null ? lv.inv : lv.pos, fills: lv.fills,
      equity: mk.equity, dayPnl: mk.day_pnl,
      eqSeries: eps.map(function (e) { return e.equity; }).filter(function (v) { return typeof v === 'number'; }).slice(-40),
      chg: list.length > 1 ? list[list.length - 1].c - list[0].o : 0,
      ec: kind === 'ec' ? { last: mk.last || {}, open: mk.open || [], recent: mk.recent || [], tracked: mk.tracked } : null
    };
    terminal.draw(st); termTex.needsUpdate = true;
    if (mterm.offsetParent) mterminal.draw(st); // phones: the same terminal, as a readable in-page panel
  }

  /* ------------------------------------------------------------ HUD */
  var raster = $('raster'), rctx = raster.getContext('2d');
  function drawRaster() {
    var rows = decHist[selected] || [], dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = raster.clientWidth, h = raster.clientHeight;
    if (raster.width !== Math.round(w * dpr)) { raster.width = Math.round(w * dpr); raster.height = Math.round(h * dpr); }
    var W = raster.width, H = raster.height, rh = H / RASTER_ROWS;
    rctx.clearRect(0, 0, W, H);
    rctx.fillStyle = 'rgba(245,245,245,0.05)';
    for (var g = 1; g < 4; g++) rctx.fillRect(Math.round(W * g / 4), 0, 1, H);
    var n = rows.length;
    rows.forEach(function (r, i) {
      var y = H - (n - i) * rh, age = (n - 1 - i) / RASTER_ROWS;
      rctx.fillStyle = i === n - 1 ? '#ffffff' : 'rgba(163,163,255,' + (0.85 - age * 0.6).toFixed(2) + ')';
      var th = Math.max(1, rh - 1);
      r.kc.forEach(function (k) { rctx.fillRect(Math.floor(k * W / N_KC), y, Math.max(1, Math.round(dpr)), th); });
    });
  }
  var traceC = $('trace'), tctx = traceC.getContext('2d');
  function drawTrace() {
    var eps = (episodes[selected] || []).slice(-40), dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = traceC.clientWidth, h = traceC.clientHeight;
    if (traceC.width !== Math.round(w * dpr)) { traceC.width = Math.round(w * dpr); traceC.height = Math.round(h * dpr); }
    var W = traceC.width, H = traceC.height;
    tctx.clearRect(0, 0, W, H);
    var vals = eps.map(function (e) { return Math.abs(num(e.weights_moved) || 0); });
    var mx = Math.max.apply(null, vals.concat([1e-9]));
    var bw = W / 40;
    tctx.fillStyle = 'rgba(245,245,245,0.18)'; tctx.fillRect(0, H - 1, W, 1);
    eps.forEach(function (e, i) {
      var v = vals[i], bh = Math.max(2 * dpr, (H - 8 * dpr) * v / mx);
      tctx.fillStyle = e.dopamine === 'PAM' ? '#00fa89' : e.dopamine === 'PPL1' ? '#ff5a7a' : 'rgba(245,245,245,0.35)';
      tctx.fillRect(W - (eps.length - i) * bw + 1, H - bh - 1, Math.max(1, bw - 2), bh);
    });
    var last = eps[eps.length - 1];
    $('trace-note').textContent = !last ? 'waiting for the first episode'
      : last.dopamine === 'none' ? 'last episode: no dopamine, weights held'
        : 'last episode: ' + last.dopamine + (last.dopamine === 'PAM' ? ' reward' : ' punishment') + ', weights moved ' + (+num(last.weights_moved).toPrecision(2));
  }

  function renderHud() {
    if (!live || !selected || !live.all[selected]) return;
    var mk = live.all[selected], dec = mk.decision || {}, kc = dec.kc_active || [];
    $('last-dec').textContent = hms(dec.ts);
    $('kc-n').textContent = kc.length.toLocaleString('en-US');
    $('kc-pct').textContent = (kc.length / N_KC * 100).toFixed(1) + '% of 2,044';
    var sc = dec.scores || [0, 0, 0, 0], mn = Math.min.apply(null, sc), mx = Math.max.apply(null, sc);
    var arms = mk.arms || ARMS, box = $('arms');
    if (box.dataset.sig !== arms.join()) {
      box.dataset.sig = arms.join(); box.textContent = '';
      arms.forEach(function (a) {
        var r = document.createElement('div'); r.className = 'arm'; r.dataset.a = a;
        r.innerHTML = '<span class="dot"></span><span class="nm"></span><span class="bar"><i></i></span>';
        r.querySelector('.nm').textContent = a; box.appendChild(r);
      });
    }
    Array.prototype.forEach.call(box.children, function (r, i) {
      r.classList.toggle('on', r.dataset.a === dec.arm);
      r.querySelector('i').style.width = (mx > mn ? 6 + 94 * (sc[i] - mn) / (mx - mn) : r.dataset.a === dec.arm ? 100 : 6) + '%';
    });
    $('strat').textContent = dec.arm || '-';
    drawRaster(); drawTrace();
  }

  function setBadge() {
    var dot = $('dot'), age = $('age');
    if (!live) { age.textContent = 'connecting'; dot.className = 'dot'; return; }
    var s = Math.max(0, Math.round((Date.now() - live.wall) / 1000));
    // the runner rewrites live.json once a minute, so anything under ~2.5 minutes is live
    dot.className = 'dot ' + (s > 150 ? 'stale' : 'live');
    age.textContent = s > 150 ? 'offline, ' + s + 's old' : 'live ' + s + 's';
    $('kill').hidden = !live.kill;
    var mode = $('mode');
    if (mode) {
      // real orders on spot, perps or event contracts; otherwise the fills are simulated
      var real = [live.orders === 'live' ? 'spot' : null, live.perp_orders === 'live' ? 'perps' : null,
                  live.ec_orders === 'live' ? 'event contracts' : null].filter(Boolean);
      mode.textContent = real.length ? 'live orders: ' + real.join(' + ') : 'paper';
    }
    var mk = selected && live.all ? live.all[selected] : null, sn = mk && mk.sniff && mk.sniff.ts;
    var el = $('sniff-age');
    if (el) el.textContent = !sn ? 'waiting for the first sniff'
      : 'sniffed the market ' + Math.max(0, Math.round((Date.now() - sn) / 1000)) + ' s ago: ' + ((mk.sniff.kc_active || []).length) + ' cells lit';
  }

  function buildTabs() {
    var tabs = $('tabs'), names = Object.keys(live.all);
    if (tabs.dataset.sig !== names.join()) {
      tabs.dataset.sig = names.join(); tabs.textContent = '';
      names.forEach(function (m) {
        var b = document.createElement('button'), kind = (live.all[m] || {}).kind;
        b.type = 'button'; b.className = 'tab'; b.setAttribute('role', 'tab'); b.dataset.m = m;
        b.textContent = kind === 'perp' ? m.split('/')[0] + ' perp' : kind === 'ec' ? 'up/down' : m.split(':')[0];
        b.onclick = function () { selected = m; afterSelect(); };
        tabs.appendChild(b);
      });
    }
    Array.prototype.forEach.call(tabs.children, function (b) { b.setAttribute('aria-selected', b.dataset.m === selected ? 'true' : 'false'); });
  }

  /* ------------------------------------------------------------ data flow */
  var brain = null, brainReady = false;

  function fetchEpisodes(sym) {
    return fetch('/episodes/' + sym.replace(/[:\/]/g, '_') + '.jsonl', { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.text() : ''; }).catch(function () { return ''; })
      .then(function (txt) {
        var out = [];
        txt.split('\n').forEach(function (ln) { if (ln.trim()) { try { out.push(JSON.parse(ln)); } catch (e) { /* partial line */ } } });
        episodes[sym] = out;
      });
  }

  function ingest() {
    Object.keys(live.all).forEach(function (sym) {
      var mk = live.all[sym], d = mk.decision;
      if (live.markets[sym]) pushTouch(sym, (live.touch || {})[sym]);
      if (d && d.ts !== lastDecTs[sym]) {
        var h = decHist[sym] || (decHist[sym] = []);
        h.push({ ts: d.ts, kc: d.kc_active || [] }); if (h.length > RASTER_ROWS) h.shift();
      }
    });
  }

  function goLabelFor(mk, d) {
    var kind = mk.kind || 'spot', f = (d && d.features) || {};
    if (kind === 'ec') { var p = mk.last && mk.last.prob_up_book; return [p != null && p < 0.5 ? 'DOWN' : 'UP', 'STONKS']; }
    if (kind === 'perp') {
      var dir = d && d.arm === 'p_carry' ? (f.funding > 0 ? -1 : 1) : (f.trend < 0 ? -1 : 1);
      return [dir < 0 ? 'SHORT' : 'LONG', 'STONKS'];
    }
    return ['STONKS'];
  }
  function fireDecision(force) {
    var mk = live.all[selected]; if (!mk || !mk.decision) return;
    var d = mk.decision;
    if (!force && lastDecTs[selected] === d.ts) return;
    lastDecTs[selected] = d.ts;
    tapAt = nowT(); typeAt = nowT();
    pressIdx = d.arm === 'flat' ? 0 : 1; setGoLabel(goLabelFor(mk, d));
    if (brainReady) {
      brain.flashKC(d.kc_active || []);
      var pool = (mk.pools || [])[Math.max(0, (mk.arms || ARMS).indexOf(d.arm))] || [];
      brain.flashPool(pool);
    }
  }
  /* the fly smells the market between decisions (spot: every minute; perps: every tape row; EC: every
     10 s inside a window). Each new sniff lights the cells the current state activates. */
  var lastSniffTs = {};
  function fireSniff() {
    var mk = live.all[selected]; if (!mk || !mk.sniff || !mk.sniff.ts) return;
    if (lastSniffTs[selected] === mk.sniff.ts) return;
    lastSniffTs[selected] = mk.sniff.ts;
    sniffAt = nowT(); typeAt = nowT();
    if (brainReady) brain.sniffKC(mk.sniff.kc_active || []);
  }
  function fireDopamine() {
    var eps = episodes[selected] || [], last = eps[eps.length - 1];
    if (!last) { if (lastEpWall[selected] === undefined) lastEpWall[selected] = null; return; }
    var seen = lastEpWall[selected];
    if (seen !== undefined && seen !== last.wall && (last.dopamine === 'PAM' || last.dopamine === 'PPL1')) {
      dopamine = { kind: last.dopamine, t: nowT() };
      if (brainReady) brain.flashDopamine(last.dopamine);
      var f = $('flash'); f.className = 'flash ' + (last.dopamine === 'PAM' ? 'pam' : 'ppl1');
      void f.offsetWidth; f.classList.add('go');
      var tg = $('dopa'); tg.textContent = last.dopamine === 'PAM' ? 'PAM reward' : 'PPL1 punish'; tg.className = 'dopa ' + last.dopamine; void tg.offsetWidth; tg.classList.add('go');
    }
    lastEpWall[selected] = last.wall;
  }

  function afterSelect() {
    buildTabs();
    fetchCandles(selected);
    fetchEpisodes(selected).then(function () { fireDopamine(); renderHud(); drawTerminal(); });
    fireDecision(true); renderHud(); drawTerminal();
  }

  function poll() {
    fetch('/live.json', { cache: 'no-store' })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) {
        live = d;
        /* one map of every fly the runner has: spot markets, perp markets, the EC fly */
        live.all = {};
        Object.keys(d.markets || {}).forEach(function (s) { live.all[s] = d.markets[s]; });
        Object.keys(d.perps || {}).forEach(function (s) { live.all[s] = d.perps[s]; });
        if (d.ec) live.all.EC = d.ec;
        var names = Object.keys(live.all), first = !selected;
        if (!selected || !live.all[selected]) selected = names[0];
        if (!wsSock) wsConnect();
        ingest();
        buildTabs(); setBadge();
        fetchCandles(selected);
        return fetchEpisodes(selected).then(function () {
          fireDecision(first); fireSniff(); fireDopamine(); renderHud(); drawTerminal();
          Object.keys(d.markets).forEach(function (s) { if (d.markets[s].decision) lastDecTs[s] = d.markets[s].decision.ts; });
          bootLive = true; if (firstFrame) bootDone();
        });
      })
      .catch(function (e) { console.warn('poll failed', e && e.message); setBadge(); if (firstFrame) bootDone(); });
  }

  /* ------------------------------------------------------------ boot
     Order: scene first (small, all procedural), then the fly's live state, then the brain data in idle
     time (2.8 MB of neurons, then 2.3 MB of cell bodies), so the page is usable within a second and the
     heavy files never block it. The splash lifts once the scene has drawn and live.json has answered. */
  var boot = $('boot'), bootBar = $('boot-bar'), bootMsg = $('boot-msg'), firstFrame = false, bootLive = false;
  var idle = window.requestIdleCallback ? function (fn) { window.requestIdleCallback(fn, { timeout: 1500 }); } : function (fn) { setTimeout(fn, 200); };
  function bootStage(msg, frac) {
    if (bootMsg) bootMsg.textContent = msg;
    if (bootBar && frac != null) bootBar.style.width = Math.round(frac * 100) + '%';
  }
  function bootDone() {
    if (!boot || boot.classList.contains('done')) return;
    bootStage('ready', 1); boot.classList.add('done');
    setTimeout(function () { boot.hidden = true; }, 700);
  }
  function loadBrain() {
    var ld = $('pip-loading');
    try {
      brain = FlyBrain.create($('pip-gl'), {
        pixelRatio: Math.max(0.6, Math.min(window.devicePixelRatio || 1, 2) * 0.5),
        antialias: false, wheel: false, kcScale: 0.14, distScale: 2.1, spin: 0, view: 'front'
      });
      /* both files in parallel, one reveal: the cell bodies recentre the camera on the whole brain, so
         showing the circuit alone first would flash a zoomed-in brain that then jumps out */
      var prog = [0, 0], weight = [0.56, 0.44];
      var show = function () { ld.textContent = 'loading the brain ' + Math.round((prog[0] * weight[0] + prog[1] * weight[1]) * 100) + '%'; };
      Promise.all([
        brain.load('data/brain.json', function (f) { prog[0] = f; show(); }),
        brain.loadSomas('data/somas.json', { size: 0.9, dim: 0.035, onProgress: function (f) { prog[1] = f; show(); } })
          .catch(function (e) { console.warn('somas', e); })
      ]).then(function () {
        brain.finishRegions();
        brainReady = true; ld.classList.add('done');
        if (live) { fireDecision(true); delete lastSniffTs[selected]; fireSniff(); }  // show the latest sniff right away
      }).catch(function (e) { ld.textContent = 'brain data unavailable'; console.warn(e); });
    } catch (e) { ld.textContent = 'WebGL unavailable'; console.warn(e); }
  }

  /* phones get the market terminal only: no 3D scene, no brain data (see the phone-note in the page) */
  var phone = !!(window.matchMedia && window.matchMedia('(max-width: 700px)').matches);
  if (phone) { firstFrame = true; bootStage('phone view: market and trades', 0.55); }
  else { bootStage('building the scene', 0.2); requestAnimationFrame(frame); }
  var fontsReady = document.fonts && document.fonts.load
    ? Promise.all([document.fonts.load('600 22px Geist'), document.fonts.load('400 13px "Geist Mono"')]).catch(function () { })
    : Promise.resolve();
  fontsReady.then(function () {
    terminal.draw({ symbol: 'SOMI:USDso', symbols: ['SOMI:USDso'], candles: [], arms: ARMS, network: '' }); termTex.needsUpdate = true;
    bootStage('reading the fly', 0.75); poll();
  });
  setTimeout(bootDone, 8000); // never hold the page hostage to a slow or absent runner
  /* a hidden tab does no work: no polls, no redraws; a fresh poll the moment it is shown again */
  var tick = function (fn) { return function () { if (!document.hidden) fn(); }; };
  setInterval(tick(poll), POLL_MS);
  setInterval(tick(drawTerminal), TERM_MS);
  setInterval(tick(setBadge), 1000);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) { poll(); drawTerminal(); setBadge(); } });
  window.addEventListener('resize', function () { renderHud(); });
  $('zoom').addEventListener('click', function () { zoomTarget = zoomTarget ? 0 : 1; document.body.classList.toggle('zoomed', !!zoomTarget); this.textContent = zoomTarget ? 'zoom out' : 'zoom to monitor'; });
})();
