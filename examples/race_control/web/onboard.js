'use strict';
/* Onboard (chase-camera) views of a circuit, rendered with three.js from the lap
   samples the server already stores. The scene mirrors the MuJoCo one in
   envs/racing.py: dark asphalt, red/white kerbs, low rails, trees and the start
   gantry. Loaded before app.js; if three.js cannot be fetched (offline classroom),
   `Onboard.ready` resolves false and app.js falls back to the 2-D map. */
const Onboard = (() => {
  const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
  let THREE = null;
  const ready = import(THREE_URL).then(m => { THREE = m; return true; }).catch(() => false);

  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

  // The centre line as a closed loop, with a unit left-normal at every point.
  function loop(tr) {
    let pts = tr.points.map(p => [p[0], p[1]]);
    const a = pts[0], b = pts[pts.length - 1];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6) pts = pts.slice(0, -1);
    const n = pts.length;
    const nrm = pts.map((_, i) => {
      const p = pts[(i - 1 + n) % n], q = pts[(i + 1) % n];
      const tx = q[0] - p[0], ty = q[1] - p[1], l = Math.hypot(tx, ty) || 1;
      return [-ty / l, tx / l];
    });
    return { pts, nrm, n };
  }

  // A band along the circuit between two (offset, height) edges; one flat colour per
  // segment so kerbs can alternate. World (x, y) maps to three.js (x, height, -y).
  function band({ pts, nrm, n }, [offA, hA], [offB, hB], colorOf) {
    const pos = [], col = [], c = new THREE.Color();
    const at = (i, off, h) => { const p = pts[i % n], q = nrm[i % n]; return [p[0] + q[0] * off, h, -(p[1] + q[1] * off)]; };
    for (let i = 0; i < n; i++) {
      const a0 = at(i, offA, hA), b0 = at(i, offB, hB), a1 = at(i + 1, offA, hA), b1 = at(i + 1, offB, hB);
      pos.push(...a0, ...b0, ...a1, ...b0, ...b1, ...a1);
      c.set(colorOf(i));
      for (let k = 0; k < 6; k++) col.push(c.r, c.g, c.b);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.computeVertexNormals();
    return new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  }

  function canvasTexture(w, h, paint, repeat) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    paint(c.getContext('2d'), w, h);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat, repeat); }
    t.anisotropy = 4;
    return t;
  }

  // Same placement rule as the MuJoCo scene (offset along the local normal, kept only
  // if it clears the whole road), plus an outer ring of woodland for the horizon.
  function treeSpots({ pts, nrm, n }, w) {
    const spots = [];
    const clear = (x, y, c) => pts.every(p => Math.hypot(p[0] - x, p[1] - y) > c);
    for (let i = 0; i < 16; i++) {
      const k = Math.floor(i * n / 16), q = nrm[k];
      for (const side of (i % 2 ? [1, -1] : [-1, 1])) {
        const d = w / 2 + 1.4 + (i % 3) * .45, x = pts[k][0] + side * q[0] * d, y = pts[k][1] + side * q[1] * d;
        if (clear(x, y, w / 2 + .9)) { spots.push([x, y, 1, i % 3]); break; }
      }
    }
    for (let i = 0; i < 56; i++) {
      const k = Math.floor((i + .5) * n / 56), q = nrm[k], side = i % 2 ? 1 : -1;
      const d = w / 2 + 3.4 + ((i * 7) % 5) * .9, x = pts[k][0] + side * q[0] * d, y = pts[k][1] + side * q[1] * d;
      if (clear(x, y, w / 2 + 2.2)) spots.push([x, y, .8 + ((i * 5) % 4) * .16, i % 3]);
    }
    return spots;
  }

  function trees(spots) {
    const g = new THREE.Group();
    const trunkGeo = new THREE.CylinderGeometry(.1, .12, .84, 8), crownGeo = new THREE.IcosahedronGeometry(.5, 1);
    const trunkMat = new THREE.MeshLambertMaterial({ color: 0x57301a });
    const crowns = [0x1f6b2e, 0x2a7a33, 0x357f2a].map(c => new THREE.MeshLambertMaterial({ color: c, flatShading: true }));
    for (const [x, y, s, kind] of spots) {
      const trunk = new THREE.Mesh(trunkGeo, trunkMat);
      trunk.position.set(x, .42 * s, -y); trunk.scale.setScalar(s);
      const crown = new THREE.Mesh(crownGeo, crowns[kind]);
      crown.position.set(x, 1.08 * s, -y); crown.scale.set(s, s * 1.15, s);
      g.add(trunk, crown);
    }
    return g;
  }

  function startGantry({ pts, nrm }, w) {
    const g = new THREE.Group(), [x, y] = pts[0], q = nrm[0];
    g.position.set(x, 0, -y);
    g.rotation.y = Math.atan2(-q[0], q[1]);   // local +x along the racing direction
    const checker = canvasTexture(32, 128, (ctx, cw, ch) => {
      for (let i = 0; i < 8; i++) for (let j = 0; j < 2; j++) {
        ctx.fillStyle = (i + j) % 2 ? '#111317' : '#f4f4f0';
        ctx.fillRect(j * cw / 2, i * ch / 8, cw / 2, ch / 8);
      }
    });
    checker.magFilter = THREE.NearestFilter;
    const line = new THREE.Mesh(new THREE.PlaneGeometry(.22, w), new THREE.MeshLambertMaterial({ map: checker }));
    line.rotation.x = -Math.PI / 2; line.position.y = .018;
    g.add(line);
    const half = w / 2 + .38, yellow = new THREE.MeshLambertMaterial({ color: 0xeab814 });
    for (const side of [-1, 1]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(.075, .075, 2.3, 12), yellow);
      post.position.set(0, 1.15, side * half);
      g.add(post);
    }
    const board = new THREE.Mesh(new THREE.BoxGeometry(.16, .84, 2 * half + .1), new THREE.MeshLambertMaterial({ color: 0x1f5bdb }));
    board.position.y = 2.18;
    g.add(board);
    const label = canvasTexture(512, 128, (ctx, cw, ch) => {
      ctx.fillStyle = '#1f5bdb'; ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = '#ffeb24'; ctx.font = '700 84px "Space Grotesk", system-ui, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('FINISH', cw / 2, ch / 2 + 4);
    });
    for (const side of [-1, 1]) {
      const face = new THREE.Mesh(new THREE.PlaneGeometry(2 * half, .7), new THREE.MeshBasicMaterial({ map: label }));
      face.position.set(side * .081, 2.18, 0);
      face.rotation.y = side * Math.PI / 2;
      g.add(face);
    }
    return g;
  }

  // The rental kart of the MuJoCo scene, in the kart frame (+x forward, +y up).
  function kart(bodyColor) {
    const g = new THREE.Group(), mat = c => new THREE.MeshLambertMaterial({ color: c });
    const paint = mat(bodyColor), black = mat(0x1a1d22), tyre = mat(0x0c0d0f), metal = mat(0xb9bdc1);
    const box = (sx, sy, sz, m, x, y, z) => {
      const o = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), m); o.position.set(x, y, z); g.add(o);
    };
    const blob = (rx, ry, rz, m, x, y, z) => {
      const o = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), m); o.scale.set(rx, ry, rz); o.position.set(x, y, z); g.add(o);
    };
    const wheel = (r, wd, x, z) => {
      for (const [rr, ww, m] of [[r, wd, tyre], [r * .55, wd + .012, metal]]) {
        const o = new THREE.Mesh(new THREE.CylinderGeometry(rr, rr, ww, 20), m);
        o.rotation.x = Math.PI / 2; o.position.set(x, r, z); g.add(o);
      }
    };
    box(.78, .042, .44, black, -.04, .262, 0);              // floor tray
    blob(.24, .082, .19, paint, .48, .33, 0);               // nose
    for (const z of [-.3, .3]) box(.5, .124, .124, paint, .28, .315, z);   // side pods
    box(.058, .058, .70, black, .65, .33, 0);               // front bumper
    box(.09, .074, .70, black, -.52, .34, 0);               // rear bumper
    box(.058, .042, .56, paint, -.52, .41, 0);
    box(.24, .2, .32, mat(0x2a2e33), -.39, .5, 0);          // engine
    for (const z of [-.39, .39]) wheel(.119, .124, .46, z);
    for (const z of [-.4, .4]) wheel(.144, .156, -.40, z);
    blob(.2, .13, .19, mat(0x0d1116), -.15, .405, 0);       // seat
    blob(.14, .2, .16, mat(0x1e2f45), -.18, .545, 0);       // driver
    blob(.12, .12, .12, mat(0x0d42a8), -.04, .8, 0);        // helmet
    blob(.05, .04, .085, black, .06, .81, 0);               // visor
    const shadow = canvasTexture(64, 64, (ctx, w, h) => {
      const r = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
      r.addColorStop(0, '#000a'); r.addColorStop(1, '#0000');
      ctx.fillStyle = r; ctx.fillRect(0, 0, w, h);
    });
    const s = new THREE.Mesh(new THREE.PlaneGeometry(1.7, 1.15),
      new THREE.MeshBasicMaterial({ map: shadow, transparent: true, depthWrite: false }));
    s.rotation.x = -Math.PI / 2; s.position.y = .02;
    g.add(s);
    return g;
  }

  function buildScene(tr, kartColor) {
    const scene = new THREE.Scene(), ring = loop(tr), w = tr.width;
    scene.background = canvasTexture(4, 256, (ctx, cw, ch) => {
      const g = ctx.createLinearGradient(0, 0, 0, ch);
      g.addColorStop(0, '#2c5e9c'); g.addColorStop(.55, '#86b4dc'); g.addColorStop(1, '#cfe3f0');
      ctx.fillStyle = g; ctx.fillRect(0, 0, cw, ch);
    });
    scene.fog = new THREE.Fog(0xb9d3e6, 14, 46);
    scene.add(new THREE.HemisphereLight(0xe8f3ff, 0x30602f, 2.2));
    const sun = new THREE.DirectionalLight(0xfff3dd, 2.2);
    sun.position.set(-6, 10, 4);
    scene.add(sun);

    const grass = canvasTexture(64, 64, (ctx, cw, ch) => {
      ctx.fillStyle = '#2f7f37'; ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = '#378a3d'; ctx.fillRect(0, 0, cw / 2, ch);   // mown stripes
    }, 48);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), new THREE.MeshLambertMaterial({ map: grass }));
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    const h = w / 2;
    scene.add(band(ring, [-h, .01], [h, .01], () => '#2b2f36'));                          // asphalt
    for (const s of [-1, 1]) {
      scene.add(band(ring, [s * (h - .09), .013], [s * (h - .05), .013], () => '#e9ecef'));    // edge line
      scene.add(band(ring, [s * h, .03], [s * (h + .2), .03],
        i => Math.floor(i / 3) % 2 ? '#f1f1ea' : '#d4261c'));                              // kerbs
      scene.add(band(ring, [s * (h + .2), .011], [s * (h + .34), .011], () => '#8f8a7d'));  // gravel strip
      scene.add(band(ring, [s * (h + .34), 0], [s * (h + .34), .32],
        i => i % 12 < 1 ? '#d8ff38' : '#28313d'));                                          // rails
    }
    scene.add(trees(treeSpots(ring, w)));
    scene.add(startGantry(ring, w));
    const car = kart(kartColor);
    scene.add(car);
    return { scene, car };
  }

  /** A chase camera on one circuit. `frame(sample, dt)` places the kart and renders. */
  class View {
    constructor(canvas, track, { kartColor = 0xe31b0f } = {}) {
      this.canvas = canvas;
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      Object.assign(this, buildScene(track, kartColor));
      this.camera = new THREE.PerspectiveCamera(58, 16 / 9, .05, 120);
      this.yaw = 0; this.last = null; this.size = '';
    }

    frame(s, dt, showKart = true) {
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      if (!w || !h) return;
      if (this.size !== `${w}x${h}`) {
        this.size = `${w}x${h}`;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
      }
      this.car.visible = showKart;
      this.car.position.set(s.x, 0, -s.y);
      this.car.rotation.y = s.heading;
      // The camera follows a smoothed heading, so the kart visibly yaws into corners;
      // a restart or a scrub snaps it into place instead of swinging round.
      const jump = !this.last || Math.hypot(s.x - this.last.x, s.y - this.last.y) > 1.5;
      this.yaw = jump ? s.heading : this.yaw + wrap(s.heading - this.yaw) * (1 - Math.exp(-dt * 4));
      this.last = s;
      const fx = Math.cos(this.yaw), fy = Math.sin(this.yaw);
      this.camera.position.set(s.x - fx * 2.7, 1.15, -(s.y - fy * 2.7));
      this.camera.lookAt(s.x + fx * 2.2, .3, -(s.y + fy * 2.2));
      this.renderer.render(this.scene, this.camera);
    }

    dispose() {
      this.scene.traverse(o => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); }
      });
      if (this.scene.background) this.scene.background.dispose();
      this.renderer.dispose();
      this.renderer.forceContextLoss();
    }
  }

  return {
    ready,
    /** A View, or null when three.js is unavailable or WebGL fails. */
    view(canvas, track, options) {
      if (!THREE || !track) return null;
      try { return new View(canvas, track, options); } catch { return null; }
    },
  };
})();
