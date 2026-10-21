'use strict';
/* Fuzzy Grand Prix leaderboard. Plain DOM, no build step: served as-is by server.py. */

const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const TRACKS = ['oval', 'chicane', 'technical'];
const SHORT = { oval: 'OVL', chicane: 'CHI', technical: 'TEC' };
const TERM_COLORS = ['#49d6ff', '#d8ff38', '#ff594d', '#ffd23f', '#b28dff', '#5cf2b0', '#ff8fd0'];
const fmt = t => t == null ? '<span class="dnf">DNF</span>' : t.toFixed(3);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

let geometry = {}, sampleDt = 0.08, board = null, version = -1;
let seenRecent = null;          // ids already toasted; null until the first load
const entryCache = new Map();   // finished entries never change, so cache them
const adminKey = new URLSearchParams(location.search).get('key') || '';
let isAdmin = false;

/* ------------------------------------------------------------ geometry */

function trackSvg(name, line, { pad = 1.2, lineColor = 'var(--acid)', road = true } = {}) {
  const tr = geometry[name];
  if (!tr) return '';
  const xs = tr.points.map(p => p[0]), ys = tr.points.map(p => -p[1]);
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const pts = arr => arr.map(p => `${p[0].toFixed(2)},${(-p[1]).toFixed(2)}`).join(' ');
  const outline = pts(tr.points);
  const [sx, sy] = tr.points[0], [nx, ny] = tr.points[1];
  const len = Math.hypot(nx - sx, ny - sy) || 1;
  const ux = -(ny - sy) / len * tr.width / 2, uy = (nx - sx) / len * tr.width / 2;
  return `<svg viewBox="${x0} ${y0} ${x1 - x0} ${y1 - y0}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    <polygon points="${outline}" fill="none" stroke="${road ? 'var(--road)' : 'var(--line)'}" stroke-width="${road ? tr.width : .5}" stroke-linejoin="round"/>
    <line x1="${sx - ux}" y1="${-(sy - uy)}" x2="${sx + ux}" y2="${-(sy + uy)}" stroke="#fff" stroke-width=".25"/>
    ${line && line.length ? `<polyline points="${pts(line)}" fill="none" stroke="${lineColor}" stroke-width=".22" stroke-linejoin="round" stroke-linecap="round"/>` : ''}
  </svg>`;
}

/* ------------------------------------------------------------ rendering */

function badgeText(summary) {
  if (!summary) return '';
  return ['throttle', 'steering'].map(k => summary[k]
    ? `<span class="badge">${k === 'throttle' ? 'THR' : 'STR'} ${esc(summary[k].mode)} · ${summary[k].rules}</span>` : '').join('');
}

function card(row, i, leader) {
  const gap = row.total == null ? `${row.finished}/3 finished`
    : i === 0 ? 'LEADER' : `+${(row.total - leader).toFixed(3)}`;
  return `<button class="kart p${i + 1}" data-id="${esc(row.id)}" data-name="${esc(row.name)}">
    <div class="kart-top">
      <span class="pos">P${i + 1}</span>
      <div><div class="driver">${esc(row.name)}</div>
        <div class="badges">${badgeText(row.summary)}<span class="badge">${row.attempts} run${row.attempts === 1 ? '' : 's'}</span></div></div>
      <div class="total"><b>${row.total == null ? '<span class="dnf">DNF</span>' : row.total.toFixed(3)}</b><small>${gap}</small></div>
    </div>
    <div class="maps">${TRACKS.map(t => {
      const lap = row.laps[t];
      const color = lap && lap.time == null ? 'var(--red)' : 'var(--acid)';
      return `<div class="map">${trackSvg(t, row.paths && row.paths[t], { lineColor: color })}<span>${SHORT[t]} <b>${lap ? fmt(lap.time) : '—'}</b></span></div>`;
    }).join('')}</div>
  </button>`;
}

function render(data) {
  board = data;
  const rows = data.standings;
  const leader = rows[0] && rows[0].total;
  $('#stat-drivers').textContent = rows.length;
  $('#stat-runs').textContent = data.runs;

  const racing = data.queue.filter(q => q.status === 'racing');
  const waiting = data.queue.length - racing.length;
  $('#ticker').classList.toggle('live', racing.length > 0);
  $('#ticker-text').innerHTML = racing.length
    ? `NOW RACING ${racing.map(r => `<b>${esc(r.name)}</b>${r.track ? ` on ${esc(r.track)}` : ''}`).join(', ')}${waiting ? ` · ${waiting} in pit lane` : ''}`
    : 'Track clear · submit your driver';

  updateBroadcast(data.records);

  const tiles = rows.filter(r => r.paths);
  $('#grid').innerHTML = tiles.length ? tiles.map((r, i) => card(r, i, leader)).join('')
    : `<div class="empty-card"><h3>The grid is empty</h3><p>Submit a fuzzy driver to take pole position.</p>
       <button class="cta" onclick="openSubmit()">ENTER THE RACE</button></div>`;

  $('#standings').innerHTML = rows.map((r, i) => `<button class="srow" data-id="${esc(r.id)}">
      <span>${i + 1}</span><span class="name">${esc(r.name)}</span>
      ${TRACKS.map(t => `<span>${r.laps[t] ? fmt(r.laps[t].time) : '—'}</span>`).join('')}
      <span class="tot">${fmt(r.total)}</span></button>`).join('')
    || '<p class="none">No finished runs yet.</p>';

  $('#queue').innerHTML = data.queue.map(q => `<div class="qrow"><b>${esc(q.name)}</b>
      <span class="${q.status}">${q.status === 'racing' ? `racing · ${esc(q.track || '…')}` : 'waiting'}</span></div>`).join('')
    || '<p class="none">Pit lane is empty.</p>';

  $('#recent').innerHTML = data.recent.map(r => `<div class="rrow" data-id="${esc(r.id)}"><b>${esc(r.name)}</b>
      <span>${r.status === 'error' ? '<span class="dnf">ERROR</span>' : r.total != null ? r.total.toFixed(3) + ' s' : `<span class="dnf">DNF</span> ${r.finished}/3`}</span></div>`).join('')
    || '<p class="none">Nothing yet.</p>';

  announce(data);
}

function announce(data) {
  const ids = data.recent.map(r => r.id);
  if (seenRecent === null) { seenRecent = new Set(ids); return; }
  for (const r of data.recent.slice().reverse()) {
    if (seenRecent.has(r.id)) continue;
    seenRecent.add(r.id);
    const rank = data.standings.findIndex(s => s.name.toLowerCase() === r.name.toLowerCase());
    if (r.status === 'error') toast(`<b>${esc(r.name)}</b><br>run failed`, 'err');
    else if (rank === 0 && r.total != null && data.standings[0].id === r.id)
      toast(`NEW LEADER<br><b>${esc(r.name)}</b> · ${r.total.toFixed(3)} s`, 'lead');
    else toast(`<b>${esc(r.name)}</b> ${r.total != null ? `finished · ${r.total.toFixed(3)} s` : `DNF · ${r.finished}/3 circuits`}${rank >= 0 ? ` · P${rank + 1}` : ''}`);
    const tile = [...document.querySelectorAll('.kart')].find(k => k.dataset.id === r.id);
    if (tile) tile.classList.add('fresh');
  }
}

function toast(html, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = html;
  $('#toasts').append(el);
  setTimeout(() => el.remove(), 6000);
}

async function poll() {
  try {
    const res = await fetch('/api/leaderboard', { cache: 'no-cache' });
    const data = await res.json();
    if (data.version !== version) { version = data.version; render(data); }
  } catch {
    $('#ticker').classList.remove('live');
    $('#ticker-text').textContent = 'Race Control offline · retrying';
  }
}

/* ----------------------------------------------------------- submission */

// Same layout as example_driver.json: one membership function and one rule per line.
function pretty(o, ind = 0) {
  const p = '  '.repeat(ind);
  if (Array.isArray(o) && o.length && o[0] && typeof o[0] === 'object' && !Array.isArray(o[0]))
    return '[\n' + o.map(x => `${p}  ${pretty(x, ind + 1)}`).join(',\n') + `\n${p}]`;
  if (Array.isArray(o) || (o && typeof o === 'object' && Object.keys(o).every(k => k === 'if' || k === 'then')))
    return compact(o);
  if (o && typeof o === 'object')
    return '{\n' + Object.entries(o).map(([k, v]) => `${p}  ${JSON.stringify(k)}: ${pretty(v, ind + 1)}`).join(',\n') + `\n${p}}`;
  return JSON.stringify(o);
}

function compact(o) {
  if (Array.isArray(o)) return '[' + o.map(compact).join(', ') + ']';
  if (o && typeof o === 'object')
    return '{' + Object.entries(o).map(([k, v]) => `${JSON.stringify(k)}: ${compact(v)}`).join(', ') + '}';
  return JSON.stringify(o);
}

let progressTimer = null;

function openSubmit(json) {
  if (!$('#driver-name').value) $('#driver-name').value = store.get('gp-name') || '';
  if (json) $('#driver-json').value = json;
  else if (!$('#driver-json').value) $('#driver-json').value = store.get('gp-draft') || '';
  $('#submit-form').hidden = false;
  $('#submit-progress').hidden = true;
  $('#submit-error').hidden = true;
  checkJson();
  $('#submit-dialog').showModal();
  ($('#driver-name').value ? $('#driver-json') : $('#driver-name')).focus();
}
window.openSubmit = openSubmit;

function checkJson() {
  const text = $('#driver-json').value.trim(), hint = $('#json-hint');
  store.set('gp-draft', $('#driver-json').value);
  if (!text) { hint.classList.remove('bad'); hint.dataset.err = ''; return null; }
  try {
    const spec = JSON.parse(text);
    hint.classList.remove('bad');
    if (hint.dataset.base) hint.innerHTML = hint.dataset.base;
    return spec;
  } catch (e) {
    if (!hint.dataset.base) hint.dataset.base = hint.innerHTML;
    hint.classList.add('bad');
    hint.textContent = `JSON syntax: ${e.message}`;
    return undefined;
  }
}

async function submit() {
  const name = $('#driver-name').value.trim();
  const spec = checkJson();
  const err = $('#submit-error');
  const fail = msg => { err.textContent = msg; err.hidden = false; };
  err.hidden = true;
  if (!name) return fail('Please enter your name.');
  if (spec === null) return fail('Paste or upload your driver JSON (or load the example).');
  if (spec === undefined) return fail('The JSON has a syntax error; see the hint under the editor.');
  store.set('gp-name', name);
  const button = $('#submit');
  button.disabled = true;
  try {
    const res = await fetch('/api/submit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, spec }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) return fail(body.error || `Race Control answered ${res.status}`);
    $('#submit-form').hidden = true;
    $('#submit-progress').hidden = false;
    trackProgress(body.id);
  } catch {
    fail('Could not reach Race Control. Check your connection and try again.');
  } finally {
    button.disabled = false;
  }
}

function trackProgress(id) {
  clearTimeout(progressTimer);
  const step = async () => {
    if (!$('#submit-dialog').open) return;
    let e;
    try { e = await (await fetch(`/api/entries/${id}`, { cache: 'no-store' })).json(); }
    catch { progressTimer = setTimeout(step, 1500); return; }
    const done = e.status === 'done' || e.status === 'error';
    const light = t => {
      const lap = e.laps[t];
      const cls = lap ? (lap.time != null ? 'ok' : 'bad') : e.track === t ? 'racing' : '';
      return `<div class="light ${cls}"><i></i>${SHORT[t]}<span>${lap ? (lap.time != null ? lap.time.toFixed(2) : esc(lap.detail || 'DNF')) : ''}</span></div>`;
    };
    let headline, sub;
    if (e.status === 'queued') { headline = 'In the pit lane'; sub = e.position ? `${e.position} driver${e.position === 1 ? '' : 's'} ahead of you` : 'Next up'; }
    else if (e.status === 'racing') { headline = 'On track'; sub = `Racing ${esc(e.track || '')}…`; }
    else if (e.status === 'error') { headline = 'Run failed'; sub = esc(e.error); }
    else {
      if (!board || board.version < e.version) await poll();   // rank against fresh standings
      const rank = board ? board.standings.findIndex(s => s.id === e.id) : -1;
      headline = e.total != null ? `${e.total.toFixed(3)} s` : `DNF · ${e.finished}/3 circuits`;
      sub = rank >= 0 ? `New personal best · P${rank + 1} on the grid` : 'Finished; your earlier run is still your best';
    }
    $('#submit-progress').innerHTML = `<div class="progress">
      <p class="eyebrow">${esc(e.name)}</p><h3>${headline}</h3><p class="muted">${sub}</p>
      <div class="lights">${TRACKS.map(light).join('')}</div>
      <div class="row">${done && e.status === 'done' ? '<button class="cta" id="see-replay">WATCH REPLAY</button>' : ''}
        <button class="ghost" id="edit-again">Edit and resubmit</button></div></div>`;
    $('#edit-again').onclick = () => { $('#submit-form').hidden = false; $('#submit-progress').hidden = true; };
    const see = $('#see-replay');
    if (see) see.onclick = () => { $('#submit-dialog').close(); openEntry(e.id); };
    if (!done) progressTimer = setTimeout(step, 1000);
  };
  step();
}

/* ---------------------------------------------------------- entry view */

async function getEntry(id) {
  if (entryCache.has(id)) return entryCache.get(id);
  const res = await fetch(`/api/entries/${id}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('not found');
  const e = await res.json();
  if (e.status === 'done' || e.status === 'error') entryCache.set(id, e);
  return e;
}

let current = null;   // {entry, ghost, tab, track}
let anim = null;      // replay animation state

async function openEntry(id, track) {
  let entry;
  try { entry = await getEntry(id); } catch { toast('That entry no longer exists.', 'err'); return; }
  const lead = board && board.standings[0];
  let ghost = null;
  if (lead && lead.id !== id && lead.total != null) {
    try { ghost = await getEntry(lead.id); } catch { /* replay without a ghost */ }
  }
  const rank = board ? board.standings.findIndex(s => s.id === id) : -1;
  // A DNF opens on the circuit it failed, which is usually the interesting replay.
  const firstDnf = TRACKS.find(t => entry.laps[t] && entry.laps[t].time == null);
  current = { entry, ghost, tab: 'replay', track: track || firstDnf || 'oval' };
  $('#entry-head').innerHTML = `<p class="eyebrow">${rank >= 0 ? `P${rank + 1} · ` : ''}${esc(entry.submitted_at.replace('T', ' '))}</p>
    <h2>${esc(entry.name)}</h2>
    <div class="entry-meta"><span>Total <b>${entry.total != null ? entry.total.toFixed(3) + ' s' : 'DNF'}</b></span>
      ${TRACKS.map(t => `<span>${esc(t)} <b>${entry.laps[t] ? fmt(entry.laps[t].time) : '—'}</b></span>`).join('')}</div>
    ${entry.description ? `<p class="hint" style="margin-top:8px">${esc(entry.description)}</p>` : ''}`;
  showTab('replay');
  const dlg = $('#entry-dialog');
  if (!dlg.open) dlg.showModal();
}

function showTab(tab) {
  current.tab = tab;
  document.querySelectorAll('#entry-tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
  stopReplay();
  const body = $('#entry-body');
  if (tab === 'replay') renderReplay(body);
  else if (tab === 'setup') body.innerHTML = renderSetup(current.entry.spec);
  else renderJson(body);
}

/* --------------------------------------------------------------- replay */

function speedColor(v) {
  // 0 m/s -> cyan, 2 m/s -> acid, 4 m/s -> red
  const s = Math.max(0, Math.min(1, v / 4));
  const stops = [[73, 214, 255], [216, 255, 56], [255, 89, 77]];
  const k = s < .5 ? 0 : 1, f = s < .5 ? s * 2 : (s - .5) * 2;
  const c = stops[k].map((a, i) => Math.round(a + (stops[k + 1][i] - a) * f));
  return `rgb(${c})`;
}

function renderReplay(body) {
  const { entry, ghost } = current;
  body.innerHTML = `<div class="replay">
    <div><canvas id="replay-canvas"></canvas>
      <input type="range" id="scrub" class="scrub" min="0" max="1000" value="0" aria-label="Replay position"></div>
    <div class="controls">
      <div class="inset" id="inset"><canvas id="onboard-canvas"></canvas><span class="tv-live"><i></i>ONBOARD</span></div>
      <div class="seg" id="replay-tracks">${TRACKS.map(t => `<button data-track="${t}">${SHORT[t]}</button>`).join('')}</div>
      <div class="seg"><button id="play" class="on">PAUSE</button></div>
      <div class="seg" id="replay-speed">${[1, 2, 4].map(s => `<button data-speed="${s}" class="${s === 1 ? 'on' : ''}">${s}×</button>`).join('')}</div>
      <div class="readout">
        <div>time <b id="r-time">0.00</b></div>
        <div>speed <b id="r-speed">0.00</b></div>
        <div>result <b id="r-result">—</b></div>
      </div>
      <div class="legend-line"><i style="background:linear-gradient(90deg,${speedColor(0)},${speedColor(2)},${speedColor(4)})"></i>racing line, slow → fast</div>
      ${ghost ? `<div class="legend-line"><i style="background:var(--white);opacity:.55"></i>ghost: leader ${esc(ghost.name)}</div>` : ''}
    </div></div>`;
  body.querySelectorAll('#replay-tracks button').forEach(b => b.onclick = () => startReplay(b.dataset.track));
  body.querySelectorAll('#replay-speed button').forEach(b => b.onclick = () => {
    anim.speed = +b.dataset.speed;
    body.querySelectorAll('#replay-speed button').forEach(x => x.classList.toggle('on', x === b));
  });
  $('#play').onclick = () => {
    if (!anim) return;
    if (anim.t >= anim.duration) anim.t = 0;
    anim.playing = !anim.playing;
    $('#play').textContent = anim.playing ? 'PAUSE' : 'PLAY';
  };
  $('#scrub').oninput = e => { anim.t = anim.duration * e.target.value / 1000; anim.playing = false; $('#play').textContent = 'PLAY'; };
  startReplay(current.track);
}

function startReplay(track) {
  stopReplay();
  current.track = track;
  document.querySelectorAll('#replay-tracks button').forEach(b => b.classList.toggle('on', b.dataset.track === track));
  const lap = current.entry.laps[track];
  const ghostLap = current.ghost && current.ghost.laps[track];
  const path = lap ? lap.path : [];
  const ghostPath = ghostLap ? ghostLap.path : null;
  const durationOf = p => p ? (p.length - 1) * sampleDt : 0;
  $('#r-result').innerHTML = lap ? (lap.time != null ? `${lap.time.toFixed(3)} s` : `<span class="dnf">${esc(lap.detail || 'DNF')}</span>`) : '—';
  anim = { track, path, ghostPath, t: 0, speed: anim ? anim.speed : 1, playing: true, last: null,
           duration: Math.max(durationOf(path), durationOf(ghostPath)), own: durationOf(path), dt: 0,
           view: path.length > 1 ? Onboard.view($('#onboard-canvas'), geometry[track]) : null };
  $('#inset').hidden = !anim.view;
  const speedBtn = document.querySelector('#replay-speed button.on');
  if (speedBtn) anim.speed = +speedBtn.dataset.speed;
  $('#play').textContent = 'PAUSE';
  anim.frame = requestAnimationFrame(tick);
}

function stopReplay() {
  if (anim && anim.frame) cancelAnimationFrame(anim.frame);
  if (anim && anim.view) anim.view.dispose();
  if (anim) { anim.frame = null; anim.view = null; }
}

function sampleAt(path, t) {
  const f = Math.min(t / sampleDt, path.length - 1);
  const i = Math.floor(f), j = Math.min(i + 1, path.length - 1), u = f - i;
  const a = path[i], b = path[j];
  const p = [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  let heading;
  if (a.length > 3) heading = a[3] + Math.atan2(Math.sin(b[3] - a[3]), Math.cos(b[3] - a[3])) * u;
  else {   // runs stored before the heading was recorded: use the direction of travel
    const k0 = path[Math.max(0, i - 1)], k1 = path[Math.min(path.length - 1, i + 2)];
    heading = Math.atan2(k1[1] - k0[1], k1[0] - k0[0]);
  }
  return { x: p[0], y: p[1], v: p[2], heading, index: f };
}

function tick(now) {
  if (!anim) return;
  const canvas = $('#replay-canvas');
  if (!canvas) return;
  anim.dt = anim.last == null ? 0 : Math.min(.1, (now - anim.last) / 1000);
  if (anim.playing) anim.t = Math.min(anim.duration, anim.t + anim.dt * anim.speed);
  anim.last = now;
  if (anim.t >= anim.duration && anim.playing) { anim.playing = false; $('#play').textContent = 'REPLAY'; }
  draw(canvas);
  const scrub = $('#scrub');
  if (scrub && anim.duration) scrub.value = Math.round(anim.t / anim.duration * 1000);
  anim.frame = requestAnimationFrame(tick);
}

function draw(canvas) {
  const tr = geometry[anim.track];
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0b3a1f'; ctx.fillRect(0, 0, w, h);

  const xs = tr.points.map(p => p[0]), ys = tr.points.map(p => p[1]);
  const pad = tr.width * 1.2;
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const s = Math.min(w / (x1 - x0), h / (y1 - y0));
  const ox = (w - (x1 - x0) * s) / 2, oy = (h - (y1 - y0) * s) / 2;
  const X = x => ox + (x - x0) * s, Y = y => h - oy - (y - y0) * s;
  const trace = pts => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); };

  // kerbs: alternating red/white under the road edge (butt caps, or the dashes merge)
  ctx.lineJoin = 'round'; ctx.lineCap = 'butt';
  trace(tr.points); ctx.closePath();
  ctx.lineWidth = (tr.width + .3) * s; ctx.strokeStyle = '#f2f2ea'; ctx.setLineDash([]); ctx.stroke();
  ctx.strokeStyle = '#d8332a'; ctx.setLineDash([.6 * s, .6 * s]); ctx.stroke();
  ctx.setLineDash([]); ctx.lineWidth = tr.width * s; ctx.strokeStyle = '#2a3442'; ctx.stroke();
  ctx.setLineDash([.4 * s, .5 * s]); ctx.lineWidth = Math.max(1, .05 * s); ctx.strokeStyle = '#ffffff40'; ctx.stroke();
  ctx.setLineDash([]);

  // chequered start/finish line
  const [sx, sy] = tr.points[0], [nx, ny] = tr.points[1];
  const len = Math.hypot(nx - sx, ny - sy), tx = (nx - sx) / len, ty = (ny - sy) / len;
  ctx.save(); ctx.translate(X(sx), Y(sy)); ctx.rotate(-Math.atan2(ty, tx));
  const cell = tr.width / 8 * s;
  for (let i = 0; i < 8; i++) for (let r = 0; r < 2; r++) {
    ctx.fillStyle = (i + r) % 2 ? '#111' : '#fff';
    ctx.fillRect((r - 1) * cell / 2, (i - 4) * cell, cell / 2, cell);
  }
  ctx.restore();

  const kart = (p, fill, alpha) => {
    ctx.save(); ctx.globalAlpha = alpha; ctx.translate(X(p.x), Y(p.y)); ctx.rotate(-p.heading);
    const k = Math.max(s * .5, 9);
    ctx.beginPath(); ctx.moveTo(k * .8, 0); ctx.lineTo(-k * .5, k * .45); ctx.lineTo(-k * .5, -k * .45); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = '#07111d'; ctx.stroke(); ctx.restore();
  };

  if (anim.ghostPath) kart(sampleAt(anim.ghostPath, anim.t), '#edf5f5', .5);

  const path = anim.path;
  if (path.length > 1) {
    const end = Math.min(anim.t, anim.own) / sampleDt;
    ctx.globalAlpha = .18; ctx.lineWidth = Math.max(2, .12 * s); ctx.strokeStyle = '#d8ff38';
    trace(path); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.lineCap = 'round'; ctx.lineWidth = Math.max(2.5, .14 * s);
    for (let i = 1; i <= Math.min(end, path.length - 1); i++) {
      ctx.beginPath(); ctx.moveTo(X(path[i - 1][0]), Y(path[i - 1][1])); ctx.lineTo(X(path[i][0]), Y(path[i][1]));
      ctx.strokeStyle = speedColor(path[i][2]); ctx.stroke();
    }
    const me = sampleAt(path, Math.min(anim.t, anim.own));
    const lap = current.entry.laps[anim.track];
    if (lap && lap.time == null && anim.t >= anim.own) {
      ctx.strokeStyle = '#ff594d'; ctx.lineWidth = 3; const r = 10;
      ctx.beginPath(); ctx.moveTo(X(me.x) - r, Y(me.y) - r); ctx.lineTo(X(me.x) + r, Y(me.y) + r);
      ctx.moveTo(X(me.x) + r, Y(me.y) - r); ctx.lineTo(X(me.x) - r, Y(me.y) + r); ctx.stroke();
    } else kart(me, '#ff594d', 1);
    if (anim.view) anim.view.frame(me, anim.dt * anim.speed);
    $('#r-time').textContent = Math.min(anim.t, anim.own).toFixed(2);
    $('#r-speed').textContent = (anim.t >= anim.own ? 0 : me.v).toFixed(2);
  }
}

/* ------------------------------------------------------------ broadcast */
// One screen per circuit replays the lap record from a chase camera, on a loop.
// With no record yet, the camera cruises the empty circuit.

const tvs = {};
let tvLast = null, tv3d = false;
const HOLD = 3;   // seconds the "lap record" caption stays up before the lap restarts

function mountBroadcast() {
  $('#broadcast').innerHTML = TRACKS.map(t => `<article class="tv vacant" data-track="${t}" tabindex="0" role="button" aria-label="${t} lap record replay">
      <canvas></canvas>
      <div class="tv-top"><span class="tv-live"><i></i>ONBOARD</span><span class="tv-circuit">${t}</span></div>
      <div class="tv-map">${trackSvg(t, null, { pad: .8 })}</div>
      <div class="tv-speed"><b>0.0</b><small>m/s</small><div class="tv-gauge"><i></i></div></div>
      <div class="tv-flash"><span>LAP RECORD</span><b></b></div>
      <div class="tv-plate"><span class="tv-tag">LAP RECORD</span><span class="tv-name">Vacant</span>
        <span class="tv-clock"><b>—</b><small></small></span></div>
    </article>`).join('');
  const seen = new IntersectionObserver(items => items.forEach(it => { tvs[it.target.dataset.track].visible = it.isIntersecting; }));
  for (const el of document.querySelectorAll('.tv')) {
    const t = el.dataset.track, svg = el.querySelector('svg');
    const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    dot.setAttribute('r', '.55'); dot.setAttribute('class', 'tv-dot');
    if (svg) svg.append(dot);
    tvs[t] = { el, track: t, dot, t: 0, visible: true, record: null, path: null,
               view: tv3d ? Onboard.view(el.querySelector('canvas'), geometry[t]) : null,
               clock: el.querySelector('.tv-clock b'), speed: el.querySelector('.tv-speed b'),
               gauge: el.querySelector('.tv-gauge i') };
    el.classList.toggle('flat', !tvs[t].view);
    el.onclick = el.onkeydown = e => {
      if (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      if (tvs[t].record) openEntry(tvs[t].record.id, t);
      else openSubmit();
    };
    seen.observe(el);
  }
  requestAnimationFrame(broadcastTick);
}

async function updateBroadcast(records) {
  for (const t of TRACKS) {
    const tv = tvs[t], rec = records[t] || null;
    if (!tv || (tv.record ? tv.record.id : null) === (rec ? rec.id : null)) continue;
    if (!rec) {
      Object.assign(tv, { record: null, path: null, t: 0 });
    } else {
      let entry;
      try { entry = await getEntry(rec.id); } catch { continue; }
      const lap = entry.laps[t];
      if (!lap || !lap.path) continue;
      // Start each screen at a different point of the lap, so the wall is not three
      // copies of the same moment when one driver holds every record.
      Object.assign(tv, { record: rec, path: lap.path, t: TRACKS.indexOf(t) * 5 });
    }
    tv.el.classList.toggle('vacant', !rec);
    tv.el.querySelector('.tv-name').textContent = rec ? rec.name : 'Vacant · set the first lap';
    tv.el.querySelector('.tv-clock small').textContent = rec ? `/ ${rec.time.toFixed(3)}` : '';
    tv.el.querySelector('.tv-flash b').textContent = rec ? `${rec.time.toFixed(3)} s · ${rec.name}` : '';
  }
}

// Cruise the centre line, for circuits nobody has finished yet.
function cruise(track, dist) {
  const tr = geometry[track];
  if (!tr.cum) {
    tr.cum = [0];
    for (let i = 1; i <= tr.points.length; i++) {
      const a = tr.points[i - 1], b = tr.points[i % tr.points.length];
      tr.cum.push(tr.cum[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
  }
  const total = tr.cum[tr.cum.length - 1], d = dist % total;
  let i = 0;
  while (tr.cum[i + 1] < d) i++;
  const a = tr.points[i], b = tr.points[(i + 1) % tr.points.length], u = (d - tr.cum[i]) / (tr.cum[i + 1] - tr.cum[i] || 1);
  return { x: a[0] + (b[0] - a[0]) * u, y: a[1] + (b[1] - a[1]) * u, v: 0, heading: Math.atan2(b[1] - a[1], b[0] - a[0]) };
}

function broadcastTick(now) {
  const dt = tvLast == null ? 0 : Math.min(.1, (now - tvLast) / 1000);
  tvLast = now;
  // Pause behind dialogs and in background tabs: there is nobody watching.
  if (!document.hidden && !document.querySelector('dialog[open]'))
    for (const t of TRACKS) if (tvs[t].visible) stepTv(tvs[t], dt);
  requestAnimationFrame(broadcastTick);
}

function stepTv(tv, dt) {
  if (!geometry[tv.track]) return;
  tv.t += dt;
  let s, lapTime = null, done = false;
  if (tv.path) {
    const duration = (tv.path.length - 1) * sampleDt;
    if (tv.t > duration + HOLD) tv.t = 0;
    lapTime = Math.min(tv.t, duration);
    done = tv.t > duration;
    s = sampleAt(tv.path, lapTime);
  } else s = cruise(tv.track, tv.t * 1.6);
  if (tv.view) tv.view.frame(s, dt, !!tv.path);
  tv.dot.setAttribute('cx', s.x.toFixed(2));
  tv.dot.setAttribute('cy', (-s.y).toFixed(2));
  tv.el.classList.toggle('done', done);
  tv.clock.textContent = lapTime == null ? '—' : lapTime.toFixed(2);
  tv.speed.textContent = s.v.toFixed(1);
  tv.gauge.style.clipPath = `inset(0 ${(100 - Math.min(100, s.v / 4 * 100)).toFixed(1)}% 0 0)`;
}

/* ---------------------------------------------------------------- setup */

const MF = {
  tri: (x, a, b, c) => Math.max(0, Math.min(b > a ? (x - a) / (b - a) : +(x >= b), c > b ? (c - x) / (c - b) : +(x <= b), 1)),
  trap: (x, a, b, c, d) => Math.max(0, Math.min(b > a ? (x - a) / (b - a) : +(x >= b), d > c ? (d - x) / (d - c) : +(x <= c), 1)),
  gauss: (x, mu, sg) => Math.exp(-.5 * ((x - mu) / Math.abs(sg)) ** 2),
  bell: (x, a, b, c) => 1 / (1 + Math.abs((x - c) / a) ** (2 * b)),
  sigmoid: (x, b, c) => 1 / (1 + Math.exp(-b * (x - c))),
};

function varSvg(name, v) {
  const [lo, hi] = v.range, W = 300, H = 120, top = 18, bottom = 100;
  const X = x => 8 + (x - lo) / (hi - lo) * (W - 16), Y = m => bottom - m * (bottom - top);
  const terms = Object.entries(v.terms || {});
  const curves = terms.map(([t, spec], k) => {
    const f = MF[spec[0]], params = spec.slice(1);
    let d = '', peak = lo, best = -1;
    for (let i = 0; i <= 150; i++) {
      const x = lo + (hi - lo) * i / 150, m = f ? f(x, ...params) : 0;
      if (m > best + 1e-9) { best = m; peak = x; }
      d += `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(m).toFixed(1)}`;
    }
    const color = TERM_COLORS[k % TERM_COLORS.length];
    const lx = Math.max(24, Math.min(W - 24, X(peak)));
    return `<path d="${d}" fill="${color}22" stroke="${color}" stroke-width="1.6"/>
      <text x="${lx}" y="${top - 5}" fill="${color}" font-size="10" text-anchor="middle" font-family="DM Mono, monospace">${esc(t)}</text>`;
  }).join('');
  const ticks = [lo, ...(lo < 0 && hi > 0 ? [0] : []), hi].map(x =>
    `<text x="${X(x)}" y="${H - 4}" fill="#8ea3b5" font-size="10" text-anchor="${x === lo ? 'start' : x === hi ? 'end' : 'middle'}" font-family="DM Mono, monospace">${+x.toFixed(3)}</text>`).join('');
  return `<div class="var"><div class="vname"><b>${esc(name)}</b><span class="muted">[${v.range.join(', ')}]</span></div>
    <svg viewBox="0 0 ${W} ${H}" aria-label="membership functions of ${esc(name)}">
      <line x1="8" x2="${W - 8}" y1="${bottom}" y2="${bottom}" stroke="#22394f"/>${curves}${ticks}</svg></div>`;
}

function thenText(then, inputs) {
  if (typeof then === 'string') return `<em>${esc(then)}</em>`;
  if (typeof then === 'number') return `<em>${+then.toFixed(4)}</em>`;
  if (Array.isArray(then)) return `<em>${+then[0].toFixed(4)}${then.slice(1).map((c, i) =>
    ` ${c < 0 ? '−' : '+'} ${Math.abs(+c.toFixed(4))}·${esc(inputs[i])}`).join('')}</em>`;
  return `<em>order-${then.order} polynomial</em>`;
}

function renderSetup(spec) {
  return `<div class="systems">${['steering', 'throttle'].map(key => {
    const sys = spec[key], inputs = Object.keys(sys.inputs);
    const mamdani = typeof sys.rules[0].then === 'string';
    const ops = mamdani ? [`AND ${sys.and || 'min'}`, `implication ${sys.implication || 'min'}`,
      `aggregation ${sys.aggregation || 'max'}`, sys.defuzz || 'centroid'] : [`AND ${sys.and || 'min'}`, 'weighted average'];
    let rules;
    const full = inputs.length === 2 && sys.rules.every(r => Object.keys(r.if).length === 2);
    if (full) {
      const [a, b] = inputs, ta = Object.keys(sys.inputs[a].terms), tb = Object.keys(sys.inputs[b].terms);
      const cell = (x, y) => { const r = sys.rules.find(r => r.if[a] === x && r.if[b] === y); return r ? thenText(r.then, inputs) : '·'; };
      rules = `<table class="rules"><tr><th>${esc(a)} ↓ · ${esc(b)} →</th>${tb.map(t => `<th>${esc(t)}</th>`).join('')}</tr>
        ${ta.map(x => `<tr><th>${esc(x)}</th>${tb.map(y => `<td class="term">${cell(x, y)}</td>`).join('')}</tr>`).join('')}</table>`;
    } else {
      rules = `<ol class="rule-list">${sys.rules.map(r => `<li>IF ${Object.entries(r.if).map(([v, t]) =>
        `<b>${esc(v)}</b> is ${esc(t)}`).join(' AND ')} THEN ${esc(key)} = ${thenText(r.then, inputs)}</li>`).join('')}</ol>`;
    }
    return `<section class="system"><h3>${key[0].toUpperCase() + key.slice(1)}
        <small>${mamdani ? 'Mamdani' : 'TSK'} · ${sys.rules.length} rules · ${ops.map(esc).join(' · ')}</small></h3>
      <div class="vars">${inputs.map(n => varSvg(n, sys.inputs[n])).join('')}
        ${mamdani ? varSvg(`${key} (output)`, sys.output) : ''}</div>${rules}</section>`;
  }).join('')}</div>`;
}

function renderJson(body) {
  const text = pretty(current.entry.spec);
  body.innerHTML = `<pre class="json">${esc(text)}</pre>
    <div class="mini-actions"><button class="ghost" id="dl">Download</button><button class="ghost" id="cp">Copy</button>
      <button class="ghost" id="remix">Use as my starting point</button>
      ${isAdmin ? '<button class="danger" id="del">Delete entry</button>' : ''}</div>`;
  $('#dl').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text + '\n'], { type: 'application/json' }));
    a.download = `${current.entry.name.replace(/[^\w-]+/g, '_')}.json`; a.click();
    URL.revokeObjectURL(a.href);
  };
  $('#cp').onclick = async () => { try { await navigator.clipboard.writeText(text); toast('Copied to clipboard'); } catch { toast('Copy failed', 'err'); } };
  $('#remix').onclick = () => { $('#entry-dialog').close(); openSubmit(text); };
  const del = $('#del');
  if (del) del.onclick = async () => {
    if (!confirm(`Delete ${current.entry.name}'s run? This cannot be undone.`)) return;
    const res = await fetch(`/api/entries/${current.entry.id}?key=${encodeURIComponent(adminKey)}`, { method: 'DELETE' });
    if (res.ok) { entryCache.delete(current.entry.id); $('#entry-dialog').close(); poll(); } else toast('Delete failed', 'err');
  };
}

/* ---------------------------------------------------------------- wiring */

$('#open-submit').onclick = () => openSubmit();
$('#submit').onclick = submit;
$('#driver-json').oninput = checkJson;
$('#driver-json').onkeydown = e => {
  if (e.key === 'Tab') {   // indent instead of leaving the editor
    e.preventDefault();
    const t = e.target, s = t.selectionStart;
    t.setRangeText('  ', s, t.selectionEnd, 'end');
  }
};
$('#format-json').onclick = () => { const spec = checkJson(); if (spec) { $('#driver-json').value = pretty(spec); checkJson(); } };
$('#load-example').onclick = async () => {
  const text = await (await fetch('/api/example', { cache: 'no-store' })).text();
  $('#driver-json').value = text.trim(); checkJson();
};
$('#upload').onchange = async e => {
  const file = e.target.files[0];
  if (file) { $('#driver-json').value = await file.text(); checkJson(); }
  e.target.value = '';
};
$('#submit-dialog').addEventListener('close', () => clearTimeout(progressTimer));
$('#entry-dialog').addEventListener('close', stopReplay);
document.querySelectorAll('#entry-tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
document.addEventListener('click', e => {
  const target = e.target.closest('.kart[data-id], .srow[data-id], .rrow[data-id]');
  if (target) openEntry(target.dataset.id);
});

(async function init() {
  try {
    const t = await (await fetch('/api/tracks')).json();
    geometry = t.tracks; sampleDt = t.sample_dt;
  } catch { /* poll() reports the outage */ }
  // A slow CDN must not hold up the leaderboard: give three.js a few seconds.
  tv3d = await Promise.race([Onboard.ready, new Promise(ok => setTimeout(() => ok(false), 5000))]);
  mountBroadcast();
  if (adminKey) {
    try { isAdmin = (await (await fetch(`/api/admin?key=${encodeURIComponent(adminKey)}`)).json()).admin; } catch { }
  }
  await poll();
  setInterval(poll, 2000);
})();
