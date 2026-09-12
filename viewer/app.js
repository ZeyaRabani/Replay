// Replay Stage 3 viewer — loads a pitchworld tracking.json and renders it in Three.js.
//
// World mapping: pitch x (0..length, goal A -> goal B) -> three X,
//                pitch y (0..width, touch S -> touch N) -> three Z, up = three Y.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ReactorWorld } from './reactor.js';

const TRANSITION_S = 0.4;

const HEAD_HEIGHT = 1.7;
const PLAYER_RADIUS = 0.3;
const PLAYER_HEIGHT = 1.7;
const TEAM_COLOURS = { A: 0xe53e3e, B: 0x3182ce, home: 0xe53e3e, away: 0x3182ce, 0: 0xe53e3e, 1: 0x3182ce, ref: 0xf6e05e };
const ID_PALETTE = [0xf56565, 0x4299e1, 0x48bb78, 0xed8936, 0x9f7aea, 0xecc94b, 0x38b2ac, 0xed64a6, 0xa0aec0, 0x667eea,
  0xf687b3, 0x68d391, 0xfbd38d, 0x63b3ed, 0xb794f4, 0xfc8181];

// ---------------------------------------------------------------- state
const state = {
  data: null,
  frame: 0,
  playing: false,
  speed: 1,
  accum: 0,
  lastTs: 0,
  mode: 'orbit',          // 'orbit' | 'anchor' | 'player'
  anchorId: null,         // free anchor name or player id
  yaw: 0, pitch: 0,       // mouse-look offsets (radians)
  fov: 60,
  playerIds: [],
  playerIndex: new Map(), // id -> Map(frame -> player record)
  colours: new Map(),
  meshes: new Map(),      // id -> {group, capsule, label}
  pitchGroup: null,
  anchors: {},            // name -> {pos: Vector3, look: Vector3}
  camPos: new THREE.Vector3(),
  camTarget: new THREE.Vector3(),
  transition: null,       // {pos0, quat0, t} while blending into a new anchor
  clips: [],              // synced thumbnail <video>s, one per camera
};

// ---------------------------------------------------------------- three setup
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
scene.fog = new THREE.Fog(0x000000, 120, 260);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.position.set(25, 35, 60);
const controls = new OrbitControls(camera, canvas);
controls.target.set(25, 0, 15);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI / 2 - 0.02;

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x203020, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(30, 60, 20);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -60; sun.shadow.camera.right = 60;
sun.shadow.camera.top = 60; sun.shadow.camera.bottom = -60;
scene.add(sun);

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------- pitch
function buildPitch(p) {
  if (state.pitchGroup) scene.remove(state.pitchGroup);
  const g = new THREE.Group();
  const L = p.length, W = p.width, cy = W / 2;

  // surroundings so head-height views have a horizon: dark ground + low stands on all four sides
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600),
    new THREE.MeshStandardMaterial({ color: 0x0d120d, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(L / 2, -0.03, cy);
  ground.receiveShadow = true;
  g.add(ground);
  const standMat = new THREE.MeshStandardMaterial({ color: 0x15171c, roughness: 0.9 });
  for (const [sx, sz, sw, sd] of [[L / 2, -14, L + 40, 6], [L / 2, W + 14, L + 40, 6], [-14, cy, 6, W + 40], [L + 14, cy, 6, W + 40]]) {
    const stand = new THREE.Mesh(new THREE.BoxGeometry(sw, 6, sd), standMat);
    stand.position.set(sx, 3, sz);
    g.add(stand);
  }
  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(L + 8, W + 8),
    new THREE.MeshStandardMaterial({ color: 0x2f7d3a, roughness: 1 }));
  grass.rotation.x = -Math.PI / 2;
  grass.position.set(L / 2, -0.01, cy);
  grass.receiveShadow = true;
  g.add(grass);

  // alternating mowing stripes
  const stripes = 10;
  for (let i = 0; i < stripes; i++) {
    if (i % 2) continue;
    const s = new THREE.Mesh(new THREE.PlaneGeometry(L / stripes, W),
      new THREE.MeshStandardMaterial({ color: 0x3a9448, roughness: 1, transparent: true, opacity: 0.35 }));
    s.rotation.x = -Math.PI / 2;
    s.position.set((i + 0.5) * L / stripes, 0, cy);
    g.add(s);
  }

  const pts = [];
  const seg = (a, b) => pts.push(a[0], 0.02, a[1], b[0], 0.02, b[1]);
  seg([0, 0], [L, 0]); seg([L, 0], [L, W]); seg([L, W], [0, W]); seg([0, W], [0, 0]);
  seg([L / 2, 0], [L / 2, W]);
  for (const [depth, wd] of [[p.penalty_depth, p.penalty_width], [p.goal_area_depth, p.goal_area_width]]) {
    if (!(depth > 0 && wd > 0)) continue;
    const h = wd / 2;
    seg([0, cy - h], [depth, cy - h]); seg([depth, cy - h], [depth, cy + h]); seg([depth, cy + h], [0, cy + h]);
    seg([L, cy - h], [L - depth, cy - h]); seg([L - depth, cy - h], [L - depth, cy + h]); seg([L - depth, cy + h], [L, cy + h]);
  }
  const arc = (c, r, a0, a1) => {
    const n = 48;
    for (let i = 0; i < n; i++) {
      const t0 = THREE.MathUtils.degToRad(a0 + (a1 - a0) * i / n), t1 = THREE.MathUtils.degToRad(a0 + (a1 - a0) * (i + 1) / n);
      seg([c[0] + r * Math.cos(t0), c[1] + r * Math.sin(t0)], [c[0] + r * Math.cos(t1), c[1] + r * Math.sin(t1)]);
    }
  };
  if (p.d_radius > 0) { arc([0, cy], p.d_radius, -90, 90); arc([L, cy], p.d_radius, 90, 270); }
  if (p.centre_circle_radius > 0) arc([L / 2, cy], p.centre_circle_radius, 0, 360);
  if (p.penalty_depth > 0) {
    // penalty spots + arcs (standard pitch only)
    arc([11, cy], 9.15, -53, 53); arc([L - 11, cy], 9.15, 127, 233);
  }
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  g.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: 0xffffff })));

  // goals: two posts + crossbar + shallow net box
  const gw = p.goal_width, gh = gw > 5 ? 2.44 : 2.0, gd = gw > 5 ? 1.5 : 0.8;
  const postMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 });
  const netMat = new THREE.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.12, side: THREE.DoubleSide });
  for (const [x, dir] of [[0, -1], [L, 1]]) {
    const goal = new THREE.Group();
    for (const dz of [-gw / 2, gw / 2]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, gh, 12), postMat);
      post.position.set(x, gh / 2, cy + dz);
      goal.add(post);
    }
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, gw, 12), postMat);
    bar.rotation.x = Math.PI / 2;
    bar.position.set(x, gh, cy);
    goal.add(bar);
    const net = new THREE.Mesh(new THREE.BoxGeometry(gd, gh, gw), netMat);
    net.position.set(x + dir * gd / 2, gh / 2, cy);
    goal.add(net);
    g.add(goal);
  }

  // small labels for goal A / goal B
  g.add(makeTextSprite('GOAL A', '#ffffff', 0.06, undefined, false).translateX(0).translateY(gh + 1).translateZ(cy));
  g.add(makeTextSprite('GOAL B', '#ffffff', 0.06, undefined, false).translateX(L).translateY(gh + 1).translateZ(cy));

  state.pitchGroup = g;
  scene.add(g);

  // free anchors (pitch-level, head height, look towards centre)
  const centre = new THREE.Vector3(L / 2, 0, cy);
  const mk = (x, z) => ({ pos: new THREE.Vector3(x, HEAD_HEIGHT, z), look: centre.clone() });
  state.anchors = {
    behind_A: mk(-6, cy),
    behind_B: mk(L + 6, cy),
    centre: { pos: new THREE.Vector3(L / 2, HEAD_HEIGHT, cy), look: new THREE.Vector3(L, 0, cy) },
    touch_S: mk(L / 2, -4),
    touch_N: mk(L / 2, W + 4),
  };
  controls.target.copy(centre);
  camera.position.set(L / 2, Math.max(L, W) * 0.7, W + Math.max(L, W) * 0.9);
}

function makeTextSprite(text, colour, worldHeight, bg = 'rgba(0,0,0,0.55)', sizeAttenuation = true) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  ctx.font = 'bold 48px system-ui, sans-serif';
  const w = Math.ceil(ctx.measureText(text).width) + 32;
  c.width = w; c.height = 72;
  ctx.font = 'bold 48px system-ui, sans-serif';
  ctx.fillStyle = bg;
  roundRect(ctx, 0, 0, w, 72, 16);
  ctx.fill();
  ctx.fillStyle = colour;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 16, 38);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation }));
  sp.scale.set(worldHeight * w / 72, worldHeight, 1);
  return sp;
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// ---------------------------------------------------------------- data
function colourFor(id, team) {
  if (team !== undefined && team !== null && TEAM_COLOURS[team] !== undefined) return TEAM_COLOURS[team];
  const n = typeof id === 'number' ? id : [...String(id)].reduce((a, ch) => a + ch.charCodeAt(0), 0);
  return ID_PALETTE[Math.abs(n) % ID_PALETTE.length];
}

function loadTracking(data, label = 'tracking.json') {
  if (!data || !Array.isArray(data.frames)) { alert('Not a pitchworld tracking.json (missing "frames")'); return; }
  state.data = data;
  state.frame = 0;
  state.playing = false;
  const pitch = { length: 105, width: 68, goal_width: 7.32, d_radius: 0, penalty_depth: 16.5, penalty_width: 40.32,
    goal_area_depth: 5.5, goal_area_width: 18.32, centre_circle_radius: 9.15, ...(data.pitch || {}) };
  buildPitch(pitch);

  for (const m of state.meshes.values()) scene.remove(m.group);
  state.meshes.clear();
  state.playerIndex.clear();
  state.colours.clear();
  const teams = new Map();
  state.jerseys = new Map();
  data.frames.forEach((fr, i) => {
    for (const pl of fr.players || []) {
      if (!state.playerIndex.has(pl.id)) state.playerIndex.set(pl.id, new Map());
      state.playerIndex.get(pl.id).set(i, pl);
      if (pl.team !== undefined && pl.team !== null) teams.set(pl.id, pl.team);
      const j = pl.jersey ?? pl.number ?? pl.jersey_number;
      if (j !== undefined && j !== null) state.jerseys.set(pl.id, j);
    }
  });
  state.teams = teams;
  // longest-seen first: the ids a judge wants to jump to are the stable tracks
  state.playerIds = [...state.playerIndex.keys()].sort((a, b) => state.playerIndex.get(b).size - state.playerIndex.get(a).size || (a > b) - (a < b));
  for (const id of state.playerIds) {
    const col = colourFor(id, teams.get(id));
    state.colours.set(id, col);
    state.meshes.set(id, makePlayer(id, col));
  }

  const fps = data.fps || 25;
  document.getElementById('meta').textContent =
    `${label} · ${data.frames.length} frames @ ${fps} fps · ${pitch.length}×${pitch.width} m · ${state.playerIds.length} players`;
  const scrub = document.getElementById('scrub');
  scrub.max = Math.max(0, data.frames.length - 1);
  scrub.value = 0;
  buildPlayerList();
  buildThumbs(data);
  showQuality(data.quality);
  setMode('orbit');
  updateFrame();
}

// ---------------------------------------------------------------- synced clip thumbnails (generic N cameras)
function buildThumbs(data) {
  const el = document.getElementById('thumbs');
  el.innerHTML = '';
  state.clips = [];
  const cams = data.cameras || [];
  const media = new Set(state.media || []);
  cams.forEach((cam, i) => {
    const idx = cam.index ?? i;
    const base = cam.synced_clip ? cam.synced_clip.split('/').pop() : `cam${idx}.mp4`;
    const stem = base.replace(/\.[^.]+$/, '');
    const candidates = [`${stem}_thumb.mp4`, base].filter(n => media.has(n));
    if (!candidates.length) return;
    const box = document.createElement('div');
    box.className = 'thumb';
    const v = document.createElement('video');
    v.src = `media/${candidates[0]}`;
    v.muted = true; v.playsInline = true; v.preload = 'auto';
    box.appendChild(v);
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = `cam${idx}`;
    box.appendChild(tag);
    el.appendChild(box);
    state.clips.push(v);
  });
}
function syncThumbs(t, hard) {
  for (const v of state.clips) {
    if (state.playing && state.speed === 1) {
      if (v.paused) v.play().catch(() => {});
      if (Math.abs(v.currentTime - t) > 0.15) v.currentTime = t;
    } else {
      if (!v.paused) v.pause();
      if (hard || Math.abs(v.currentTime - t) > 0.04) v.currentTime = t;
    }
  }
}

function makePlayer(id, colour) {
  const group = new THREE.Group();
  const capsule = new THREE.Mesh(
    new THREE.CapsuleGeometry(PLAYER_RADIUS, PLAYER_HEIGHT - 2 * PLAYER_RADIUS, 6, 16),
    new THREE.MeshStandardMaterial({ color: colour, roughness: 0.6 }));
  capsule.position.y = PLAYER_HEIGHT / 2;
  capsule.castShadow = true;
  group.add(capsule);
  const ring = new THREE.Mesh(new THREE.RingGeometry(PLAYER_RADIUS + 0.05, PLAYER_RADIUS + 0.2, 32),
    new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity: 0.8 }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.03;
  group.add(ring);
  const label = makeTextSprite(String(id), '#ffffff', 0.6, '#' + colour.toString(16).padStart(6, '0') + 'cc');
  label.position.y = PLAYER_HEIGHT + 0.5;
  group.add(label);
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.05, 0), 1, colour, 0.3, 0.2);
  arrow.visible = false;
  group.add(arrow);
  group.visible = false;
  scene.add(group);
  return { group, capsule, label, arrow };
}

// finite-difference velocity in pitch coords (m/s), central window of ±k frames
function velocity(id, frame, k = 3) {
  const track = state.playerIndex.get(id);
  const fps = state.data.fps || 25;
  let a = null, b = null, fa = frame, fb = frame;
  for (let i = frame; i >= Math.max(0, frame - k); i--) if (track.has(i)) { a = track.get(i); fa = i; break; }
  for (let i = frame; i <= Math.min(state.data.frames.length - 1, frame + k); i++) if (track.has(i)) { b = track.get(i); fb = i; break; }
  if (!a || !b || fa === fb) {
    // fall back to a one-sided window
    for (let i = frame - 1; i >= Math.max(0, frame - 2 * k); i--) if (track.has(i)) { a = track.get(i); fa = i; break; }
    if (!a || !b || fa === fb) return new THREE.Vector3();
  }
  const dt = (fb - fa) / fps;
  return new THREE.Vector3((b.x - a.x) / dt, 0, (b.y - a.y) / dt);
}

// ---------------------------------------------------------------- per-frame update
const lastHeading = new Map();
function updateFrame() {
  const d = state.data;
  if (!d) return;
  const fr = d.frames[state.frame];
  const present = new Set();
  for (const pl of fr.players || []) {
    present.add(pl.id);
    const m = state.meshes.get(pl.id);
    if (!m) continue;
    m.group.visible = true;
    m.group.position.set(pl.x, 0, pl.y);
    const v = velocity(pl.id, state.frame);
    const speed = v.length();
    if (speed > 0.3) {
      lastHeading.set(pl.id, v.clone().normalize());
      m.arrow.visible = true;
      m.arrow.setDirection(v.clone().normalize());
      m.arrow.setLength(Math.min(3, 0.5 + speed * 0.3), 0.3, 0.2);
    } else {
      m.arrow.visible = false;
    }
  }
  for (const [id, m] of state.meshes) if (!present.has(id)) m.group.visible = false;

  const fps = d.fps || 25;
  const t = fr.t ?? state.frame / fps;
  document.getElementById('scrub').value = state.frame;
  document.getElementById('time').textContent =
    `${t.toFixed(2)} s / ${(d.duration_s ?? d.frames.length / fps).toFixed(2)} s · f ${state.frame}`;
  const playBtn = document.getElementById('play');
  playBtn.textContent = state.playing ? '❚❚' : '▶';
  playBtn.classList.toggle('playing', state.playing);
  updatePlayerListRows(fr);
  syncThumbs(t, !state.playing);
}

function updatePlayerListRows(fr) {
  const byId = new Map((fr.players || []).map(p => [p.id, p]));
  for (const row of document.querySelectorAll('#players .player')) {
    const id = row.dataset.id;
    const p = byId.get(coerceId(id));
    row.classList.toggle('absent', !p);
    row.querySelector('.pos').textContent = p ? `${p.x.toFixed(1)}, ${p.y.toFixed(1)}` : '—';
  }
}
function coerceId(s) {
  return state.playerIds.find(id => String(id) === s);
}

// ---------------------------------------------------------------- camera modes
function setMode(mode, anchorId = null) {
  const prev = { mode: state.mode, anchorId: state.anchorId };
  state.mode = mode;
  state.anchorId = anchorId;
  state.yaw = 0; state.pitch = 0;
  state.fov = 60;
  controls.enabled = mode === 'orbit';
  for (const b of document.querySelectorAll('.anchor')) {
    b.classList.toggle('active', mode === 'orbit' ? b.dataset.anchor === 'orbit' : mode === 'anchor' && b.dataset.anchor === anchorId);
  }
  for (const r of document.querySelectorAll('#players .player')) {
    r.classList.toggle('active', mode === 'player' && String(anchorId) === r.dataset.id);
  }
  const hud = document.getElementById('hud');
  if (mode === 'orbit') {
    hud.classList.add('hidden');
    if (state.data) { camera.fov = 60; camera.updateProjectionMatrix(); }
  } else {
    hud.classList.remove('hidden');
    hud.textContent = mode === 'player' ? `camera on player ${anchorId} (head height ${HEAD_HEIGHT} m, looking along velocity)` : `anchor: ${anchorId}`;
    // smooth ~0.4 s blend from the current camera pose into the new anchor
    const t = anchorPose();
    if (t) {
      state.transition = { pos0: camera.position.clone(), quat0: camera.quaternion.clone(), t: 0 };
      state.camPos.copy(t.pos); state.camTarget.copy(t.target);
    }
  }
  // hide the capsule we're sitting inside
  for (const [id, m] of state.meshes) m.capsule.visible = !(mode === 'player' && id === anchorId);
  if (mode !== 'orbit' && (prev.mode !== mode || prev.anchorId !== anchorId)) reactor.jumpTo(anchorPose(), prevPose(prev));
}

function prevPose(prev) {
  const saved = { mode: state.mode, anchorId: state.anchorId };
  state.mode = prev.mode; state.anchorId = prev.anchorId;
  const p = prev.mode === 'orbit' ? { pos: camera.position.clone(), target: controls.target.clone() } : anchorPose();
  state.mode = saved.mode; state.anchorId = saved.anchorId;
  return p;
}

function anchorPose() {
  if (state.mode === 'anchor') {
    const a = state.anchors[state.anchorId];
    if (!a) return null;
    return { pos: a.pos.clone(), target: a.look.clone() };
  }
  if (state.mode === 'player') {
    const track = state.playerIndex.get(state.anchorId);
    if (!track) return null;
    let pl = track.get(state.frame);
    if (!pl) { // nearest earlier frame
      for (let i = state.frame; i >= 0 && !pl; i--) pl = track.get(i);
      if (!pl) return null;
    }
    const pos = new THREE.Vector3(pl.x, HEAD_HEIGHT, pl.y);
    let dir = lastHeading.get(state.anchorId);
    if (!dir) {
      const v = velocity(state.anchorId, state.frame);
      const { length: L, width: W } = state.data.pitch;
      const toCentre = new THREE.Vector3(L / 2 - pl.x, 0, W / 2 - pl.y).normalize();
      // face the run direction when clearly moving and it keeps the pitch in view; otherwise face the centre
      const ahead = v.clone().normalize().multiplyScalar(8);
      const onPitch = pl.x + ahead.x > -2 && pl.x + ahead.x < L + 2 && pl.y + ahead.z > -2 && pl.y + ahead.z < W + 2;
      dir = v.lengthSq() > 1 && onPitch ? v.normalize() : toCentre;
    }
    return { pos, target: pos.clone().add(dir.clone().multiplyScalar(10).setY(-0.6)) };
  }
  return null;
}

function applyCamera(dt) {
  if (state.mode === 'orbit') { controls.update(); return; }
  const t = anchorPose();
  if (!t) return;
  const k = state.mode === 'player' ? 1 - Math.exp(-dt * 12) : 1;
  state.camPos.lerp(t.pos, k);
  state.camTarget.lerp(t.target, k);
  // base direction, then apply mouse-look yaw/pitch offsets
  const base = state.camTarget.clone().sub(state.camPos);
  const baseYaw = Math.atan2(base.x, base.z);
  const basePitch = Math.atan2(base.y, Math.hypot(base.x, base.z));
  const yaw = baseYaw + state.yaw, pitch = THREE.MathUtils.clamp(basePitch + state.pitch, -1.4, 1.4);
  const look = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
  const goalQuat = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().lookAt(state.camPos, state.camPos.clone().add(look), camera.up));
  const tr = state.transition;
  if (tr) {
    tr.t += dt;
    const s = THREE.MathUtils.smoothstep(tr.t / TRANSITION_S, 0, 1);
    camera.position.lerpVectors(tr.pos0, state.camPos, s);
    camera.quaternion.slerpQuaternions(tr.quat0, goalQuat, s);
    if (tr.t >= TRANSITION_S) state.transition = null;
  } else {
    camera.position.copy(state.camPos);
    camera.quaternion.copy(goalQuat);
  }
  if (camera.fov !== state.fov) { camera.fov = state.fov; camera.updateProjectionMatrix(); }
}

// mouse-look
let dragging = false, lx = 0, ly = 0;
const worldVideo = document.getElementById('world');
for (const surf of [canvas, worldVideo]) {
  surf.addEventListener('pointerdown', e => { if (state.mode !== 'orbit' || surf === worldVideo) { dragging = true; lx = e.clientX; ly = e.clientY; surf.setPointerCapture(e.pointerId); } });
  surf.addEventListener('pointermove', e => {
    if (!dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    state.yaw -= dx * 0.004;
    state.pitch -= dy * 0.004;
    lx = e.clientX; ly = e.clientY;
    reactor.lookDrag(dx, dy);
  });
  surf.addEventListener('pointerup', () => { dragging = false; reactor.lookRelease(); });
}
canvas.addEventListener('wheel', e => {
  if (state.mode === 'orbit') return;
  e.preventDefault();
  state.fov = THREE.MathUtils.clamp(state.fov + e.deltaY * 0.03, 20, 110);
}, { passive: false });

// ---------------------------------------------------------------- UI wiring
function buildPlayerList() {
  const el = document.getElementById('players');
  el.innerHTML = '';
  for (const id of state.playerIds) {
    const row = document.createElement('div');
    row.className = 'player';
    row.dataset.id = String(id);
    const col = '#' + state.colours.get(id).toString(16).padStart(6, '0');
    const frames = state.playerIndex.get(id).size;
    const team = state.teams.get(id), jersey = state.jerseys.get(id);
    const extra = [team !== undefined ? `team ${team}` : null, jersey !== undefined ? `№${jersey}` : null, `${frames}f`].filter(Boolean).join(' · ');
    row.innerHTML = `<span class="swatch" style="background:${col}"></span><span class="pid">#${id}</span>
      <span class="muted">${extra}</span><span class="pos">—</span>`;
    row.title = `attach camera to player ${id}`;
    row.addEventListener('click', () => setMode(state.mode === 'player' && state.anchorId === id ? 'orbit' : 'player', id));
    el.appendChild(row);
  }
  document.getElementById('player-count').textContent = `(${state.playerIds.length})`;
}

function showQuality(q) {
  const b = document.getElementById('banner');
  if (!q) { b.classList.add('hidden'); return; }
  const items = [];
  if (q.calibration_low_confidence) items.push('calibration_low_confidence — player positions may be off by metres');
  if (q.sync_low_confidence) items.push('sync_low_confidence — cameras may not be time-aligned');
  const dis = q.cross_camera_disagreement_m;
  if (dis && typeof dis === 'object') {
    for (const [k, v] of Object.entries(dis)) if (typeof v === 'number' && v > 1) items.push(`cross-camera disagreement ${k}: ${v.toFixed(2)} m`);
  } else if (typeof dis === 'number' && dis > 1) items.push(`cross-camera disagreement: ${dis.toFixed(2)} m`);
  for (const w of q.warnings || []) items.push(w);
  if (!items.length) { b.classList.add('hidden'); return; }
  b.classList.remove('hidden');
  b.classList.toggle('danger', !!q.calibration_low_confidence);
  b.innerHTML = `<strong>${q.calibration_low_confidence ? 'LOW CONFIDENCE calibration' : 'Quality warnings'}</strong> — from tracking.json["quality"]<ul>${items.map(i => `<li>${escapeHtml(i)}</li>`).join('')}</ul>`;
}
function escapeHtml(s) { return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

document.querySelectorAll('.anchor').forEach(b => b.addEventListener('click', () => {
  const a = b.dataset.anchor;
  if (!state.data) return;
  a === 'orbit' ? setMode('orbit') : setMode('anchor', a);
}));

const scrub = document.getElementById('scrub');
scrub.addEventListener('input', () => { state.frame = +scrub.value; updateFrame(); });
document.getElementById('play').addEventListener('click', togglePlay);
document.getElementById('speed').addEventListener('change', e => { state.speed = +e.target.value; });
function togglePlay() { if (!state.data) return; state.playing = !state.playing; state.accum = 0; updateFrame(); }
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.code === 'Escape') setMode('orbit');
  else if (e.code === 'ArrowRight' && state.data) { state.frame = Math.min(state.data.frames.length - 1, state.frame + 1); updateFrame(); }
  else if (e.code === 'ArrowLeft' && state.data) { state.frame = Math.max(0, state.frame - 1); updateFrame(); }
  else if (e.code === 'Home' && state.data) { state.frame = 0; updateFrame(); }
});

// file loading
document.getElementById('file').addEventListener('change', e => { const f = e.target.files[0]; if (f) readFile(f); });
const drop = document.getElementById('drop');
window.addEventListener('dragover', e => { e.preventDefault(); drop.classList.remove('hidden'); });
window.addEventListener('dragleave', e => { if (!e.relatedTarget) drop.classList.add('hidden'); });
window.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.add('hidden');
  const f = e.dataTransfer.files[0];
  if (f) readFile(f);
});
function readFile(f) {
  f.text().then(t => loadTracking(JSON.parse(t), f.name)).catch(err => alert('Failed to parse JSON: ' + err));
}
async function fetchJson(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    loadTracking(await r.json(), url);
  } catch (err) { alert(`Failed to load ${url}: ${err}`); }
}
const params = new URLSearchParams(location.search);
const src = params.get('src');

// ---------------------------------------------------------------- Reactor live world (falls back to Three.js)
function setModeBadge(kind, text) {
  const m = document.getElementById('mode');
  m.classList.toggle('live', kind === 'live');
  m.classList.toggle('busy', kind === 'busy');
  document.getElementById('mode-text').textContent = text;
  worldVideo.classList.toggle('hidden', kind !== 'live');
  document.getElementById('reactor-status').textContent = reactor.statusLine();
}
const reactor = new ReactorWorld({
  video: worldVideo,
  worldId: params.get('world'),
  onStatus: (kind, text) => setModeBadge(kind, text),
});
document.getElementById('reactor-attach').addEventListener('click', () => reactor.attach(prompt('encrypted_world_id', reactor.worldId || '') || null));
document.getElementById('reactor-create').addEventListener('click', () => reactor.create(state.data));
document.getElementById('reactor-stop').addEventListener('click', () => reactor.shutdown('stopped by user'));

(async () => {
  let cfg = { reactor: false, media: [] };
  try { cfg = await (await fetch('config')).json(); } catch { /* plain static hosting */ }
  state.media = cfg.media || [];
  reactor.configured = cfg.reactor;
  reactor.seedUrl = params.get('seed') || cfg.seed_url || null;
  if (src) await fetchJson(src);
  else if (cfg.media.includes('tracking.json')) await fetchJson('media/tracking.json');
  else buildPitch({ length: 50, width: 30, goal_width: 3.66, d_radius: 6, penalty_depth: 0, penalty_width: 0, goal_area_depth: 0, goal_area_width: 0, centre_circle_radius: 0 });
  if (cfg.reactor && params.get('world')) reactor.attach(params.get('world'));
  else if (cfg.reactor && params.has('create')) reactor.create(state.data);
  else setModeBadge('fallback', cfg.reactor ? '3D fallback (Three.js) — Reactor key present, no ?world=<id>' : '3D fallback (Three.js) — Reactor not configured');
})();

// ---------------------------------------------------------------- loop
function tick(ts) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, (ts - state.lastTs) / 1000 || 0);
  state.lastTs = ts;
  if (state.playing && state.data) {
    const fps = state.data.fps || 25;
    state.accum += dt * state.speed;
    const step = Math.floor(state.accum * fps);
    if (step > 0) {
      state.accum -= step / fps;
      state.frame += step;
      if (state.frame >= state.data.frames.length) state.frame = 0;
      updateFrame();
    }
  }
  applyCamera(dt);
  renderer.render(scene, camera);
}
requestAnimationFrame(tick);
