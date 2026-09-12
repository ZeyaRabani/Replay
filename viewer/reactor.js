// Reactor live-world layer for the Replay viewer.
//
// Uses the HappyOyster typed SDK (the only Reactor model with *persistent* worlds) loaded at runtime
// from esm.sh so the viewer stays build-free. The JWT is fetched from our own server (`POST /token`),
// which is the only place the API key lives. Every failure path calls `shutdown()` so the Three.js
// render underneath always remains the fallback.
//
// Anchor jumps are APPROXIMATE in Reactor mode: HappyOyster Adventure only exposes held
// move/look controls (Front/Back/Left/Right, Mouse_*), not absolute camera poses, so a jump is
// translated into a short burst: turn toward the target for a duration proportional to the yaw
// delta, then move Front for a duration proportional to the distance (~3 m/s assumed), then turn to
// the anchor's look direction. There is no ground-truth camera in a world model, so drift accumulates.

const SDK_URL = 'https://esm.sh/@reactor-models/happy-oyster@1.0.0';
const MODEL_SLUG = 'reactor/happy-oyster-adventure';
const TURN_RATE = Math.PI / 2;   // rad/s the model turns while Mouse_Left/Right is held (guess)
const WALK_SPEED = 3;            // m/s while Front is held (guess)
const MAX_JUMP_S = 6;

export class ReactorWorld {
  constructor({ video, worldId, seedUrl, onStatus }) {
    this.video = video;
    this.worldId = worldId || null;
    this.seedUrl = seedUrl || null;
    this.onStatus = onStatus;
    this.configured = false;
    this.model = null;
    this.live = false;
    this.phase = 'idle';
    this.error = null;
    this.busy = Promise.resolve();
    this.dragAxis = null;
  }

  statusLine() {
    if (!this.configured) return 'Reactor: not configured (set REACTOR_API_KEY and run viewer/server.py)';
    if (this.live) return `Reactor: LIVE ${MODEL_SLUG} · world ${short(this.worldId)}`;
    if (this.error) return `Reactor: ${this.error}`;
    return `Reactor: key present · ${this.phase}${this.worldId ? ' · world ' + short(this.worldId) : ''}`;
  }

  _status(kind, text) { this.phase = text; this.onStatus?.(kind, text); }

  async _connect() {
    if (this.model) return this.model;
    this._status('busy', 'Reactor: fetching token…');
    const r = await fetch('token', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: MODEL_SLUG }) });
    const tok = await r.json();
    if (!r.ok || !tok.jwt) throw new Error(tok.error || `token ${r.status}`);
    this._status('busy', 'Reactor: loading SDK…');
    const { HappyOysterModel } = await import(SDK_URL);
    const model = new HappyOysterModel({ mode: 'adventure', videoElement: this.video });
    model.onWorldState?.((s) => {
      if (s?.encrypted_world_id) this.worldId = s.encrypted_world_id;
      if (s?.phase && !this.live) this._status('busy', `Reactor: world ${s.phase}`);
      if (s?.phase === 'failed') this.shutdown('world build failed');
    });
    this._status('busy', 'Reactor: connecting (waiting for GPU)…');
    await model.connect(tok.jwt);
    this.model = model;
    return model;
  }

  async attach(worldId) {
    if (!worldId) return;
    this.worldId = worldId;
    return this._run(async () => {
      const model = await this._connect();
      this._status('busy', 'Reactor: attachWorld…');
      await model.attachWorld(worldId);
      await this._travel();
    });
  }

  async create(tracking) {
    return this._run(async () => {
      const model = await this._connect();
      // HappyOyster fetches the seed itself, so a public URL (?seed= or SEED_IMAGE_URL) is the reliable
      // path; the upload-based Blob path is tried when no URL is configured.
      let image;
      if (this.seedUrl) {
        image = { firstFrameImageUrl: this.seedUrl };
      } else {
        this._status('busy', 'Reactor: fetching seed image…');
        const seed = await fetch('media/seed.jpg');
        if (!seed.ok) throw new Error('media/seed.jpg missing (pitchworld reactor_export makes one)');
        image = { firstFrameImage: await seed.blob() };
      }
      const p = tracking?.pitch || {};
      const prompt = `First-person view standing on a ${p.length || 50} x ${p.width || 30} metre artificial-turf football pitch at dusk, ` +
        'white pitch lines, small goals with nets, floodlights on, amateur players in orange and yellow bibs. ' +
        'The camera moves freely around the pitch; the pitch, goals and buildings stay fixed. Photorealistic, handheld video look.';
      this._status('busy', 'Reactor: createWorld (can take a minute)…');
      const world = await model.createWorld({ prompt, ...image, perspective: 'first_person' });
      this.worldId = world?.encrypted_world_id || this.worldId;
      if (this.worldId) {
        const u = new URL(location.href); u.searchParams.set('world', this.worldId); history.replaceState(null, '', u);
      }
      await this._travel();
    });
  }

  async _travel() {
    this._status('busy', 'Reactor: startTravel…');
    const res = await this.model.startTravel();
    if (res && res.streaming === false) throw new Error('startTravel refused (travel budget?)');
    this.live = true;
    this.error = null;
    this._status('live', `LIVE · Reactor ${MODEL_SLUG.split('/')[1]} · world ${short(this.worldId)}`);
    this.video.play?.().catch(() => {});
  }

  _run(fn) {
    if (!this.configured) { this._status('fallback', '3D fallback (Three.js) — Reactor not configured'); return; }
    this.busy = this.busy.then(fn).catch((e) => this.shutdown(String(e?.message || e)));
    return this.busy;
  }

  async shutdown(reason) {
    this.live = false;
    this.error = reason || null;
    const m = this.model; this.model = null;
    try { await m?.stop?.(); } catch { /* ignore */ }
    try { await m?.disconnect?.(); } catch { /* ignore */ }
    this._status('fallback', `3D fallback (Three.js)${reason ? ' — Reactor: ' + reason : ''}`);
  }

  // ---- approximate anchor jump: yaw toward target, walk, yaw to look direction
  jumpTo(to, from) {
    if (!this.live || !to || !from) return;
    const m = this.model;
    const dx = to.pos.x - from.pos.x, dz = to.pos.z - from.pos.z;
    const dist = Math.hypot(dx, dz);
    const fromYaw = Math.atan2(from.target.x - from.pos.x, from.target.z - from.pos.z);
    const toYaw = Math.atan2(to.target.x - to.pos.x, to.target.z - to.pos.z);
    const travelYaw = dist > 0.5 ? Math.atan2(dx, dz) : toYaw;
    const steps = [];
    const turn = (a, b) => {
      const d = wrap(b - a);
      if (Math.abs(d) < 0.05) return;
      steps.push({ rotation: d > 0 ? 'Mouse_Left' : 'Mouse_Right', ms: Math.min(3000, Math.abs(d) / TURN_RATE * 1000) });
    };
    turn(fromYaw, travelYaw);
    if (dist > 0.5) steps.push({ translation: 'Front', ms: Math.min(MAX_JUMP_S * 1000, dist / WALK_SPEED * 1000) });
    turn(travelYaw, toYaw);
    this.busy = this.busy.then(async () => {
      for (const s of steps) {
        if (!this.live) return;
        await m.hold({ translation: s.translation || 'None', rotation: s.rotation || 'None' });
        await sleep(s.ms);
      }
      await m.stop();
    }).catch(() => {});
  }

  // ---- mouse-look while dragging on the live video: hold a look axis, release on pointer-up
  lookDrag(dx, dy) {
    if (!this.live) return;
    const h = Math.abs(dx) > 2 ? (dx < 0 ? 'Left' : 'Right') : '';
    const v = Math.abs(dy) > 2 ? (dy < 0 ? 'Up' : 'Down') : '';
    const axis = h ? `Mouse_${h}` : v ? `Mouse_${v}` : null;  // single axis: diagonal names not verified
    if (axis && axis !== this.dragAxis) { this.dragAxis = axis; this.model?.look(axis).catch?.(() => {}); }
  }
  lookRelease() {
    if (!this.live || !this.dragAxis) return;
    this.dragAxis = null;
    this.model?.release({ rotation: true }).catch?.(() => {});
  }
}

function wrap(a) { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; }
function short(id) { return id ? id.slice(0, 8) + '…' : '—'; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
