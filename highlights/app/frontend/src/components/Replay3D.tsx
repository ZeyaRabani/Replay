import { ChevronDown, ChevronRight, Download, Maximize2, Pause, Play,
         Volume2, VolumeX } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { useProjectApi } from "../api";
import { DEFAULT_PITCH, pitchShapes } from "../lib/pitch";
import { fetchPlayers, usePlayerIdentities } from "../lib/players";
import { fmtClock } from "../lib/time";
import { fromOut, inReplay, toOut } from "../lib/timemap";
import type { Candidate, DirectorFull, DirectorSegment, PitchDims,
              PlayerIdentity, PlayersPaths, ReplayInfo } from "../types";
import { posAt } from "./RadarReplay";

const card = "card p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnGhost =
  "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";
const chip =
  "rounded-full px-2.5 py-0.5 text-[11px] bg-zinc-800 text-zinc-300 hover:bg-zinc-700";

const TEAM: Record<string, number> = { A: 0x22c55e, B: 0xf97316 };
const TEAM_HEX: Record<string, string> = { A: "#22c55e", B: "#f97316" };

const hexNum = (h?: string | null): number | undefined =>
  h ? parseInt(h.slice(1), 16) : undefined;
const lighten = (h: string, f = 0.45): string => {
  const n = parseInt(h.slice(1), 16);
  const m = (c: number) => Math.round(c + (255 - c) * f);
  return `#${((1 << 24) | (m(n >> 16) << 16) | (m((n >> 8) & 255) << 8)
    | m(n & 255)).toString(16).slice(1)}`;
};
const NO_IDENT_RING = 0x9ca3af;
const CAM_MODES: [string, string][] = [
  ["follow", "Follow the play"],
  ["broadcast", "Broadcast"],
  ["tactical", "Tactical"],
  ["goalA", "Goal cam left"],
  ["goalB", "Goal cam right"],
  ["free", "Free orbit"],
];

type LivePlayer = {
  id: string; team: string; label: string | null; ident: boolean;
  xy: [number, number]; bridged: boolean;
};
type Seg = { start: number; end: number; first: [number, number]; last: [number, number] };
type SegMap = Map<string, { team: string; segs: Seg[] }>;

/** Per identity: sorted list of its visible tracks' [start,end] segments. */
function identitySegments(paths: PlayersPaths): SegMap {
  const m: SegMap = new Map();
  for (const tr of paths.tracks) {
    if (tr.hidden || !tr.identity_id || !tr.pts.length) continue;
    const p0 = tr.pts[0], p1 = tr.pts[tr.pts.length - 1];
    const e = m.get(tr.identity_id) ?? { team: tr.team ?? "A", segs: [] };
    e.segs.push({ start: p0[0], end: p1[0],
                  first: [p0[1], p0[2]], last: [p1[1], p1[2]] });
    m.set(tr.identity_id, e);
  }
  for (const e of m.values()) e.segs.sort((a, b) => a.start - b.start);
  return m;
}

/** World mapping: pitch x 0..L -> three x, pitch y 0..W -> three -z, z up -> three y. */
const V = (x: number, y: number, z = 0) => new THREE.Vector3(x, z, -y);

function makeLabel(text: string, color: string): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const x = c.getContext("2d")!;
  x.font = "bold 40px system-ui";
  x.textAlign = "center";
  const w = x.measureText(text).width;
  x.fillStyle = "rgba(0,0,0,.55)";
  x.beginPath();
  if (typeof x.roundRect === "function") x.roundRect(128 - w / 2 - 14, 6, w + 28, 52, 12);
  else x.fillRect(128 - w / 2 - 14, 6, w + 28, 52);
  x.fill();
  x.fillStyle = color;
  x.fillText(text, 128, 46);
  const sp = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  sp.scale.set(2.4, 0.6, 1);
  return sp;
}

interface PlayerRig extends THREE.Group {
  userData: {
    body: THREE.Group; lL: THREE.Group; rL: THREE.Group;
    lA: THREE.Group; rA: THREE.Group; team: string;
    prev: { x: number; z: number } | null;
    sm: { x: number; z: number } | null;
    phase: number; heading: number; speed: number; kick: number;
    lbl: string | null;
  };
}

function makePlayer(team: string, name: string | null, ident: boolean,
                    kitHex: Record<string, number>,
                    lhex: Record<string, string>): PlayerRig {
  const g = new THREE.Group() as PlayerRig;
  const col = kitHex[team] ?? 0x9ca3af;
  const kit = new THREE.MeshStandardMaterial({ color: col });
  const skin = new THREE.MeshStandardMaterial({ color: 0xd9b38c });
  const white = new THREE.MeshStandardMaterial({ color: 0xf3f4f6 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x111827 });
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.45, 4, 12), kit);
  torso.position.y = 1.15; torso.scale.z = 0.6;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), skin);
  head.position.y = 1.65;
  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.135, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2), dark);
  hair.position.y = 1.66;
  const mkLimb = (len: number, r: number, mat: THREE.Material) => {
    const p = new THREE.Group();
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.8, len, 8), mat);
    m.position.y = -len / 2; p.add(m); return p;
  };
  const lL = mkLimb(0.85, 0.08, white), rL = mkLimb(0.85, 0.08, white);
  lL.position.set(0.12, 0.85, 0); rL.position.set(-0.12, 0.85, 0);
  const lA = mkLimb(0.6, 0.055, skin), rA = mkLimb(0.6, 0.055, skin);
  lA.position.set(0.27, 1.4, 0); rA.position.set(-0.27, 1.4, 0);
  for (const l of [lL, rL]) {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.26), dark);
    b.position.set(0, -0.85, 0.05); l.add(b);
  }
  const body = new THREE.Group();
  body.add(torso, head, hair, lL, rL, lA, rA);
  body.traverse((o) => { if ((o as THREE.Mesh).isMesh) o.castShadow = true; });
  g.add(body);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.4, 0.52, 32),
    new THREE.MeshBasicMaterial({
      color: ident ? col : NO_IDENT_RING, transparent: true, opacity: ident ? 0.7 : 0.4,
    }));
  body.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) (m.material as THREE.Material).transparent = true;
  });
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.02; g.add(ring);
  if (name) {
    const lab = makeLabel(name, lhex[team] ?? "#e4e4e7");
    lab.position.y = 2.15; g.add(lab);
  }
  g.userData = { body, lL, rL, lA, rA, team, prev: null, sm: null, phase: 0, heading: 0,
                 speed: 0, kick: 0, lbl: name ?? null };
  return g;
}

function updatePlayer(g: PlayerRig, xy: [number, number], dt: number, ballPos: THREE.Vector3 | null) {
  const u = g.userData, tx = xy[0], tz = -xy[1];
  if (!u.sm || dt === 0) u.sm = { x: tx, z: tz };
  else {
    const k = 1 - Math.exp(-dt * 6);
    u.sm.x += (tx - u.sm.x) * k;
    u.sm.z += (tz - u.sm.z) * k;
  }
  const x = u.sm.x, z = u.sm.z;
  if (u.prev && dt > 0) {
    const vx = (x - u.prev.x) / dt, vz = (z - u.prev.z) / dt;
    const sp = Math.hypot(vx, vz);
    u.speed += (Math.min(sp, 9) - u.speed) * Math.min(1, dt * 6);
    if (sp > 0.6) {
      const h = Math.atan2(vx, vz);
      let d = h - u.heading;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      u.heading += d * Math.min(1, dt * 8);
    } else if (ballPos) {
      const h = Math.atan2(ballPos.x - x, ballPos.z - z);
      let d = h - u.heading;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      u.heading += d * Math.min(1, dt * 3);
    }
  }
  u.prev = { x, z };
  g.position.set(x, 0, z);
  u.body.rotation.y = u.heading;
  const stride = u.speed > 0.4 ? Math.min(1, u.speed / 6) : 0;
  u.phase += dt * (4 + u.speed * 1.6);
  const sw = Math.sin(u.phase) * 0.9 * stride;
  if (ballPos && Math.hypot(ballPos.x - x, ballPos.z - z) < 1.3 && u.kick <= 0) u.kick = 0.35;
  u.kick = Math.max(0, u.kick - dt);
  const k = u.kick > 0 ? Math.sin((u.kick / 0.35) * Math.PI) * 1.3 : 0;
  u.lL.rotation.x = sw; u.rL.rotation.x = -sw - k;
  u.lA.rotation.x = -sw * 0.8; u.rA.rotation.x = sw * 0.8;
  u.body.position.y = Math.abs(Math.sin(u.phase)) * 0.05 * stride;
  u.body.rotation.x = stride * 0.15;
}

function buildPitch(group: THREE.Group, pitch: PitchDims) {
  group.clear();
  const L = pitch.len_m, W = pitch.wid_m;
  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(L + 12, W + 12),
    new THREE.MeshStandardMaterial({ color: 0x2f8f3a }));
  grass.rotation.x = -Math.PI / 2;
  grass.position.set(L / 2, 0, -W / 2);
  grass.receiveShadow = true; group.add(grass);
  for (let i = 0; i < 8; i++) {
    const s = new THREE.Mesh(
      new THREE.PlaneGeometry(L / 8, W),
      new THREE.MeshStandardMaterial({ color: i % 2 ? 0x2f8f3a : 0x35a043 }));
    s.rotation.x = -Math.PI / 2;
    s.position.set(L / 16 + (i * L) / 8, 0.005, -W / 2);
    s.receiveShadow = true; group.add(s);
  }
  const lm = new THREE.LineBasicMaterial({ color: 0xffffff });
  const faint = new THREE.LineBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.5 });
  const dotM = new THREE.MeshBasicMaterial({ color: 0xffffff });
  for (const sh of pitchShapes(pitch)) {
    if (sh.k === "dot") {
      const dm = new THREE.Mesh(new THREE.CircleGeometry(0.15, 16), dotM);
      dm.rotation.x = -Math.PI / 2; dm.position.set(sh.x, 0.02, -sh.y);
      group.add(dm);
      continue;
    }
    const pts = sh.closed ? [...sh.pts, sh.pts[0]] : sh.pts;
    group.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(
        pts.map((p) => new THREE.Vector3(p[0], 0.02, -p[1]))),
      sh.faint ? faint : lm));
  }
  const small = pitch.template === "small";
  const gw = pitch.goal_w_m ?? (small ? 3.66 : 7.32);
  const gh = small ? 2.0 : 2.44;
  const nd = small ? 1.0 : 1.5;
  for (const [x0, d] of [[0, 1], [L, -1]] as [number, number][]) {
    const gm = new THREE.MeshStandardMaterial({ color: 0xffffff });
    for (const yy of [W / 2 - gw / 2, W / 2 + gw / 2]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, gh), gm);
      p.position.set(x0, gh / 2, -yy); group.add(p);
    }
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, gw), gm);
    bar.rotation.x = Math.PI / 2; bar.position.set(x0, gh, -W / 2); group.add(bar);
    const net = new THREE.Mesh(
      new THREE.BoxGeometry(nd, gh, gw),
      new THREE.MeshStandardMaterial({
        color: 0xffffff, transparent: true, opacity: 0.18, side: THREE.DoubleSide,
      }));
    net.position.set(x0 - d * nd / 2, gh / 2, -W / 2); group.add(net);
  }
}

/** Players alive at shared t, deduped by identity_id (mean over its tracks). */
function livePlayers(paths: PlayersPaths, t: number,
                     labels: Record<string, string | null>,
                     segs: SegMap, showIds: boolean): LivePlayer[] {
  const byIdent = new Map<string, LivePlayer & { n: number }>();
  const out: (LivePlayer & { n?: number })[] = [];
  for (const tr of paths.tracks) {
    if (tr.hidden) continue;
    const p = posAt(tr.pts, t);
    if (!p || p[2] < 1) continue;
    const iid = tr.identity_id ?? null;
    if (iid) {
      const d = byIdent.get(iid);
      if (d) {
        d.xy = [(d.xy[0] * d.n + p[0]) / (d.n + 1),
                (d.xy[1] * d.n + p[1]) / (d.n + 1)];
        d.n += 1;
        continue;
      }
      const e: LivePlayer & { n: number } = {
        id: iid, team: tr.team ?? "A",
        label: labels[iid] ?? null,
        ident: showIds,
        xy: [p[0], p[1]], n: 1, bridged: false,
      };
      byIdent.set(iid, e); out.push(e);
    } else {
      out.push({ id: `t${tr.id}`, team: tr.team ?? "A", label: null,
                 ident: false, xy: [p[0], p[1]], bridged: false });
    }
  }
  // bridge short gaps: identities with no live track at t
  for (const [iid, e] of segs) {
    if (byIdent.has(iid)) continue;
    let prev: Seg | null = null, next: Seg | null = null;
    for (const sg of e.segs) {
      if (sg.end <= t && (!prev || sg.end > prev.end)) prev = sg;
      if (sg.start >= t && (!next || sg.start < next.start)) next = sg;
    }
    let xy: [number, number] | null = null;
    if (prev && next && next.start - prev.end <= 45) {
      const f = (t - prev.end) / Math.max(1e-6, next.start - prev.end);
      xy = [prev.last[0] + (next.first[0] - prev.last[0]) * f,
            prev.last[1] + (next.first[1] - prev.last[1]) * f];
    } else if (prev && t - prev.end <= 8) {
      xy = prev.last;
    }
    if (xy) {
      const pl: LivePlayer & { n: number } = {
        id: iid, team: e.team,
        label: labels[iid] ?? null,
        ident: showIds, xy, bridged: true, n: 1,
      };
      byIdent.set(iid, pl); out.push(pl);
    }
  }
  return out;
}

/** Ball xy at t if a sample exists within 0.6 s, else null. */
function ballAt(ball: [number, number, number][], t: number): [number, number] | null {
  if (!ball.length) return null;
  let lo = 0, hi = ball.length - 1, best = 0, bd = Infinity;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const d = Math.abs(ball[mid][0] - t);
    if (d < bd) { bd = d; best = mid; }
    if (ball[mid][0] < t) lo = mid + 1; else hi = mid - 1;
  }
  return bd <= 0.6 ? [ball[best][1], ball[best][2]] : null;
}

export default function Replay3D({ onSeek }: { onSeek: (t: number) => void }) {
  const api = useProjectApi();
  const [open, setOpen] = useState(false);
  const [paths, setPaths] = useState<PlayersPaths | null>(null);
  const [identList, setIdentList] = useState<PlayerIdentity[]>([]);
  const [cands, setCands] = useState<Candidate[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [camMode, setCamMode] = useState("follow");
  const [win, setWin] = useState<[number, number]>([0, 0]);
  const [t, setT] = useState(0);
  const [recording, setRecording] = useState<number | null>(null);
  const [teamHex, setTeamHex] = useState<Record<string, string>>(TEAM_HEX);
  const [footage, setFootage] = useState(() => {
    try { return localStorage.getItem("replay.3d.footage") !== "0"; }
    catch { return true; }
  });
  const [vidSrc, setVidSrc] = useState<string | null>(null);
  const [director, setDirector] = useState<DirectorFull | null>(null);
  const [cutLo, setCutLo] = useState<number | null>(null);
  const [angleLabels, setAngleLabels] = useState<string[]>([]);
  const [muted, setMuted] = useState(true);
  const [vidChip, setVidChip] = useState("");
  const showIds = usePlayerIdentities();
  const cvRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const recStopRef = useRef<(() => void) | null>(null);
  const stateRef = useRef<{
    t: number; playing: boolean; speed: number; cam: string;
    win: [number, number]; idents: PlayerIdentity[];
    ids: boolean; rec: boolean; recT: number;
    foot: boolean; replays: ReplayInfo[]; loV: number;
    segs: DirectorSegment[]; alabels: string[];
    kit: Record<string, number>; lhex: Record<string, string>;
  }>({ t: 0, playing: false, speed: 1, cam: "follow", win: [0, 0], idents: [],
       ids: false, rec: false, recT: 0, foot: false, replays: [], loV: 0,
       segs: [], alabels: [], kit: TEAM, lhex: TEAM_HEX });
  stateRef.current = { t, playing, speed, cam: camMode, win, idents: identList,
                       ids: showIds, rec: recording !== null,
                       recT: recording === null ? 0 : stateRef.current.recT,
                       foot: footage && vidSrc !== null,
                       replays: director?.replays ?? [],
                       loV: cutLo ?? (paths?.window_shared?.[0] ?? 0),
                       segs: director?.segments ?? [],
                       alabels: angleLabels,
                       kit: { A: hexNum(teamHex.A) ?? TEAM.A,
                              B: hexNum(teamHex.B) ?? TEAM.B },
                       lhex: { A: lighten(teamHex.A),
                               B: lighten(teamHex.B) } };

  const lo = paths?.window_shared?.[0] ?? 0;
  const hi = paths?.window_shared?.[1] ?? 0;
  const pitch = paths?.pitch
    ?? (paths?.pitch_len_m ? { ...DEFAULT_PITCH, len_m: paths.pitch_len_m } : DEFAULT_PITCH);

  useEffect(() => {
    if (!open || paths) return;
    Promise.all([api.playerPaths(), fetchPlayers(api), api.listCandidates("time").catch(() => [])])
      .then(([p, pl, cs]) => {
        setPaths(p);
        if (pl?.teams) setTeamHex({
          A: pl.teams.A?.hex || TEAM_HEX.A,
          B: pl.teams.B?.hex || TEAM_HEX.B,
        });
        const w: [number, number] = p.window_shared ?? [0, 30];
        setWin(w); setT(w[0]);
        setCands(cs.filter(
          (c) => c.status === "confirmed" && (c.type === "goal" || c.type === "shot")));
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [open, paths, api]);

  // director-cut footage: video URL (+ proxy if ready), director segments /
  // replays for the live↔output time map, cut range, camera labels
  useEffect(() => {
    if (!open || !paths) return;
    let dead = false;
    void (async () => {
      try {
        const [v, ps, dir, cuts, ma] = await Promise.all([
          api.getVideo(),
          api.proxyStatus().catch(() => ({ ready: false, progress: 0 })),
          api.multiangleDirector().catch(() => null),
          api.listCuts().catch(() => null),
          api.multiangle().catch(() => null),
        ]);
        if (dead) return;
        setVidSrc(api.videoUrl(ps.ready ? "proxy" : "source",
                               String(v.registered_at)));
        setDirector(dir);
        setCutLo(cuts?.range?.[0] ?? null);
        setAngleLabels(
          (ma?.angles ?? []).map((a) => a.label || `Camera ${a.index + 1}`));
      } catch {
        if (!dead) setVidSrc(null);
      }
    })();
    return () => { dead = true; };
  }, [open, paths, api]);

  // video element masters playback only while playing w/ footage on;
  // here we just keep its transport in sync with that decision
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (playing && footage && vidSrc && recording === null) {
      v.playbackRate = speed;
      void v.play().catch(() => { /* autoplay blocked: 3D keeps running */ });
    } else v.pause();
  }, [playing, speed, footage, vidSrc, recording]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.muted = muted;
  }, [muted, vidSrc, footage]);

  // identities fetched whenever open: anchored numbers show even with
  // the per-player-cards toggle off (404 -> none)
  useEffect(() => {
    if (!open) return;
    void api.identities()
      .then((d) => setIdentList(d.identities))
      .catch(() => setIdentList([]));
  }, [open, api]);

  // renderer lifecycle — created once paths exist and the card is open
  useEffect(() => {
    const canvas = cvRef.current;
    if (!canvas || !paths || !open) return;
    const L = pitch.len_m, W = pitch.wid_m;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.shadowMap.enabled = true;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87a9d6);
    const camera = new THREE.PerspectiveCamera(40, 2, 0.1, 500);
    const controls = new OrbitControls(camera, canvas);
    controls.enabled = false;
    scene.add(new THREE.HemisphereLight(0xffffff, 0x224422, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(30, 60, 40); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -50, right: 50, top: 50, bottom: -50, far: 200 });
    scene.add(sun);
    const pitchG = new THREE.Group(); scene.add(pitchG);
    buildPitch(pitchG, pitch);
    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 24, 16),
      new THREE.MeshStandardMaterial({ color: 0xffffff }));
    ball.castShadow = true; scene.add(ball);
    const ballShadow = new THREE.Mesh(
      new THREE.CircleGeometry(0.25, 16),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }));
    ballShadow.rotation.x = -Math.PI / 2; scene.add(ballShadow);
    const players = new Map<string, PlayerRig>();
    const segs = identitySegments(paths);
    let ballSm: THREE.Vector3 | null = null;
    const camState = {
      pos: new THREE.Vector3(L / 2, 25, W + 40),
      look: new THREE.Vector3(L / 2, 0, -W / 2),
      aim: new THREE.Vector3(L / 2, 0, -W / 2),
      vel: new THREE.Vector3(0, 0, 0),
    };
    camera.position.copy(camState.pos);
    const stopRec = () => {
      const f = recStopRef.current; recStopRef.current = null; f?.();
    };

    let raf = 0, last = performance.now(), lastRecSec = -1, lastChip = "";
    const resize = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const pr = Math.min(2, window.devicePixelRatio || 1);
      if (w && h && (canvas.width !== Math.round(w * pr) || canvas.height !== Math.round(h * pr))) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h; camera.updateProjectionMatrix();
      }
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const st = stateRef.current;
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      resize();
      const vid = videoRef.current;
      const vidMaster = st.playing && st.foot && !st.rec && vid;
      if (vidMaster) {
        // video is the master clock: map its output time back to shared t
        const out = vid.currentTime;
        if (out >= toOut(st.win[1] - st.loV, st.replays)) {
          st.t = st.win[0]; setT(st.t);
          vid.currentTime = toOut(st.win[0] - st.loV, st.replays);
        } else {
          st.t = THREE.MathUtils.clamp(
            fromOut(out, st.replays) + st.loV, st.win[0], st.win[1]);
          setT(st.t);
        }
      } else if (st.playing) {
        const nt = st.t + dt * (st.rec ? 1 : st.speed);
        const wrapped = nt > st.win[1];
        st.t = wrapped ? st.win[0] : nt; setT(st.t);
        if (st.rec) {
          st.recT += dt;
          const sec = Math.floor(st.recT);
          if (sec !== lastRecSec) { lastRecSec = sec; setRecording(sec); }
          if (wrapped || st.recT >= st.win[1] - st.win[0] + 1) stopRec();
        }
      }
      if (vid && st.foot && !vidMaster) {
        // 3D clock is master (paused, scrubbing, recording): keep the
        // video parked on the right output frame
        const target = toOut(st.t - st.loV, st.replays);
        const thr = st.rec ? 0.5 : 0.25;
        if (Math.abs(vid.currentTime - target) > thr) vid.currentTime = target;
      }
      if (vid && st.foot) {
        const o = vid.currentTime;
        const seg = st.segs.find((s) => o >= s.t_start && o < s.t_end);
        const lbl = seg
          ? (st.alabels[seg.angle] ?? `Camera ${seg.angle + 1}`) : "";
        const txt = lbl + (inReplay(o, st.replays) ? " · REPLAY" : "");
        if (txt !== lastChip) { lastChip = txt; setVidChip(txt); }
      }
      const idl = Object.fromEntries(st.idents.map((i) => [i.id,
        st.ids
          ? (i.name || i.anchor_name || i.id)
          : (i.anchor_name || null)]));
      const pls = livePlayers(paths, st.t, idl, segs, st.ids);
      const bx = ballAt(paths.ball, st.t);
      const bp = bx ? V(bx[0], bx[1], 0) : null;
      const seen = new Set<string>();
      for (const pl of pls) {
        let g = players.get(pl.id);
        if (!g) {
          g = makePlayer(pl.team, pl.label, pl.ident, st.kit, st.lhex);
          players.set(pl.id, g); scene.add(g);
        } else if (g.userData.lbl !== pl.label) {
          const gg = g;
          gg.children.filter((o) => (o as THREE.Sprite).isSprite)
            .forEach((o) => gg.remove(o));
          if (pl.label) {
            const lab = makeLabel(pl.label, st.lhex[pl.team] ?? "#e4e4e7");
            lab.position.y = 2.15; gg.add(lab);
          }
          gg.userData.lbl = pl.label;
        }
        g.visible = true;
        const op = pl.bridged ? 0.45 : 1.0;
        g.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.isMesh) {
            const mat = m.material as THREE.MeshBasicMaterial;
            mat.opacity = (m.geometry as THREE.RingGeometry).type === "RingGeometry"
              ? (pl.ident ? 0.7 : 0.4) * op : op;
          }
        });
        updatePlayer(g, pl.xy, st.playing ? dt * st.speed : 0, bp);
        seen.add(pl.id);
      }
      for (const [id, g] of players) if (!seen.has(id)) g.visible = false;
      if (bx) {
        ball.visible = ballShadow.visible = true;
        const v = V(bx[0], bx[1], 0.22);
        if (!ballSm || !st.playing) (ballSm ??= new THREE.Vector3()).copy(v);
        else ballSm.lerp(v, 1 - Math.exp(-dt * 10));
        ball.position.copy(ballSm);
        ballShadow.position.set(ballSm.x, 0.03, ballSm.z);
      } else { ball.visible = ballShadow.visible = false; ballSm = null; }

      // camera
      if (st.cam === "free") {
        controls.enabled = true; controls.update();
      } else {
        controls.enabled = false;
        // cluster centroid: median player, mean of those within 12 m
        let cc = new THREE.Vector3(L / 2, 0, -W / 2);
        if (pls.length) {
          const xs = [...pls.map((p) => p.xy[0])].sort((a, c) => a - c);
          const ys = [...pls.map((p) => p.xy[1])].sort((a, c) => a - c);
          const mx = xs[Math.floor(xs.length / 2)], my = ys[Math.floor(ys.length / 2)];
          const near = pls.filter((p) => Math.hypot(p.xy[0] - mx, p.xy[1] - my) <= 12);
          const n = near.length || 1;
          cc = new THREE.Vector3(
            near.reduce((s, p) => s + p.xy[0], 0) / n, 0,
            -near.reduce((s, p) => s + p.xy[1], 0) / n);
        }
        const b = bp ?? cc;
        let tp: THREE.Vector3, tl: THREE.Vector3;
        let k = 1 - Math.exp(-dt * 2.2);
        if (st.cam === "follow") {
          const raw = bp && bp.distanceTo(cc) <= 15 ? bp : cc;
          const ka = st.playing ? 1 - Math.exp(-dt * 1.5) : 1;
          const px = camState.aim.x, pz = camState.aim.z;
          camState.aim.x += (raw.x - camState.aim.x) * ka;
          camState.aim.z += (raw.z - camState.aim.z) * ka;
          if (st.playing && dt > 0) {
            camState.vel.x += ((camState.aim.x - px) / dt - camState.vel.x)
              * Math.min(1, dt * 4);
            camState.vel.z += ((camState.aim.z - pz) / dt - camState.vel.z)
              * Math.min(1, dt * 4);
          } else camState.vel.set(0, 0, 0);
          const ax = camState.aim.x
            + THREE.MathUtils.clamp(camState.vel.x * 0.8, -6, 6);
          const az = camState.aim.z
            + THREE.MathUtils.clamp(camState.vel.z * 0.8, -6, 6);
          let spread = 0, ns = 0;
          for (const p of pls) {
            const dd = Math.hypot(p.xy[0] - ax, -p.xy[1] - az);
            if (dd <= 12) { spread += dd * dd; ns++; }
          }
          spread = ns ? Math.sqrt(spread / ns) : 0;
          const d = THREE.MathUtils.clamp(18 + spread * 1.4, 22, 44);
          tp = new THREE.Vector3(THREE.MathUtils.clamp(ax, 6, L - 6), 0.42 * d, az + d);
          tl = new THREE.Vector3(ax, 0.8, az);
          k = st.playing ? 1 - Math.exp(-dt * 1.2) : 1;
        } else if (st.cam === "broadcast") {
          const x = THREE.MathUtils.clamp(b.x, 8, L - 8);
          tp = new THREE.Vector3(x, 18, W * 0.1 + 30);
          tl = new THREE.Vector3(THREE.MathUtils.clamp(b.x, 4, L - 4), 0.5,
                                 THREE.MathUtils.clamp(b.z, -W + 4, -4));
        } else if (st.cam === "tactical") {
          tp = new THREE.Vector3(THREE.MathUtils.clamp(b.x, 12, L - 12), 32, -W / 2 + 30);
          tl = new THREE.Vector3(b.x, 0, -W / 2);
        } else if (st.cam === "goalA") {
          tp = new THREE.Vector3(-12, 6, -W / 2);
          tl = new THREE.Vector3(Math.max(b.x, 4), 1, b.z);
        } else {
          tp = new THREE.Vector3(L + 12, 6, -W / 2);
          tl = new THREE.Vector3(Math.min(b.x, L - 4), 1, b.z);
        }
        camState.pos.lerp(tp, k);
        camState.look.lerp(tl, Math.min(1, k * 1.4));
        camera.position.copy(camState.pos); camera.lookAt(camState.look);
      }
      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf); ro.disconnect(); controls.dispose();
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh || (o as THREE.Sprite).isSprite) {
          (m.geometry as THREE.BufferGeometry | undefined)?.dispose?.();
          const mat = m.material as THREE.Material | THREE.Material[] | undefined;
          if (Array.isArray(mat)) mat.forEach((mm) => mm.dispose()); else mat?.dispose?.();
        }
      });
      renderer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paths, open, pitch.len_m, pitch.wid_m, pitch.template,
      pitch.goal_w_m, pitch.d_radius_m]);

  const pickWindow = (w: [number, number]) => {
    setWin(w); setT(w[0]); setPlaying(true);
    const v = videoRef.current;
    if (v) v.currentTime = toOut(w[0] - stateRef.current.loV,
                                 stateRef.current.replays);
  };

  const recSupported = typeof MediaRecorder !== "undefined";
  const exportClip = () => {
    const canvas = cvRef.current;
    if (!canvas || recording !== null || !recSupported) return;
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9" : "video/webm";
    const mr = new MediaRecorder(canvas.captureStream(30), { mimeType: mime });
    const chunks: Blob[] = [];
    mr.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    mr.onstop = () => {
      const mid = (win[0] + win[1]) / 2;
      const c = cands.reduce<Candidate | null>((b, x) =>
        !b || Math.abs(x.t - mid) < Math.abs(b.t - mid) ? x : b, null);
      const tag = `${c?.type ?? "clip"}-`
        + fmtClock(c?.t ?? mid).slice(-5).replace(":", "-");
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      a.download = `3d-${tag}.webm`;
      a.click();
      URL.revokeObjectURL(a.href);
      setRecording(null); setPlaying(false);
    };
    stateRef.current.recT = 0;
    recStopRef.current = () => mr.stop();
    setT(win[0]); setPlaying(true); setRecording(0);
    mr.start(250);
  };

  return (
    <div className={`${card} min-w-0`}>
      <button type="button" className="flex items-center gap-2 w-full text-left"
        onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? <ChevronDown size={14} className="text-zinc-500" />
          : <ChevronRight size={14} className="text-zinc-500" />}
        <Maximize2 size={14} className="text-zinc-500" />
        <span className={head}>3D replay (beta)</span>
        {paths && (
          <span className="ml-auto text-[11px] text-zinc-500">
            {fmtClock(win[1] - win[0])} window
          </span>
        )}
      </button>
      {open && (
        <div className="mt-3 flex flex-col gap-3">
          {error && <div className="text-xs text-red-300">{error}</div>}
          {!paths ? (
            <div className="text-sm text-zinc-500">
              {error ? "3D replay unavailable — run Player analysis first." : "Loading…"}
            </div>
          ) : (
            <>
              <div ref={wrapRef} className="relative w-full aspect-video bg-black rounded-md">
                <canvas ref={cvRef} className="absolute inset-0 w-full h-full" />
                <button type="button" className={`${btnGhost} absolute top-2 right-2`}
                  onClick={() => void wrapRef.current?.requestFullscreen?.()}
                  aria-label="fullscreen">
                  <Maximize2 size={12} />
                </button>
              </div>
              {footage && vidSrc && (
                <div className="relative w-full aspect-video bg-black rounded-md overflow-hidden">
                  <video ref={videoRef} src={vidSrc} muted={muted} playsInline
                    preload="metadata" className="absolute inset-0 w-full h-full" />
                  <span className="absolute top-2 left-2 rounded bg-black/60 px-2 py-0.5 text-[11px] text-zinc-200">
                    {vidChip || "Footage"}
                  </span>
                  <button type="button" className={`${btnGhost} absolute top-2 right-2`}
                    onClick={() => setMuted((m) => !m)}
                    aria-label={muted ? "unmute footage" : "mute footage"}>
                    {muted ? <VolumeX size={12} /> : <Volume2 size={12} />}
                  </button>
                </div>
              )}
              <div className="flex items-center gap-2 flex-wrap">
                <button type="button" className={chip}
                  onClick={() => pickWindow([lo, hi])}>Whole match</button>
                {cands.map((c) => (
                  <button key={c.id} type="button" className={chip}
                    onClick={() => pickWindow([
                      Math.max(lo, c.t - 10),
                      Math.min(hi, c.t + 10),
                    ])}>
                    {c.type === "goal" ? "Goal" : "Shot"} {fmtClock(c.t)}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <button type="button" className={btnGhost} disabled={recording !== null}
                  onClick={() => setPlaying((p) => !p)}>
                  {playing ? <Pause size={12} /> : <Play size={12} />}
                  {playing ? "Pause" : "Play"}
                </button>
                <select className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs text-zinc-300 disabled:opacity-40"
                  value={camMode} disabled={recording !== null}
                  onChange={(e) => setCamMode(e.target.value)}>
                  {CAM_MODES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
                <select className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-1 text-xs text-zinc-300 disabled:opacity-40"
                  value={speed} disabled={recording !== null}
                  onChange={(e) => setSpeed(Number(e.target.value))}>
                  {[0.25, 0.5, 1, 2].map((s) => <option key={s} value={s}>{s}×</option>)}
                </select>
                <input type="range" className="flex-1 accent-amber-400 min-w-32 disabled:opacity-40"
                  min={win[0]} max={win[1]} step={0.1} value={t}
                  disabled={recording !== null}
                  onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }}
                  aria-label="scrub 3D replay" />
                <span className="text-[11px] font-mono text-zinc-400">
                  {fmtClock(t)}
                </span>
                {recording !== null && (
                  <span className="text-[11px] text-red-400">Recording… {recording}s</span>
                )}
                <button type="button" className={btnGhost} disabled={recording !== null}
                  onClick={() => onSeek(toOut(
                    Math.max(0, t - stateRef.current.loV),
                    stateRef.current.replays))}>
                  Jump to video
                </button>
                <button type="button" className={btnGhost}
                  disabled={!recSupported || recording !== null || win[1] - win[0] > 60}
                  title={!recSupported ? "Not supported in this browser"
                    : win[1] - win[0] > 60 ? "Pick a goal/shot window (≤60 s)" : undefined}
                  onClick={exportClip}>
                  <Download size={12} /> Export this clip (WebM)
                </button>
                <label className="flex items-center gap-1 text-[11px] text-zinc-400"
                  title={vidSrc ? undefined : "No video on this project"}>
                  <input type="checkbox" checked={footage} disabled={!vidSrc}
                    onChange={(e) => {
                      setFootage(e.target.checked);
                      try {
                        localStorage.setItem("replay.3d.footage",
                                             e.target.checked ? "1" : "0");
                      } catch { /* private mode */ }
                    }} />
                  Footage
                </label>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
