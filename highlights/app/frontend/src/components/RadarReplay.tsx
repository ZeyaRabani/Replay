import { ChevronDown, ChevronRight, Loader2, Maximize2, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useProjectApi } from "../api";
import { applyH, homography, type Mat3 } from "../lib/homography";
import { DEFAULT_PITCH, drawPitch, pitchView } from "../lib/pitch";
import { IDENTITIES_CHANGED_EVENT, fetchPlayers, identityHex, usePlayerIdentities } from "../lib/players";
import { fmtClock } from "../lib/time";
import type { CalibResponse, MultiangleInfo, PitchDims, PlayersPaths, RadarPitch } from "../types";
import CameraCalib, { rmsTone } from "./CameraCalib";
import CameraPlacement from "./CameraPlacement";

const card = "card p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnGhost = "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";

const TEAM_HEX: Record<string, string> = { A: "#22c55e", B: "#f97316" };
const TRAIL_S = 1.5;
const VISIBLE_STEP_S = 0.5;
/** seconds a dot stays drawn (alpha 1 -> 0.35) after its last sample */
const HOLD_S = 2.0;

type Pt = [number, number, number];

/** seconds a dot ramps in (alpha 0.35 -> 1) after its first sample */
const FADE_IN_S = 0.6;

export type PosEx = { x: number; y: number; alpha: number; out: boolean };

function catmull(p0: number, p1: number, p2: number, p3: number, f: number): number {
  const f2 = f * f, f3 = f2 * f;
  return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2
    + (-p0 + 3 * p1 - 3 * p2 + p3) * f3);
}

/** position of a track at shared t with Catmull-Rom interpolation between
 *  samples; alpha ramps in over FADE_IN_S and fades 1 -> 0.35 over HOLD_S
 *  after the last sample (out=true while holding). null if not alive. */
export function posAtEx(pts: Pt[], t: number): PosEx | null {
  if (!pts.length || t < pts[0][0] - 0.5) return null;
  const last = pts[pts.length - 1][0];
  if (t > last) {
    const g = t - last;
    if (g > HOLD_S) return null;
    return { x: pts[pts.length - 1][1], y: pts[pts.length - 1][2],
             alpha: 1 - 0.65 * (g / HOLD_S), out: true };
  }
  let lo = 0, hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid][0] <= t) lo = mid; else hi = mid - 1;
  }
  const a = pts[lo];
  const b = pts[Math.min(lo + 1, pts.length - 1)];
  const dt = b[0] - a[0];
  const f = Math.min(1, Math.max(0, (t - a[0]) / Math.max(1e-6, dt)));
  const since = t - pts[0][0];
  const alpha = since >= FADE_IN_S ? 1 : 0.35 + 0.65 * Math.max(0, since) / FADE_IN_S;
  // only curve across regular samples; a long gap is bridged linearly
  if (dt > 0 && dt <= 1.6) {
    const p0 = pts[Math.max(0, lo - 1)], p3 = pts[Math.min(pts.length - 1, lo + 2)];
    return { x: catmull(p0[1], a[1], b[1], p3[1], f),
             y: catmull(p0[2], a[2], b[2], p3[2], f), alpha, out: false };
  }
  return { x: a[1] + (b[1] - a[1]) * f, y: a[2] + (b[2] - a[2]) * f, alpha, out: false };
}

/** interpolated (a,b) of a track's pts at shared t, or null if the track
 *  isn't alive; third value = alpha (fade-in / hold fade-out) */
export function posAt(pts: Pt[], t: number): Pt | null {
  const p = posAtEx(pts, t);
  return p ? [p.x, p.y, p.alpha] : null;
}

/** frame-space (old data) -> pitch metres: calibrated H of the reference
 *  angle, else the legacy 4 corners, else the whole frame. */
function frameToPitch(calib: CalibResponse | null, ref: number, corners: RadarPitch["corners"], p: PitchDims): Mat3 | null {
  const H = calib?.angles[String(ref)]?.H;
  if (H && H.length === 3) return H.flat() as Mat3;
  const c = corners ?? [[0, 1], [1, 1], [1, 0], [0, 0]];
  // near-left, near-right, far-right, far-left
  return homography(c, [[0, p.wid_m], [p.len_m, p.wid_m], [p.len_m, 0], [0, 0]]);
}

function VisibleStrip({ hist, lo, hi, t }: { hist: number[]; lo: number; hi: number; t: number }) {
  const max = Math.max(1, ...hist);
  const width = Math.max(1e-3, hi - lo);
  return (
    <svg className="w-full h-7 block" preserveAspectRatio="none" viewBox={`0 0 ${width} 1`}>
      <title>players visible over time</title>
      {hist.map((v, k) => {
        const x = k * VISIBLE_STEP_S;
        const h = Math.max(0, Math.min(1, v / max));
        const active = t >= lo + x && t < lo + x + VISIBLE_STEP_S;
        return (
          <rect key={k} x={x} y={1 - h} width={VISIBLE_STEP_S} height={h}
            fill={active ? "#fbbf24" : "#52525b"} />
        );
      })}
      <line x1={t - lo} x2={t - lo} y1={0} y2={1} stroke="#fbbf24" strokeWidth={1}
        vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

export default function RadarReplay({ onSeek }: { onSeek?: (t: number) => void }) {
  const api = useProjectApi();
  const [open, setOpen] = useState(false);
  const [paths, setPaths] = useState<PlayersPaths | null>(null);
  const [corners, setCorners] = useState<RadarPitch["corners"]>(null);
  const [calib, setCalib] = useState<CalibResponse | null>(null);
  const [calibOpen, setCalibOpen] = useState(false);
  const [camsOpen, setCamsOpen] = useState(true);
  const [rosterNames, setRosterNames] = useState<Record<string, string>>({});
  const [identNames, setIdentNames] = useState<Record<string, string | null>>({});
  const [hover, setHover] = useState<{ x: number; y: number; label: string } | null>(null);
  const drawn = useRef<{ x: number; y: number; id: number; ident: string | null }[]>([]);
  const [teamHex, setTeamHex] = useState<Record<string, string>>(TEAM_HEX);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [t, setT] = useState(0);
  const [visible, setVisible] = useState(0);
  const cvRef = useRef<HTMLCanvasElement>(null);
  const smooth = useRef(new Map<number, [number, number]>());
  const lastTs = useRef(0);
  const showIds = usePlayerIdentities();
  // synced camera footage panel
  const [footage, setFootage] = useState(() => {
    try { return localStorage.getItem("replay.radar.footage") !== "0"; }
    catch { return true; }
  });
  const [cam, setCam] = useState(0);
  const [maInfo, setMaInfo] = useState<MultiangleInfo | null>(null);
  const [vidSrc, setVidSrc] = useState<string | null>(null);
  const [muted, setMuted] = useState(true);
  const [vidDur, setVidDur] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const offsets = maInfo?.sync?.offsets ?? [];
  const camClamped = Math.min(cam, Math.max(0, (maInfo?.angles.length ?? 1) - 1));
  const off = offsets[camClamped] ?? 0;
  // shared-T = file_t + offset  ->  file_t = t - offset
  const fileT = t - off;
  const angleDur = vidDur || (maInfo?.angles[camClamped]?.duration ?? 0);
  const covered = vidSrc != null && fileT >= -0.05 && (angleDur <= 0 || fileT <= angleDur);

  const lo = paths?.window_shared?.[0] ?? 0;
  const hi = paths?.window_shared?.[1] ?? 0;
  const inPitch = paths?.space === "pitch";

  useEffect(() => {
    if (!open || paths) return;
    Promise.all([
      api.playerPaths(),
      api.radarPitch().catch(() => null),
      api.calib().catch(() => null),
      fetchPlayers(api),
    ])
      .then(([p, rp, c, pl]) => {
        setPaths(p);
        setCorners(rp?.corners ?? null);
        setCalib(c);
        setT(p.window_shared?.[0] ?? 0);
        if (pl) {
          const names: Record<string, string> = {};
          for (const r of pl.roster.players) names[r.id] = r.name;
          setRosterNames(names);
          if (pl.teams)
            setTeamHex({ A: pl.teams.A?.hex || TEAM_HEX.A, B: pl.teams.B?.hex || TEAM_HEX.B });
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [open, paths, api]);

  useEffect(() => {
    if (!open || !showIds) {
      setIdentNames({});
      return;
    }
    const load = () => void api.identities()
      .then((d) => setIdentNames(Object.fromEntries(d.identities.map((i) => [i.id, i.name]))))
      .catch(() => setIdentNames({}));
    load();
    // re-linking renumbers identities: refetch the paths too
    const relink = () => { load(); setPaths(null); };
    window.addEventListener(IDENTITIES_CHANGED_EVENT, relink);
    return () => window.removeEventListener(IDENTITIES_CHANGED_EVENT, relink);
  }, [open, api, showIds]);

  const pitch: PitchDims = useMemo(
    () => paths?.pitch ?? calib?.pitch
      ?? (paths?.pitch_len_m ? { ...DEFAULT_PITCH, len_m: paths.pitch_len_m } : DEFAULT_PITCH),
    [paths, calib],
  );

  // pitch-space data needs no transform
  const H = useMemo(
    () => (inPitch || !paths ? null : frameToPitch(calib, paths.ref_angle, corners, pitch)),
    [inPitch, paths, calib, corners, pitch],
  );

  // multiangle info for the footage panel (labels + sync offsets)
  useEffect(() => {
    if (!open || !footage || maInfo !== null) return;
    let dead = false;
    void api.multiangle()
      .then((m) => { if (!dead) setMaInfo(m); })
      .catch(() => { if (!dead) setMaInfo({} as MultiangleInfo); });
    return () => { dead = true; };
  }, [open, footage, maInfo, api]);

  // pick the video source for the chosen camera: proxy if ready, else the
  // source immediately + kick off the proxy build and poll until ready
  useEffect(() => {
    if (!open || !footage || !maInfo) return;
    const n = maInfo.angles?.length ?? 0;
    if (!n) { setVidSrc(null); return; }
    const i = Math.min(cam, n - 1);
    let dead = false, poll = 0;
    void api.angleProxyStatus(i).then((s) => {
      if (dead) return;
      if (s.status === "ready") { setVidSrc(api.angleProxyUrl(i)); return; }
      setVidSrc(api.angleVideoUrl(i));
      if (s.status === "missing") void api.startAngleProxy(i).catch(() => undefined);
      const tick = () => {
        void api.angleProxyStatus(i).then((st) => {
          if (dead) return;
          if (st.status === "ready") setVidSrc(api.angleProxyUrl(i));
          else poll = window.setTimeout(tick, 15000);
        }).catch(() => { poll = window.setTimeout(tick, 15000); });
      };
      poll = window.setTimeout(tick, 15000);
    }).catch(() => { if (!dead) setVidSrc(api.angleVideoUrl(i)); });
    return () => { dead = true; window.clearTimeout(poll); };
  }, [open, footage, maInfo, cam, api]);

  // video transport: master clock while playing; paused otherwise
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (playing && footage && vidSrc) {
      v.playbackRate = Math.min(4, speed);
      void v.play().catch(() => { /* autoplay blocked: radar keeps running */ });
    } else v.pause();
  }, [playing, speed, footage, vidSrc]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.muted = muted;
  }, [muted, vidSrc]);

  // paused/scrubbing or camera/source switch: park the video on the mapped
  // file time (debounced so dragging the slider doesn't thrash seeks)
  useEffect(() => {
    if (playing || !footage || !vidSrc) return;
    const id = window.setTimeout(() => {
      const v = videoRef.current;
      if (!v) return;
      const ft = t - off;
      if (ft >= 0 && Math.abs(v.currentTime - ft) > 0.3) v.currentTime = ft;
    }, 150);
    return () => window.clearTimeout(id);
  }, [t, playing, footage, vidSrc, camClamped, off]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const step = (ts: number) => {
      const v = videoRef.current;
      if (footage && vidSrc && v) {
        // video is the master clock
        const shared = v.currentTime + off;
        if (shared >= hi) { setT(lo); v.currentTime = lo - off; }
        else setT(shared);
      } else if (lastTs.current) {
        setT((v2) => {
          const nt = v2 + ((ts - lastTs.current) / 1000) * speed;
          return nt >= hi ? lo : nt;
        });
      }
      lastTs.current = ts;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => { cancelAnimationFrame(raf); lastTs.current = 0; };
  }, [playing, speed, lo, hi, footage, vidSrc, off]);

  useEffect(() => {
    const cv = cvRef.current;
    if (!cv || !paths || (!inPitch && !H)) return;
    const W = cv.clientWidth || 800;
    const Hh = Math.round(W * (pitch.wid_m / pitch.len_m) * 1.08);
    if (cv.width !== W) cv.width = W;
    if (cv.height !== Hh) cv.height = Hh;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const v = pitchView(W, Hh, Math.max(14, W * 0.03), pitch);
    drawPitch(ctx, W, Hh, v, pitch);
    const toM = (a: number, b: number): [number, number] => {
      const [x, y] = inPitch || !H ? [a, b] : applyH(H, a, b);
      return [Math.min(pitch.len_m * 1.04, Math.max(-pitch.len_m * 0.04, x)),
              Math.min(pitch.wid_m * 1.04, Math.max(-pitch.wid_m * 0.04, y))];
    };
    const r = Math.max(5, W / 140);
    const labels: [string, number, number, number][] = [];
    const hits: typeof drawn.current = [];
    let on = 0;
    // cross-camera duplicates of one identity are drawn as a single dot
    const dots: { tr: PlayersPaths["tracks"][number]; m: [number, number]; a: number; n: number; out: boolean }[] = [];
    const byIdent = new Map<string, (typeof dots)[number]>();
    for (const tr of paths.tracks) {
      if (tr.hidden || (tr.team !== "A" && tr.team !== "B")) continue;
      const q = posAtEx(tr.pts, t);
      if (!q) continue;
      const p: Pt = [q.x, q.y, q.alpha];
      let m = toM(p[0], p[1]);
      if (!inPitch) {
        // EMA smoothing of the jittery single-camera projection
        const prev = smooth.current.get(tr.id);
        if (prev && !q.out) m = [prev[0] + 0.6 * (m[0] - prev[0]), prev[1] + 0.6 * (m[1] - prev[1])];
        smooth.current.set(tr.id, m);
      }
      const iid = tr.identity_id ?? null;
      const d = iid ? byIdent.get(iid) : undefined;
      if (d) {
        d.m = [(d.m[0] * d.n + m[0]) / (d.n + 1), (d.m[1] * d.n + m[1]) / (d.n + 1)];
        d.n += 1;
        d.a = Math.max(d.a, p[2]);
        continue;
      }
      const dot = { tr, m: m as [number, number], a: p[2], n: 1, out: q.out };
      dots.push(dot);
      if (iid) byIdent.set(iid, dot);
    }
    for (const { tr, m, a, out } of dots) {
      if (!out) on += 1;
      const x = v.X(m[0]), y = v.Y(m[1]);
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(x, y + 1.5, r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      const team = teamHex[tr.team ?? ""] ?? "#a1a1aa";
      const iid = tr.identity_id ?? null;
      ctx.fillStyle = showIds && iid ? identityHex(iid) : team;
      ctx.fill();
      ctx.lineWidth = showIds && iid ? 2.5 : 2;
      ctx.strokeStyle = showIds && iid ? team : "#09090b";
      ctx.stroke();
      const nm = showIds
        ? (iid && identNames[iid]) || (tr.player_id ? rosterNames[tr.player_id] : null)
        : null;
      if (nm) labels.push([nm, x, y - r - 5, a]);
      if (!out) hits.push({ x, y, id: tr.id, ident: iid });
    }
    // labels on top of all dots
    ctx.font = `600 ${Math.max(10, Math.round(W / 90))}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.lineJoin = "round";
    for (const [nm, x, y, a] of labels) {
      ctx.globalAlpha = a;
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(9,9,11,0.85)";
      ctx.strokeText(nm, x, y);
      ctx.fillStyle = "#fafafa";
      ctx.fillText(nm, x, y);
    }
    ctx.globalAlpha = 1;
    const b = posAt(paths.ball, t);
    if (b) {
      const trail: [number, number][] = [];
      for (let k = 12; k >= 1; k--) {
        const q = posAt(paths.ball, t - (TRAIL_S * k) / 12);
        if (q && q[2] === 1) trail.push(toM(q[0], q[1]));
      }
      const bm = toM(b[0], b[1]);
      trail.push(bm);
      for (let i = 1; i < trail.length; i++) {
        ctx.beginPath();
        ctx.moveTo(v.X(trail[i - 1][0]), v.Y(trail[i - 1][1]));
        ctx.lineTo(v.X(trail[i][0]), v.Y(trail[i][1]));
        ctx.strokeStyle = `rgba(255,255,255,${(0.55 * i) / trail.length})`;
        ctx.lineWidth = Math.max(1, r * 0.35);
        ctx.lineCap = "round";
        ctx.stroke();
      }
      ctx.globalAlpha = b[2];
      ctx.beginPath();
      ctx.arc(v.X(bm[0]), v.Y(bm[1]), Math.max(2.5, r * 0.45), 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#09090b";
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    drawn.current = hits;
    setVisible(on);
  }, [t, paths, H, inPitch, pitch, teamHex, rosterNames, identNames, showIds]);

  const onCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = e.currentTarget;
    const rect = cv.getBoundingClientRect();
    const sx = cv.width / Math.max(1, rect.width);
    const mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sx;
    const r = Math.max(5, cv.width / 140) * 1.8;
    let best: (typeof drawn.current)[number] | null = null;
    let bd = r * r;
    for (const h of drawn.current) {
      const d = (h.x - mx) ** 2 + (h.y - my) ** 2;
      if (d <= bd) { bd = d; best = h; }
    }
    if (!best) { setHover(null); return; }
    const nm = showIds && best.ident ? identNames[best.ident] : null;
    const label = showIds && best.ident
      ? (nm ? `${nm} (${best.ident})` : `${best.ident} · unnamed`)
      : `track ${best.id}`;
    setHover({ x: best.x / sx, y: best.y / sx, label });
  };

  const nAngles = calib ? Object.keys(calib.angles).length : 0;
  const nCams = calib ? Object.keys(calib.cameras ?? {}).length : 0;
  const nSolved = calib ? Object.values(calib.angles).filter((a) => a.H).length : 0;
  const worst = calib
    ? Math.max(0, ...Object.values(calib.angles).map((a) => a.rms_m ?? 0))
    : null;

  return (
    <div className={`${card} min-w-0`}>
      <button type="button" className="flex items-center gap-2 w-full text-left"
        onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
        <Maximize2 size={14} className="text-zinc-500" />
        <span className={head}>Radar replay &amp; camera calibration</span>
        {paths && (
          <span className="ml-auto text-[11px] text-zinc-500">
            {paths.tracks.filter((x) => !x.hidden).length} tracks
            {inPitch ? " · multi-camera" : " · main camera"}
          </span>
        )}
      </button>
      {open && (
        <div className="mt-3 flex flex-col gap-3">
          {error && <div className="text-xs text-red-300">{error}</div>}
          <div className="flex items-center gap-2 flex-wrap text-xs">
            <span className="text-zinc-400 flex items-center gap-1.5">
              {calib && nSolved > 0 && (
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: rmsTone(worst).hex }} />
              )}
              {!calib
                ? "Cameras not placed yet."
                : nSolved > 0
                  ? `Cameras calibrated ${nSolved}/${Math.max(nAngles, 3)}`
                  : `Cameras placed ${nCams}/${Math.max(nAngles, 3)} · awaiting fine calibration`}
            </span>
            {nSolved > 0 && (
              <button type="button" className={btnGhost} onClick={() => setCamsOpen((o) => !o)}>
                {camsOpen ? "Hide camera map" : "Camera map"}
              </button>
            )}
            <button type="button"
              className="text-zinc-500 hover:text-zinc-300 underline underline-offset-2 text-[11px]"
              onClick={() => setCalibOpen((o) => !o)}>
              {calibOpen ? "Close landmarks" : "Advanced: landmarks"}
            </button>
            {calib && (
              <span className="flex items-center gap-1.5 text-[11px] text-zinc-500">
                Pitch:
                <select
                  className="bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-xs text-zinc-300"
                  value={calib.pitch.template ?? "full"}
                  onChange={(e) => {
                    const template = e.target.value as "full" | "small";
                    void api.putCalibPitch({
                      ...calib.pitch, template,
                      ...(template === "small"
                        ? { goal_w_m: calib.pitch.goal_w_m ?? 3.66,
                            d_radius_m: calib.pitch.d_radius_m ?? 9.0 }
                        : {}),
                    }).then(setCalib).catch(() => setError("pitch save failed"));
                  }}>
                  <option value="full">Full size</option>
                  <option value="small">Small-sided</option>
                </select>
                <input type="number" aria-label="pitch length (m)"
                  className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 font-mono text-xs text-zinc-200"
                  defaultValue={calib.pitch.len_m} key={`L${calib.pitch.len_m}`}
                  onBlur={(e) => {
                    const v = Number(e.target.value);
                    if (v && v !== calib.pitch.len_m)
                      void api.putCalibPitch({ ...calib.pitch, len_m: v })
                        .then(setCalib).catch(() => setError("pitch save failed"));
                  }} />
                ×
                <input type="number" aria-label="pitch width (m)"
                  className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 font-mono text-xs text-zinc-200"
                  defaultValue={calib.pitch.wid_m} key={`W${calib.pitch.wid_m}`}
                  onBlur={(e) => {
                    const v = Number(e.target.value);
                    if (v && v !== calib.pitch.wid_m)
                      void api.putCalibPitch({ ...calib.pitch, wid_m: v })
                        .then(setCalib).catch(() => setError("pitch save failed"));
                  }} />
                m
              </span>
            )}
          </div>
          {calibOpen && <CameraCalib onSaved={setCalib} defaultT={paths?.frame_t ?? null} />}
          {(nSolved === 0 || camsOpen) && (
            <CameraPlacement onSaved={setCalib} frameT={paths?.frame_t ?? null} />
          )}
          {!paths ? (
            <div className="text-sm text-zinc-500 flex items-center gap-2">
              {!error && <Loader2 size={14} className="animate-spin" />}
              {error ? "Radar unavailable — run Player analysis first." : "Loading…"}
            </div>
          ) : (
            <>
              <div className="relative">
                <canvas ref={cvRef} className="w-full rounded-md" onMouseMove={onCanvasMove}
                  onMouseLeave={() => setHover(null)} />
                {hover && (
                  <div data-radar-hover
                    className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded bg-zinc-950/90 border border-zinc-700 px-1.5 py-0.5 text-[11px] text-zinc-100 whitespace-nowrap"
                    style={{ left: hover.x, top: hover.y - 10 }}>
                    {hover.label}
                  </div>
                )}
                <div className="absolute top-2 left-2 flex items-center gap-2 rounded bg-zinc-950/75 px-2 py-1">
                  <span className="text-[10px] uppercase tracking-wide text-zinc-400">Players visible</span>
                  <span className="font-mono text-sm font-semibold text-amber-300" data-radar-count>{visible}</span>
                </div>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <button type="button" className={btnGhost} onClick={() => setPlaying((p) => !p)}>
                  {playing ? <Pause size={12} /> : <Play size={12} />}
                  {playing ? "Pause" : "Play"}
                </button>
                {[1, 2, 4].map((s) => (
                  <button key={s} type="button"
                    className={`rounded px-2 py-1 text-xs ${speed === s ? "bg-amber-500 text-zinc-950 font-semibold font-semibold" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}
                    onClick={() => setSpeed(s)}>
                    {s}×
                  </button>
                ))}
                <span className="text-[11px] font-mono text-zinc-400 ml-1">
                  {fmtClock(t - lo)} / {fmtClock(hi - lo)}
                </span>
                <label className="flex items-center gap-1 text-[11px] text-zinc-400"
                  title={vidSrc ? undefined : "No multi-camera footage on this project"}>
                  <input type="checkbox" checked={footage} disabled={!vidSrc}
                    onChange={(e) => {
                      setFootage(e.target.checked);
                      try {
                        localStorage.setItem("replay.radar.footage",
                                             e.target.checked ? "1" : "0");
                      } catch { /* private mode */ }
                    }} />
                  Footage
                </label>
                {onSeek && (
                  <button type="button" className={`${btnGhost} ml-auto`} onClick={() => onSeek(Math.max(0, t - lo))}>
                    Jump video here
                  </button>
                )}
              </div>
              <div className="flex flex-col gap-0.5 w-full">
                {paths.visible_hist && paths.visible_hist.length > 0 && (
                  <VisibleStrip hist={paths.visible_hist} lo={lo} hi={hi} t={t} />
                )}
                <input
                  type="range"
                  className="w-full accent-amber-400"
                  min={lo} max={hi} step={0.1}
                  value={t}
                  onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }}
                  aria-label="scrub radar"
                />
              </div>
              {footage && (maInfo?.angles?.length ?? 0) > 0 && (
                <div className="rounded-md border border-zinc-800 overflow-hidden">
                  <div className="flex items-center gap-2 px-2 py-1.5 bg-zinc-900">
                    <select
                      className="bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-xs text-zinc-300"
                      value={camClamped} onChange={(e) => setCam(Number(e.target.value))}
                      aria-label="footage camera">
                      {(maInfo?.angles ?? []).map((a) => (
                        <option key={a.index} value={a.index}>
                          {a.label || `Camera ${a.index + 1}`}
                        </option>
                      ))}
                    </select>
                    <button type="button" className={btnGhost}
                      onClick={() => setMuted((m) => !m)}
                      title={muted ? "Unmute" : "Mute"}>
                      {muted ? <VolumeX size={12} /> : <Volume2 size={12} />}
                      {muted ? "Muted" : "Sound"}
                    </button>
                  </div>
                  <div className="relative bg-black">
                    {vidSrc && (
                      <video ref={videoRef} src={vidSrc} muted playsInline
                        preload="auto" className="w-full max-h-72"
                        onLoadedMetadata={(e) => {
                          const v = e.currentTarget;
                          setVidDur(v.duration || 0);
                          const ft = t - off;
                          if (ft >= 0) v.currentTime = ft;
                        }} />
                    )}
                    {!covered && (
                      <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/70 text-xs text-zinc-400">
                        {(maInfo?.angles[camClamped]?.label
                          || `Camera ${camClamped + 1}`)} not covering this moment
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
