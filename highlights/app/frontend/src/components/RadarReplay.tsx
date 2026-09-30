import { ChevronDown, ChevronRight, Loader2, Maximize2, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useProjectApi } from "../api";
import { applyH, homography, type Mat3 } from "../lib/homography";
import { DEFAULT_PITCH, drawPitch, pitchView } from "../lib/pitch";
import { IDENTITIES_CHANGED_EVENT, fetchPlayers, identityHex } from "../lib/players";
import { fmtClock } from "../lib/time";
import type { CalibResponse, PitchDims, PlayersPaths, RadarPitch } from "../types";
import CameraCalib, { rmsTone } from "./CameraCalib";
import CameraPlacement from "./CameraPlacement";

const card = "card p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnGhost = "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";

const TEAM_HEX: Record<string, string> = { A: "#22c55e", B: "#f97316" };
const TRAIL_S = 1.5;
const VISIBLE_STEP_S = 0.5;

type Pt = [number, number, number];

/** interpolated (a,b) of a track's pts at shared t, or null if the track
 *  isn't alive; third value = alpha fade after the last point */
function posAt(pts: Pt[], t: number): Pt | null {
  if (!pts.length || t < pts[0][0] - 0.5) return null;
  const last = pts[pts.length - 1][0];
  if (t > last) {
    const a = 1 - Math.min(1, (t - last) / 1.5);
    return a <= 0 ? null : [pts[pts.length - 1][1], pts[pts.length - 1][2], a];
  }
  let lo = 0, hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid][0] <= t) lo = mid; else hi = mid - 1;
  }
  const a = pts[lo];
  const b = pts[Math.min(lo + 1, pts.length - 1)];
  const f = Math.min(1, Math.max(0, (t - a[0]) / Math.max(1e-6, b[0] - a[0])));
  return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, 1];
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
    if (!open) return;
    const load = () => void api.identities()
      .then((d) => setIdentNames(Object.fromEntries(d.identities.map((i) => [i.id, i.name]))))
      .catch(() => setIdentNames({}));
    load();
    // re-linking renumbers identities: refetch the paths too
    const relink = () => { load(); setPaths(null); };
    window.addEventListener(IDENTITIES_CHANGED_EVENT, relink);
    return () => window.removeEventListener(IDENTITIES_CHANGED_EVENT, relink);
  }, [open, api]);

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

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const step = (ts: number) => {
      if (lastTs.current) {
        setT((v) => {
          const nt = v + ((ts - lastTs.current) / 1000) * speed;
          return nt >= hi ? lo : nt;
        });
      }
      lastTs.current = ts;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => { cancelAnimationFrame(raf); lastTs.current = 0; };
  }, [playing, speed, lo, hi]);

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
    const dots: { tr: PlayersPaths["tracks"][number]; m: [number, number]; a: number; n: number }[] = [];
    const byIdent = new Map<string, (typeof dots)[number]>();
    for (const tr of paths.tracks) {
      if (tr.hidden) continue;
      const p = posAt(tr.pts, t);
      if (!p) continue;
      let m = toM(p[0], p[1]);
      if (!inPitch) {
        // EMA smoothing of the jittery single-camera projection
        const prev = smooth.current.get(tr.id);
        if (prev && p[2] === 1) m = [prev[0] + 0.5 * (m[0] - prev[0]), prev[1] + 0.5 * (m[1] - prev[1])];
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
      const dot = { tr, m: m as [number, number], a: p[2], n: 1 };
      dots.push(dot);
      if (iid) byIdent.set(iid, dot);
    }
    for (const { tr, m, a } of dots) {
      if (a === 1) on += 1;
      const x = v.X(m[0]), y = v.Y(m[1]);
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(x, y + 1.5, r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      const team = tr.team ? (teamHex[tr.team] ?? "#a1a1aa") : "#d4d4d8";
      const iid = tr.identity_id ?? null;
      ctx.fillStyle = iid ? identityHex(iid) : team;
      ctx.fill();
      ctx.lineWidth = iid ? 2.5 : 2;
      ctx.strokeStyle = iid ? team : "#09090b";
      ctx.stroke();
      const nm = (iid && identNames[iid]) || (tr.player_id ? rosterNames[tr.player_id] : null);
      if (nm) labels.push([nm, x, y - r - 5, a]);
      if (a === 1) hits.push({ x, y, id: tr.id, ident: iid });
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
  }, [t, paths, H, inPitch, pitch, teamHex, rosterNames, identNames]);

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
    const nm = best.ident ? identNames[best.ident] : null;
    const label = best.ident
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
            </>
          )}
        </div>
      )}
    </div>
  );
}
