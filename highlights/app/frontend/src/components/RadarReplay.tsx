import { ChevronDown, ChevronRight, Loader2, Maximize2, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useProjectApi } from "../api";
import { applyH, homography } from "../lib/homography";
import { fetchPlayers } from "../lib/players";
import { fmtClock } from "../lib/time";
import type { PlayersPaths } from "../types";

const card = "rounded-lg border border-zinc-800 bg-zinc-900 p-4";
const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnGhost = "flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-xs disabled:opacity-40";

const CORNER_LABELS = ["Near-left corner", "Near-right corner", "Far-right corner", "Far-left corner"];
// pitch-plane coords (metres) for those corners: near = bottom edge
const PITCH_W = 100, PITCH_H = 64;
const DST: [number, number][] = [[0, PITCH_H], [PITCH_W, PITCH_H], [PITCH_W, 0], [0, 0]];
const DEFAULT_CORNERS: [number, number][] = [[0, 1], [1, 1], [1, 0], [0, 0]];
const TEAM_HEX: Record<string, string> = { A: "#22c55e", B: "#f97316" };

/** interpolated (fx,fy) of a track's pts at shared t, or null if the
 *  track isn't alive; second value = alpha fade after the last point */
function posAt(pts: [number, number, number][], t: number): [number, number, number] | null {
  if (!pts.length || t < pts[0][0] - 0.5) return null;
  const last = pts[pts.length - 1][0];
  if (t > last) {
    const a = 1 - Math.min(1, (t - last) / 1.5);
    return a <= 0 ? null : [pts[pts.length - 1][1], pts[pts.length - 1][2], a];
  }
  // binary search for the sample just before t
  let lo = 0, hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid][0] <= t) lo = mid; else hi = mid - 1;
  }
  const a = pts[lo];
  const b = pts[Math.min(lo + 1, pts.length - 1)];
  const span = Math.max(1e-6, b[0] - a[0]);
  const f = Math.min(1, Math.max(0, (t - a[0]) / span));
  return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, 1];
}

function drawPitch(cv: HTMLCanvasElement, w: number, h: number, pad: number) {
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#14532d";
  ctx.fillRect(0, 0, w, h);
  const sx = (w - 2 * pad) / PITCH_W, sy = (h - 2 * pad) / PITCH_H;
  const X = (x: number) => pad + x * sx, Y = (y: number) => pad + y * sy;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = Math.max(1, w / 900);
  const line = (x1: number, y1: number, x2: number, y2: number) => {
    ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x2), Y(y2)); ctx.stroke();
  };
  const rect = (x: number, y: number, rw: number, rh: number) => {
    ctx.strokeRect(X(x), Y(y), rw * sx, rh * sy);
  };
  rect(0, 0, PITCH_W, PITCH_H);
  line(PITCH_W / 2, 0, PITCH_W / 2, PITCH_H);
  const R = 9.15 * Math.min(sx, sy);
  ctx.beginPath(); ctx.arc(X(PITCH_W / 2), Y(PITCH_H / 2), R, 0, Math.PI * 2); ctx.stroke();
  const boxD = 16.5 * PITCH_W / PITCH_W, boxW = 40.3;             // 18-yard box
  rect(0, (PITCH_H - boxW) / 2, boxD, boxW);
  rect(PITCH_W - boxD, (PITCH_H - boxW) / 2, boxD, boxW);
  const sixD = 5.5, sixW = 18.3;
  rect(0, (PITCH_H - sixW) / 2, sixD, sixW);
  rect(PITCH_W - sixD, (PITCH_H - sixW) / 2, sixD, sixW);
  ctx.beginPath(); ctx.arc(X(11), Y(PITCH_H / 2), 2, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill();
  ctx.beginPath(); ctx.arc(X(PITCH_W - 11), Y(PITCH_H / 2), 2, 0, Math.PI * 2); ctx.fill();
}

export default function RadarReplay({ onSeek }: { onSeek?: (t: number) => void }) {
  const api = useProjectApi();
  const [open, setOpen] = useState(false);
  const [paths, setPaths] = useState<PlayersPaths | null>(null);
  const [corners, setCorners] = useState<[number, number][] | null>(null);
  const [setting, setSetting] = useState(false);
  const [picked, setPicked] = useState<[number, number][]>([]);
  const [rosterNames, setRosterNames] = useState<Record<string, string>>({});
  const [teamHex, setTeamHex] = useState<Record<string, string>>(TEAM_HEX);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [t, setT] = useState(0);
  const cvRef = useRef<HTMLCanvasElement>(null);
  const pickRef = useRef<HTMLCanvasElement>(null);
  const pickImg = useRef<HTMLImageElement | null>(null);
  const dragIdx = useRef<number | null>(null);
  const [pickReady, setPickReady] = useState(false);
  const smooth = useRef(new Map<number, [number, number]>());
  const lastTs = useRef(0);

  const lo = paths?.window_shared?.[0] ?? 0;
  const hi = paths?.window_shared?.[1] ?? 0;

  useEffect(() => {
    if (!open || paths) return;
    Promise.all([api.playerPaths(), api.radarPitch(), fetchPlayers(api)])
      .then(([p, pitch, pl]) => {
        setPaths(p);
        setCorners(pitch.corners);
        setT(p.window_shared?.[0] ?? 0);
        if (pl) {
          const names: Record<string, string> = {};
          for (const r of pl.roster.players) names[r.id] = r.name;
          setRosterNames(names);
          if (pl.teams)
            setTeamHex({
              A: pl.teams.A?.hex || TEAM_HEX.A,
              B: pl.teams.B?.hex || TEAM_HEX.B,
            });
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [open, paths, api]);

  const H = useMemo(() => {
    const c = (setting || corners == null)
      ? (picked.length === 4 ? picked : DEFAULT_CORNERS)
      : corners;
    return homography(c, DST);
  }, [corners, picked, setting]);

  // animation loop
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

  // draw
  useEffect(() => {
    const cv = cvRef.current;
    if (!cv || !paths || !H) return;
    const W = cv.clientWidth || 800;
    const Hh = Math.round(W * (PITCH_H / PITCH_W)) + 24;
    if (cv.width !== W) cv.width = W;
    if (cv.height !== Hh) cv.height = Hh;
    const pad = 12;
    drawPitch(cv, W, Hh, pad);
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const sx = (W - 2 * pad) / PITCH_W, sy = (Hh - 2 * pad) / PITCH_H;
    const X = (x: number) => pad + x * sx, Y = (y: number) => pad + y * sy;
    let on = 0;
    for (const tr of paths.tracks) {
      if (tr.hidden) continue;
      const p = posAt(tr.pts, t);
      if (!p) continue;
      let [fx, fy, alpha] = p;
      const [px0, py0] = applyH(H, fx, fy);
      // clamp to pitch +5%
      const px = Math.min(PITCH_W * 1.05, Math.max(-PITCH_W * 0.05, px0));
      const py = Math.min(PITCH_H * 1.05, Math.max(-PITCH_H * 0.05, py0));
      // EMA smooth (alpha ~0.5)
      const prev = smooth.current.get(tr.id);
      const sm: [number, number] = prev && alpha === 1
        ? [prev[0] + 0.5 * (px - prev[0]), prev[1] + 0.5 * (py - prev[1])]
        : [px, py];
      smooth.current.set(tr.id, sm);
      on += 1;
      const col = tr.team ? (teamHex[tr.team] ?? "#a1a1aa") : "#a1a1aa";
      ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
      ctx.beginPath();
      ctx.arc(X(sm[0]), Y(sm[1]), Math.max(4, W / 160), 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.stroke();
      const nm = tr.player_id ? rosterNames[tr.player_id] : null;
      if (nm) {
        ctx.fillStyle = "#e4e4e7";
        ctx.font = "10px sans-serif";
        ctx.fillText(nm, X(sm[0]) + 7, Y(sm[1]) + 3);
      }
      ctx.globalAlpha = 1;
    }
    const b = posAt(paths.ball, t);
    if (b) {
      ctx.globalAlpha = b[2];
      ctx.beginPath();
      const bp = applyH(H, b[0], b[1]);
      ctx.arc(X(Math.min(PITCH_W * 1.05, Math.max(-PITCH_W * 0.05, bp[0]))),
              Y(Math.min(PITCH_H * 1.05, Math.max(-PITCH_H * 0.05, bp[1]))),
              Math.max(2.5, W / 320), 0, Math.PI * 2);
      ctx.fillStyle = "#fafafa";
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    cv.dataset.nOnRadar = String(on);
    const count = cv.parentElement?.querySelector("[data-radar-count]");
    if (count) count.textContent = `${on} on radar`;
  }, [t, paths, H, teamHex, rosterNames]);

  const saveCorners = () => {
    void api.putRadarPitch(picked, paths?.frame_t ?? null)
      .then((d) => { setCorners(d.corners); setSetting(false); setPicked([]); })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  };

  // corner-picker: image drawn centred with 50% padding on every side,
  // so clicks can land off-image -> normalised coords in [-0.5, 1.5]
  const IMG_FRAC = 0.5; // image occupies the middle half of each axis

  useEffect(() => {
    if (!setting || !paths) return;
    setPickReady(false);
    const img = new Image();
    img.onload = () => { pickImg.current = img; setPickReady(true); };
    img.src = api.angleFrameUrl(paths.ref_angle, paths.frame_t ?? undefined);
  }, [setting, paths, api]);

  useEffect(() => {
    const cv = pickRef.current;
    const img = pickImg.current;
    if (!cv || !img || !setting) return;
    const W = cv.clientWidth || 800;
    const iw = W * IMG_FRAC, ih = iw * (img.naturalHeight / img.naturalWidth);
    const ox = W * ((1 - IMG_FRAC) / 2), oy = ih / 2;
    if (cv.width !== W) cv.width = W;
    cv.height = 2 * ih;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#18181b";
    ctx.fillRect(0, 0, W, cv.height);
    ctx.drawImage(img, ox, oy, iw, ih);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.strokeRect(ox, oy, iw, ih);
    const SX = (fx: number) => ox + fx * iw, SY = (fy: number) => oy + fy * ih;
    if (picked.length >= 2) {
      ctx.beginPath();
      picked.forEach(([x, y], i) =>
        i ? ctx.lineTo(SX(x), SY(y)) : ctx.moveTo(SX(x), SY(y)));
      if (picked.length === 4) ctx.closePath();
      ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 1.5; ctx.stroke();
      if (picked.length === 4) {
        ctx.fillStyle = "rgba(251,191,36,0.12)"; ctx.fill();
      }
    }
    picked.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(SX(x), SY(y), 7, 0, Math.PI * 2);
      ctx.fillStyle = "#fbbf24"; ctx.fill();
      ctx.strokeStyle = "#18181b"; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = "#18181b";
      ctx.font = "bold 9px sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), SX(x), SY(y));
    });
  }, [picked, setting, pickReady]);

  const pickPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cv = e.currentTarget;
    const r = cv.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    // canvas -> image-normalised coords (padding is 50% of the image)
    const ix = (fx - (1 - IMG_FRAC) / 2) / IMG_FRAC;
    const iy = (fy - (1 - IMG_FRAC) / 2) / IMG_FRAC;
    return [ix, iy] as [number, number];
  };

  const onPickDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const [ix, iy] = pickPos(e);
    const hit = picked.findIndex(([x, y]) =>
      Math.hypot(x - ix, y - iy) < 0.06);
    if (hit >= 0) {
      dragIdx.current = hit;
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };
  const onPickMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragIdx.current == null) return;
    const [ix, iy] = pickPos(e);
    setPicked((v) => v.map((p, i) =>
      i === dragIdx.current ? [ix, iy] as [number, number] : p));
  };
  const onPickUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragIdx.current != null) {
      dragIdx.current = null;
      return;
    }
    if (picked.length >= 4) return;
    const [ix, iy] = pickPos(e);
    setPicked((v) => [...v, [ix, iy]]);
  };

  const headerBtn = (
    <button type="button" className="flex items-center gap-2 w-full text-left"
      onClick={() => setOpen((o) => !o)} aria-expanded={open}>
      {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
      <Maximize2 size={14} className="text-zinc-500" />
      <span className={head}>Radar replay (optional)</span>
      {paths && (
        <span className="ml-auto text-[11px] text-zinc-500" data-radar-count>
          {paths.tracks.filter((x) => !x.hidden).length} tracks
        </span>
      )}
    </button>
  );

  return (
    <div className={`${card} min-w-0`}>
      {headerBtn}
      {open && (
        <div className="mt-3 flex flex-col gap-3">
          {error && <div className="text-xs text-red-300">{error}</div>}
          {!paths ? (
            <div className="text-sm text-zinc-500 flex items-center gap-2">
              <Loader2 size={14} className="animate-spin" />
              {error ? "Radar unavailable — run Player analysis first." : "Loading…"}
            </div>
          ) : (
            <>
              {/* step 1: pitch corners */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs text-zinc-400">
                  {corners ? "Pitch corners set." : "No pitch corners set — using the full frame (distorted)."}
                </span>
                <button className={btnGhost}
                  onClick={() => {
                    setSetting((s) => !s);
                    setPicked(corners ? [...corners] as [number, number][] : []);
                  }}>
                  {setting ? "Cancel" : corners ? "Re-set corners" : "Set pitch corners"}
                </button>
              </div>
              {setting && (
                <div className="flex flex-col gap-1.5">
                  <div className="text-[11px] text-zinc-400">
                    Click the 4 pitch corners in order — they may be outside
                    the picture; click where they would be.
                    Next: {CORNER_LABELS[picked.length] ?? "done — drag a marker to adjust"}
                    {" "}
                    <button className="text-amber-300 hover:text-amber-200 disabled:opacity-40"
                      disabled={picked.length !== 4} onClick={saveCorners}>Save</button>
                    {" "}
                    <button className="text-zinc-400 hover:text-zinc-200 disabled:opacity-40"
                      disabled={!picked.length} onClick={() => setPicked([])}>Reset</button>
                  </div>
                  <canvas ref={pickRef}
                    className="w-full max-w-[960px] rounded cursor-crosshair touch-none"
                    onPointerDown={onPickDown}
                    onPointerMove={onPickMove}
                    onPointerUp={onPickUp}
                  />
                </div>
              )}

              {/* step 2: radar */}
              <canvas ref={cvRef} className="w-full rounded" />
              <div className="flex items-center gap-2 flex-wrap">
                <button className={btnGhost}
                  onClick={() => setPlaying((p) => !p)}>
                  {playing ? <Pause size={12} /> : <Play size={12} />}
                  {playing ? "Pause" : "Play"}
                </button>
                {[1, 2, 4].map((s) => (
                  <button key={s}
                    className={`rounded px-2 py-1 text-xs ${speed === s ? "bg-amber-500 text-zinc-900 font-semibold" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}
                    onClick={() => setSpeed(s)}>
                    {s}×
                  </button>
                ))}
                <span className="text-[11px] font-mono text-zinc-400 ml-1">
                  {fmtClock(t - lo)} / {fmtClock(hi - lo)}
                </span>
                {onSeek && (
                  <button className={btnGhost}
                    onClick={() => onSeek(Math.max(0, t - lo))}>
                    Jump video here
                  </button>
                )}
              </div>
              <input
                type="range"
                className="w-full accent-amber-400"
                min={lo} max={hi} step={0.1}
                value={t}
                onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }}
                onMouseUp={() => onSeek?.(t - lo)}
                aria-label="scrub radar"
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}
