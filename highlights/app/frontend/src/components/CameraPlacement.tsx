import { Loader2, Save } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useProjectApi } from "../api";
import { DEFAULT_PITCH, pitchShapes } from "../lib/pitch";
import type { CalibCamera, CalibResponse, PitchDims } from "../types";

const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnPrimary =
  "flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 text-white font-semibold font-semibold rounded px-3 py-1.5 text-xs disabled:opacity-40";

const MARGIN = 15;         // metres of grey margin around the pitch
const ARROW_LEN = 7;       // arrow length in metres
const HIT = 2.5;           // drag hit radius in metres
const HANDLE_HIT = 2.0;

function defaultCameras(n: number, p: PitchDims): Record<string, CalibCamera> {
  // spread along the near touchline, facing into the pitch (dir 270 = -y)
  const out: Record<string, CalibCamera> = {};
  for (let i = 0; i < n; i++) {
    const x = p.len_m * ((i + 1) / (n + 1));
    out[String(i)] = { x_m: x, y_m: p.wid_m + 6, dir_deg: 270 };
  }
  return out;
}

/** Top-down pitch; the user drags each camera into position and drags the
 *  arrow tip to set where it faced. Coordinates in pitch metres. */
export default function CameraPlacement({ onSaved, frameT }: {
  onSaved?: (c: CalibResponse) => void;
  frameT?: number | null;
}) {
  const api = useProjectApi();
  const [pitch, setPitch] = useState<PitchDims>(DEFAULT_PITCH);
  const [nAngles, setNAngles] = useState(3);
  const [cams, setCams] = useState<Record<string, CalibCamera> | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const drag = useRef<{ key: string; part: "body" | "handle" } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    Promise.all([
      api.calib().catch(() => null),
      api.multiangle().then((m) => m.angles.length).catch(() => 3),
    ])
      .then(([c, n]) => {
        const p = c?.pitch ?? DEFAULT_PITCH;
        setPitch(p);
        const na = Math.max(1, n);
        setNAngles(na);
        const savedCams = c?.cameras ?? {};
        setCams({ ...defaultCameras(na, p), ...savedCams });
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api]);

  const toM = (ev: { clientX: number; clientY: number }): [number, number] => {
    const svg = svgRef.current!;
    const r = svg.getBoundingClientRect();
    const W = pitch.len_m + 2 * MARGIN, H = pitch.wid_m + 2 * MARGIN;
    return [
      -MARGIN + ((ev.clientX - r.left) / r.width) * W,
      -MARGIN + ((ev.clientY - r.top) / r.height) * H,
    ];
  };

  const hitCam = (x: number, y: number) => {
    for (const [k, c] of Object.entries(cams ?? {})) {
      const a = (c.dir_deg * Math.PI) / 180;
      const hx = c.x_m + ARROW_LEN * Math.cos(a);
      const hy = c.y_m + ARROW_LEN * Math.sin(a);
      if (Math.hypot(x - hx, y - hy) <= HANDLE_HIT) return { key: k, part: "handle" as const };
      if (Math.hypot(x - c.x_m, y - c.y_m) <= HIT) return { key: k, part: "body" as const };
    }
    return null;
  };

  const onDown = (ev: React.PointerEvent) => {
    const [x, y] = toM(ev);
    const hit = hitCam(x, y);
    if (!hit) return;
    drag.current = hit;
    (ev.target as Element).setPointerCapture?.(ev.pointerId);
  };
  const onMove = (ev: React.PointerEvent) => {
    const d = drag.current;
    if (!d || !cams) return;
    const [x, y] = toM(ev);
    setCams((all) => {
      if (!all) return all;
      const c = all[d.key];
      const next = { ...c };
      if (d.part === "body") {
        next.x_m = Math.max(-30, Math.min(pitch.len_m + 30, x));
        next.y_m = Math.max(-30, Math.min(pitch.wid_m + 30, y));
      } else {
        next.dir_deg = Math.round(
          ((Math.atan2(y - c.y_m, x - c.x_m) * 180) / Math.PI + 360) % 360);
      }
      return { ...all, [d.key]: next };
    });
    setDirty(true);
  };
  const onUp = () => { drag.current = null; };

  const save = async () => {
    if (!cams) return;
    setSaving(true);
    setError(null);
    try {
      const r = await api.putCalibCameras(cams);
      setDirty(false);
      onSaved?.(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const vb = useMemo(
    () => `${-MARGIN} ${-MARGIN} ${pitch.len_m + 2 * MARGIN} ${pitch.wid_m + 2 * MARGIN}`,
    [pitch],
  );

  if (!cams) {
    return (
      <div className="text-sm text-zinc-500 flex items-center gap-2">
        {!error && <Loader2 size={14} className="animate-spin" />}
        {error ?? "Loading…"}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2.5 rounded border border-zinc-800 bg-zinc-950/40 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={head}>Camera positions</span>
        <span className="text-[11px] text-zinc-500">
          Drag each camera to where it stood and point the arrow the way it faced.
        </span>
        <span className={`ml-auto text-[11px] ${dirty ? "text-amber-300" : "text-emerald-400"}`}>
          {dirty ? "Unsaved changes" : "Saved"}
        </span>
        <button type="button" className={btnPrimary} disabled={saving || !dirty}
          onClick={() => void save()}>
          {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
          Save camera positions
        </button>
      </div>
      {error && <div className="text-xs text-red-300">{error}</div>}

      <div className="flex flex-col md:flex-row gap-3 min-w-0">
        <svg ref={svgRef} viewBox={vb}
          className="w-full md:flex-1 rounded bg-zinc-800 touch-none select-none cursor-grab"
          role="img" aria-label="place cameras on the pitch"
          onPointerDown={onDown} onPointerMove={onMove}
          onPointerUp={onUp} onPointerCancel={onUp}>
          {/* pitch markings */}
          <rect x={0} y={0} width={pitch.len_m} height={pitch.wid_m}
            fill="#1d6a3c" />
          {pitchShapes(pitch).map((sh, i) =>
            sh.k === "dot" ? (
              <circle key={i} cx={sh.x} cy={sh.y} r={0.3} fill="rgba(255,255,255,0.85)" />
            ) : (
              <polyline key={i}
                points={[...sh.pts, ...(sh.closed ? [sh.pts[0]] : [])]
                  .map(([x, y]) => `${x},${y}`).join(" ")}
                fill="none"
                stroke={sh.faint ? "rgba(255,255,255,0.45)" : "rgba(255,255,255,0.8)"}
                strokeWidth={0.25} />
            ),
          )}
          {/* camera markers */}
          {Object.entries(cams).map(([k, c]) => {
            const a = (c.dir_deg * Math.PI) / 180;
            const hx = c.x_m + ARROW_LEN * Math.cos(a);
            const hy = c.y_m + ARROW_LEN * Math.sin(a);
            const n = +k + 1;
            return (
              <g key={k}>
                <line x1={c.x_m} y1={c.y_m} x2={hx} y2={hy}
                  stroke="#fbbf24" strokeWidth={0.4} />
                <polygon
                  points={`${hx},${hy} ${hx - 1.6 * Math.cos(a - 0.45)},${hy - 1.6 * Math.sin(a - 0.45)} ${hx - 1.6 * Math.cos(a + 0.45)},${hy - 1.6 * Math.sin(a + 0.45)}`}
                  fill="#fbbf24" />
                <circle cx={hx} cy={hy} r={HANDLE_HIT} fill="transparent" />
                <circle cx={c.x_m} cy={c.y_m} r={2.6} fill="#fbbf24"
                  stroke="#09090b" strokeWidth={0.4} />
                <text x={c.x_m} y={c.y_m + 1.2} textAnchor="middle"
                  fontSize={3.4} fontWeight={700} fill="#18181b"
                  pointerEvents="none">{n}</text>
              </g>
            );
          })}
        </svg>
        <div className="md:w-44 shrink-0 flex md:flex-col flex-row gap-2">
          {Array.from({ length: nAngles }, (_, i) => (
            <div key={i} className="flex md:flex-row flex-col md:items-center items-start gap-1.5">
              <img src={api.angleFrameUrl(i, frameT ?? undefined)}
                alt={`Camera ${i + 1}`}
                className="w-20 md:w-full rounded border border-zinc-700 bg-zinc-800 aspect-video object-cover" />
              <span className="text-[11px] text-zinc-400">Camera {i + 1}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
