import { Loader2, Save, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useProjectApi } from "../api";
import { DEFAULT_PITCH, pitchShapes } from "../lib/pitch";
import { fmtClock, parseClock } from "../lib/time";
import type { CalibLandmark, CalibPoint, CalibResponse, PitchDims } from "../types";
import CalibCanvas from "./CalibCanvas";

const head = "text-xs font-semibold uppercase tracking-wide text-zinc-500";
const btnPrimary =
  "flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-950 font-semibold font-semibold rounded px-3 py-1.5 text-xs disabled:opacity-40";

export const CALIB_SAVED_EVENT = "hl:calib-saved";

export function rmsTone(rms: number | null | undefined): { cls: string; hex: string; label: string } {
  if (rms == null) return { cls: "text-zinc-500", hex: "#52525b", label: "not solved" };
  const label = `±${rms.toFixed(2)} m`;
  if (rms < 1) return { cls: "text-emerald-400", hex: "#34d399", label };
  if (rms < 2.5) return { cls: "text-amber-300", hex: "#fbbf24", label };
  return { cls: "text-red-400", hex: "#f87171", label };
}

/** Tiny top-down pitch highlighting the selected landmark. */
function MiniPitch({ pitch, landmarks, placed, sel }: {
  pitch: PitchDims; landmarks: CalibLandmark[]; placed: Set<string>; sel: string | null;
}) {
  const pad = 4, W = pitch.wid_m;
  const P = (x: number, y: number) => `${x.toFixed(2)},${y.toFixed(2)}`;
  const s = landmarks.find((l) => l.name === sel);
  return (
    <svg viewBox={`${-pad} ${-pad} ${pitch.len_m + 2 * pad} ${W + 2 * pad}`}
      className="w-full rounded bg-[#1a5f36]" role="img" aria-label="pitch map">
      {pitchShapes(pitch).map((sh, i) =>
        sh.k === "dot" ? (
          <circle key={i} cx={sh.x} cy={sh.y} r={0.5} fill="rgba(255,255,255,0.7)" />
        ) : (
          <polyline key={i} points={[...sh.pts, ...(sh.closed ? [sh.pts[0]] : [])].map(([x, y]) => P(x, y)).join(" ")}
            fill="none" stroke={sh.faint ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.7)"} strokeWidth={0.4} />
        ),
      )}
      {landmarks.map((l) => (
        <circle key={l.name} cx={l.x} cy={l.y} r={placed.has(l.name) ? 1.3 : 0.9}
          fill={placed.has(l.name) ? "#fbbf24" : "rgba(24,24,27,0.8)"} stroke="rgba(255,255,255,0.6)" strokeWidth={0.2} />
      ))}
      {s && (
        <g>
          <circle cx={s.x} cy={s.y} r={3.2} fill="none" stroke="#fde68a" strokeWidth={0.7}>
            <animate attributeName="r" values="2.2;4;2.2" dur="1.4s" repeatCount="indefinite" />
          </circle>
          <circle cx={s.x} cy={s.y} r={1.4} fill="#fde68a" />
        </g>
      )}
    </svg>
  );
}

function LandmarkList({ landmarks, placed, sel, onSelect, onRemove }: {
  landmarks: CalibLandmark[]; placed: Set<string>; sel: string | null;
  onSelect: (n: string) => void; onRemove: (n: string) => void;
}) {
  return (
    <ul className="flex flex-col gap-0.5 max-h-80 overflow-y-auto pr-1 text-[11px]">
      {landmarks.map((l, i) => {
        const on = l.name === sel, has = placed.has(l.name);
        return (
          <li key={l.name}>
            <div className={`flex items-center gap-1.5 rounded px-1.5 py-1 ${
              on ? "bg-amber-500/15 ring-1 ring-amber-400" : "hover:bg-zinc-800"}`}>
              <button type="button" className="flex items-center gap-1.5 flex-1 min-w-0 text-left"
                onClick={() => onSelect(l.name)} aria-pressed={on}>
                <span className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center text-[9px] font-bold ${
                  has ? "bg-amber-400 text-zinc-950 font-semibold" : "bg-zinc-800 text-zinc-500 border border-zinc-700"}`}>
                  {i + 1}
                </span>
                <span className={`truncate ${has ? "text-zinc-200" : "text-zinc-400"}`}>{l.label}</span>
              </button>
              {has && (
                <button type="button" onClick={() => onRemove(l.name)}
                  className="shrink-0 text-zinc-500 hover:text-red-300" aria-label={`remove ${l.label}`}>
                  <X size={12} />
                </button>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/** Per-angle camera calibration: place known pitch landmarks on each still. */
export default function CameraCalib({ onSaved, defaultT }: {
  onSaved?: (c: CalibResponse) => void;
  defaultT?: number | null;
}) {
  const api = useProjectApi();
  const [landmarks, setLandmarks] = useState<CalibLandmark[] | null>(null);
  const [pitch, setPitch] = useState<PitchDims>(DEFAULT_PITCH);
  const [saved, setSaved] = useState<CalibResponse | null>(null);
  const [nAngles, setNAngles] = useState(3);
  const [angle, setAngle] = useState(0);
  const [pts, setPts] = useState<Record<string, CalibPoint[]>>({});
  const [dirty, setDirty] = useState<Set<string>>(new Set());
  const [sel, setSel] = useState<string | null>(null);
  const [tText, setTText] = useState(defaultT != null ? fmtClock(defaultT) : "");
  const [frameT, setFrameT] = useState<number | undefined>(defaultT ?? undefined);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.calibLandmarks(),
      api.calib().catch(() => null),
      api.multiangle().then((m) => m.angles.length).catch(() => 3),
    ])
      .then(([lm, c, n]) => {
        setLandmarks(lm.landmarks);
        setPitch(c?.pitch ?? lm.pitch);
        setNAngles(Math.max(1, n));
        if (c) {
          setSaved(c);
          setPts(Object.fromEntries(Object.entries(c.angles).map(([k, a]) => [k, a.pts])));
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [api]);

  const key = String(angle);
  const cur = useMemo(() => pts[key] ?? [], [pts, key]);
  const placed = useMemo(() => new Set(cur.map((p) => p.name)), [cur]);
  const index = useMemo(
    () => new Map((landmarks ?? []).map((l, i) => [l.name, { n: i + 1, label: l.label }])),
    [landmarks],
  );
  const num = useCallback((n: string) => index.get(n)?.n ?? 0, [index]);
  const label = useCallback((n: string) => index.get(n)?.label ?? n, [index]);

  const edit = (fn: (list: CalibPoint[]) => CalibPoint[]) => {
    setPts((all) => ({ ...all, [key]: fn(all[key] ?? []) }));
    setDirty((d) => new Set(d).add(key));
  };
  const place = (name: string, fx: number, fy: number, click: boolean) => {
    edit((list) => [...list.filter((p) => p.name !== name), { name, fx, fy }]);
    if (click) setSel(null);
  };
  const remove = (name: string) => {
    edit((list) => list.filter((p) => p.name !== name));
    if (sel === name) setSel(null);
  };

  const ready = [...dirty].filter((k) => (pts[k]?.length ?? 0) >= 4);
  const angleStatus = (k: string) => {
    const n = pts[k]?.length ?? 0;
    if (n < 4) return { label: "needs ≥4 points", cls: "text-amber-300/80" };
    if (dirty.has(k)) return { label: "unsaved", cls: "text-zinc-400" };
    const fit = saved?.angles[k];
    if (fit?.H) {
      const tone = rmsTone(fit.rms_m);
      return { label: `saved ${tone.label}`, cls: tone.cls };
    }
    return { label: "not solved", cls: "text-zinc-500" };
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const body = Object.fromEntries(ready.map((k) => [k, { pts: pts[k] ?? [] }]));
      const r = await api.putCalib(body);
      setSaved(r);
      setPts((all) => ({
        ...all,
        ...Object.fromEntries(ready.map((k) => [k, r.angles[k]?.pts ?? all[k] ?? []])),
      }));
      setDirty((all) => new Set([...all].filter((k) => !ready.includes(k))));
      onSaved?.(r);
      window.dispatchEvent(new Event(CALIB_SAVED_EVENT));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const applyT = () => {
    const t = parseClock(tText);
    setFrameT(t == null ? undefined : Math.max(0, t));
  };

  if (!landmarks) {
    return (
      <div className="text-sm text-zinc-500 flex items-center gap-2">
        {!error && <Loader2 size={14} className="animate-spin" />}
        {error ?? "Loading calibration…"}
      </div>
    );
  }

  const rms = rmsTone(dirty.has(key) ? undefined : saved?.angles[key]?.rms_m);
  const selLabel = sel ? label(sel) : null;

  return (
    <div className="flex flex-col gap-2.5 rounded border border-zinc-800 bg-zinc-950/40 p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={head}>Camera calibration</span>
        <div className="flex gap-1 ml-2">
          {Array.from({ length: nAngles }, (_, i) => {
            const k = String(i);
            const tone = rmsTone(dirty.has(k) ? undefined : saved?.angles[k]?.rms_m);
            const status = angleStatus(k);
            return (
              <button key={i} type="button" onClick={() => { setAngle(i); setSel(null); }}
                className={`flex items-center gap-1.5 rounded px-2 py-1 text-xs ${
                  angle === i ? "bg-amber-500 text-zinc-950 font-semibold font-semibold" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`}>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: tone.hex }} />
                Angle {i + 1}
                <span className={`text-[10px] ${status.cls}`}>· {status.label}</span>
              </button>
            );
          })}
        </div>
        <label className="ml-auto flex items-center gap-1.5 text-[11px] text-zinc-400">
          Frame time
          <input value={tText} onChange={(e) => setTText(e.target.value)} placeholder="h:mm:ss"
            onKeyDown={(e) => { if (e.key === "Enter") applyT(); }} onBlur={applyT}
            className="w-16 bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 font-mono text-xs text-zinc-200" />
        </label>
        <button type="button" className={btnPrimary} disabled={saving || ready.length === 0}
          title={ready.length ? `Saves angle ${ready.map((k) => +k + 1).join(", ")}` : "No angle with ≥4 unsaved points"}
          onClick={() => void save()}>
          {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
          Save calibration
        </button>
      </div>

      <div className="flex items-center gap-3 flex-wrap text-[11px]">
        <span className="text-zinc-300">
          <span className="font-mono text-amber-300">{cur.length}</span> placed
          <span className="text-zinc-500"> · recommended 8–12</span>
        </span>
        <span className={`flex items-center gap-1.5 ${rms.cls}`}>
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: rms.hex }} />
          {dirty.has(key) ? "unsaved" : `fit ${rms.label}`}
        </span>
        <span className="text-zinc-500 truncate">
          {selLabel
            ? <>Click where <span className="text-amber-200">{selLabel}</span> is — outside the picture is fine.</>
            : "Select a landmark, then click its position. Drag markers to adjust."}
        </span>
      </div>
      {error && <div className="text-xs text-red-300">{error}</div>}

      <div className="flex flex-col md:flex-row gap-3 min-w-0">
        <div className="md:w-60 shrink-0 flex flex-col gap-2">
          <MiniPitch pitch={pitch} landmarks={landmarks} placed={placed} sel={sel} />
          <LandmarkList landmarks={landmarks} placed={placed} sel={sel}
            onSelect={(n) => setSel((s) => (s === n ? null : n))} onRemove={remove} />
        </div>
        <CalibCanvas src={api.angleFrameUrl(angle, frameT)} pts={cur} num={num} label={label}
          sel={sel} onPlace={place} onSelect={setSel} />
      </div>
    </div>
  );
}
