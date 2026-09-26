import { useRef, useState } from "react";
import { useProjectApi } from "../api";
import type { AngleInfo, ZoneKeyframe, ZonePolygon } from "../types";

interface Rect {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

const rectToPoly = (r: Rect): ZonePolygon => [
  [r.x1, r.y1], [r.x2, r.y1], [r.x2, r.y2], [r.x1, r.y2],
];

const polyBounds = (p: ZonePolygon): Rect => {
  const xs = p.map((pt) => pt[0]);
  const ys = p.map((pt) => pt[1]);
  return { x1: Math.min(...xs), y1: Math.min(...ys), x2: Math.max(...xs), y2: Math.max(...ys) };
};

const clamp = (v: number, hi: number) => Math.min(Math.max(0, v), Math.max(0, hi));

export function zoneStillTimes(duration: number | null): [number, number, number] {
  const d = duration ?? 120;
  return [clamp(60, d - 1), clamp(d / 2, d - 1), clamp(d - 300, d - 1)];
}

interface Props {
  angle: AngleInfo;
  /** v2 keyframes saved for this angle (may be empty / a single legacy kf). */
  keyframes: ZoneKeyframe[];
  onChange: (kfs: ZoneKeyframe[]) => void;
}

function OverlayRects({ zones, draft }: { zones: ZonePolygon[]; draft?: Rect | null }) {
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox="0 0 1 1" preserveAspectRatio="none">
      {zones.map((p, i) => {
        const r = polyBounds(p);
        return (
          <rect key={i} x={r.x1} y={r.y1} width={r.x2 - r.x1} height={r.y2 - r.y1}
            fill="rgba(251,146,60,0.25)" stroke="#fb923c" strokeWidth={0.004} />
        );
      })}
      {draft && (
        <rect x={Math.min(draft.x1, draft.x2)} y={Math.min(draft.y1, draft.y2)}
          width={Math.abs(draft.x2 - draft.x1)} height={Math.abs(draft.y2 - draft.y1)}
          fill="rgba(251,146,60,0.15)" stroke="#fb923c" strokeWidth={0.004}
          strokeDasharray="0.01" />
      )}
    </svg>
  );
}

function Still({
  url, t, zones, onPaint, onClear, onCopyAll,
}: {
  url: string;
  t: number;
  zones: ZonePolygon[];
  onPaint: (poly: ZonePolygon) => void;
  onClear: () => void;
  onCopyAll: () => void;
}) {
  const box = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState<Rect | null>(null);

  const norm = (e: React.PointerEvent): [number, number] => {
    const r = box.current!.getBoundingClientRect();
    return [
      Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    ];
  };
  const onDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const [x, y] = norm(e);
    setDraft({ x1: x, y1: y, x2: x, y2: y });
  };
  const onMove = (e: React.PointerEvent) => {
    if (!draft) return;
    const [x, y] = norm(e);
    setDraft((d) => (d ? { ...d, x2: x, y2: y } : d));
  };
  const onUp = () => {
    if (!draft) return;
    const r = {
      x1: Math.min(draft.x1, draft.x2), y1: Math.min(draft.y1, draft.y2),
      x2: Math.max(draft.x1, draft.x2), y2: Math.max(draft.y1, draft.y2),
    };
    setDraft(null);
    if (r.x2 - r.x1 > 0.01 && r.y2 - r.y1 > 0.01) onPaint(rectToPoly(r));
  };

  return (
    <div className="flex-1 min-w-0 flex flex-col gap-1">
      <div
        ref={box}
        className="relative select-none touch-none cursor-crosshair"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      >
        <img
          src={url}
          alt={`@${Math.round(t)}s`}
          className="w-full rounded border border-zinc-800 pointer-events-none"
          draggable={false}
        />
        <OverlayRects zones={zones} draft={draft} />
        <span className="absolute bottom-1 left-1 text-[9px] font-mono bg-zinc-950/70 rounded px-1 text-zinc-300">
          {Math.round(t)}s · draw here
        </span>
      </div>
      <div className="flex items-center gap-2 text-[10px]">
        <button
          className="text-zinc-400 hover:text-zinc-200 underline disabled:opacity-40"
          disabled={zones.length === 0}
          onClick={onClear}
        >
          Clear ({zones.length})
        </button>
        <button
          className="text-zinc-400 hover:text-zinc-200 underline disabled:opacity-40"
          disabled={zones.length === 0}
          onClick={onCopyAll}
          title="Copy this still's zones to the other two"
        >
          Copy to all
        </button>
      </div>
    </div>
  );
}

export default function ZoneEditor({ angle, keyframes, onChange }: Props) {
  const api = useProjectApi();
  const times = zoneStillTimes(angle.duration);

  // map saved keyframes onto the three stills by nearest t (a single
  // legacy keyframe lands on whichever still is closest, others empty);
  // each saved keyframe is assigned to exactly one still
  const assigned = times.map(() => [] as ZonePolygon[]);
  keyframes.forEach((kf) => {
    let bi = 0;
    times.forEach((t, si) => {
      if (Math.abs(kf.t - t) < Math.abs(kf.t - times[bi])) bi = si;
    });
    assigned[bi] = [...assigned[bi], ...(kf.zones ?? [])];
  });
  const total = assigned.reduce((n, z) => n + z.length, 0);

  const emit = (next: ZonePolygon[][]) =>
    onChange(times.map((t, i) => ({ t, zones: next[i] })).filter((kf) => kf.zones.length > 0));

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-[11px] text-zinc-400">
        <span className="font-medium text-zinc-300">a{angle.index} · {angle.label}</span>
        <span className="ml-auto text-[10px] text-zinc-500">{total} zone{total === 1 ? "" : "s"}</span>
      </div>
      <div className="flex gap-2 mob:flex-col">
        {times.map((t, i) => (
          <Still
            key={t}
            t={t}
            url={api.angleFrameUrl(angle.index, t)}
            zones={assigned[i]}
            onPaint={(poly) => {
              const next = assigned.map((z) => [...z]);
              next[i] = [...next[i], poly];
              emit(next);
            }}
            onClear={() => {
              const next = assigned.map((z) => [...z]);
              next[i] = [];
              emit(next);
            }}
            onCopyAll={() => emit([assigned[i], assigned[i], assigned[i]])}
          />
        ))}
      </div>
    </div>
  );
}
