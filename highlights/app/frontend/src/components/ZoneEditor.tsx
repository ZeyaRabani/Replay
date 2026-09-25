import { useRef, useState } from "react";
import { useProjectApi } from "../api";
import type { AngleInfo, ZonePolygon } from "../types";

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

interface Props {
  angle: AngleInfo;
  zones: ZonePolygon[];
  onChange: (zones: ZonePolygon[]) => void;
  onTChange?: (t: number) => void;
}

export default function ZoneEditor({ angle, zones, onChange, onTChange }: Props) {
  const api = useProjectApi();
  const box = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState<Rect | null>(null);
  const [t, setT] = useState<number | null>(null);
  const duration = angle.duration ?? 600;

  const norm = (e: React.MouseEvent): [number, number] => {
    const r = box.current!.getBoundingClientRect();
    return [
      Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    ];
  };

  const onDown = (e: React.MouseEvent) => {
    const [x, y] = norm(e);
    setDraft({ x1: x, y1: y, x2: x, y2: y });
  };
  const onMove = (e: React.MouseEvent) => {
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
    if (r.x2 - r.x1 > 0.01 && r.y2 - r.y1 > 0.01) onChange([...zones, rectToPoly(r)]);
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-[11px] text-zinc-400">
        <span className="font-medium text-zinc-300">a{angle.index} · {angle.label}</span>
        <span>frame t</span>
        <input
          type="range"
          min={0}
          max={Math.floor(duration)}
          step={10}
          value={t ?? Math.floor(duration * 0.3)}
          onChange={(e) => {
            setT(Number(e.target.value));
            onTChange?.(Number(e.target.value));
          }}
          className="w-28 accent-amber-400"
        />
        <span className="font-mono">{t ?? Math.floor(duration * 0.3)}s</span>
        <button
          className="ml-auto text-[10px] text-zinc-400 hover:text-zinc-200 underline disabled:opacity-40"
          disabled={zones.length === 0}
          onClick={() => onChange([])}
        >
          Clear ({zones.length})
        </button>
      </div>
      <div
        ref={box}
        className="relative w-full max-w-md select-none cursor-crosshair"
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onUp}
      >
        <img
          src={api.angleFrameUrl(angle.index, t ?? undefined)}
          alt={`angle ${angle.index} frame`}
          className="w-full rounded border border-zinc-800 pointer-events-none"
          draggable={false}
        />
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
        >
          {zones.map((p, i) => {
            const r = polyBounds(p);
            return (
              <rect
                key={i}
                x={r.x1} y={r.y1}
                width={r.x2 - r.x1} height={r.y2 - r.y1}
                fill="rgba(251,146,60,0.25)"
                stroke="#fb923c"
                strokeWidth={0.004}
              />
            );
          })}
          {draft && (
            <rect
              x={Math.min(draft.x1, draft.x2)}
              y={Math.min(draft.y1, draft.y2)}
              width={Math.abs(draft.x2 - draft.x1)}
              height={Math.abs(draft.y2 - draft.y1)}
              fill="rgba(251,146,60,0.15)"
              stroke="#fb923c"
              strokeWidth={0.004}
              strokeDasharray="0.01"
            />
          )}
        </svg>
      </div>
    </div>
  );
}
