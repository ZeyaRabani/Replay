import { useRef } from "react";
import { fmtClock } from "../lib/time";
import type { Candidate, EventType } from "../types";

export const TYPE_COLORS: Record<string, string> = {
  goal: "#f87171",
  shot: "#fb923c",
  goalmouth: "#fbbf24",
  crowd: "#c084fc",
  attack: "#facc15",
  chance: "#facc15",
  excitement: "#60a5fa",
  other: "#9ca3af",
};

export const TYPE_LABEL: Record<EventType, string> = {
  goal: "GOAL",
  shot: "Shot on goal",
  goalmouth: "Goalmouth action",
  crowd: "Crowd reaction",
  attack: "Attack",
  chance: "Attack",
  excitement: "Excitement",
  other: "Other",
};

interface Props {
  duration: number;
  candidates: Candidate[];
  selectedId: string | null;
  playhead: number;
  onSeek: (t: number) => void;
  onSelect: (c: Candidate) => void;
}

export default function Timeline(props: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const { duration } = props;
  const W = 1000;
  const H = 80;
  const base = 64;
  const x = (t: number) => (duration > 0 ? (t / duration) * W : 0);

  const toTime = (clientX: number) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || duration <= 0) return 0;
    return Math.max(0, Math.min(duration, ((clientX - rect.left) / rect.width) * duration));
  };

  const tickEvery = duration > 3600 ? 600 : 300;
  const ticks: number[] = [];
  for (let t = 0; t <= duration; t += tickEvery) ticks.push(t);
  const fmt = fmtClock;

  return (
    <div className="bg-zinc-900 rounded-lg p-2 overflow-x-auto">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full mob:min-w-[640px] cursor-pointer"
        onClick={(e) => props.onSeek(toTime(e.clientX))}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={base} y2={base + 6} stroke="#52525b" strokeWidth={1} />
            <text x={x(t)} y={H - 2} fontSize={9} fill="#71717a" textAnchor="middle">
              {fmt(t)}
            </text>
          </g>
        ))}
        <line x1={0} x2={W} y1={base} y2={base} stroke="#52525b" strokeWidth={2} />
        {props.candidates.map((c) => {
          const h = 8 + c.confidence * 40;
          return (
            <g
              key={c.id}
              onClick={(e) => {
                e.stopPropagation();
                props.onSelect(c);
                props.onSeek(c.t);
              }}
              className="cursor-pointer"
              opacity={c.status === "rejected" ? 0.3 : 1}
            >
              <line
                x1={x(c.t)}
                x2={x(c.t)}
                y1={base - h}
                y2={base}
                stroke={TYPE_COLORS[c.type] ?? "#9ca3af"}
                strokeWidth={3}
              />
              {c.id === props.selectedId && (
                <circle cx={x(c.t)} cy={base - h} r={5} fill="none" stroke="#fff" strokeWidth={1.5} />
              )}
              <title>{`${TYPE_LABEL[c.type] ?? c.type} @ ${fmt(c.t)} (${(c.confidence * 100).toFixed(0)}%)`}</title>
            </g>
          );
        })}
        <line x1={x(props.playhead)} x2={x(props.playhead)} y1={0} y2={base + 4} stroke="#fff" strokeWidth={1.5} />
      </svg>
    </div>
  );
}
