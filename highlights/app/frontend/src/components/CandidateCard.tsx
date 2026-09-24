import { Check, RotateCcw, X } from "lucide-react";
import type { Candidate } from "../types";
import { TYPE_COLORS } from "./Timeline";

const XV_STYLE: Record<string, string> = {
  confirmed: "bg-emerald-800 text-emerald-200",
  pipeline_only: "bg-zinc-700 text-zinc-300",
  visual_only: "bg-purple-800 text-purple-200",
  rejected: "bg-red-900 text-red-200",
};

const STATUS_STYLE: Record<string, string> = {
  pending: "bg-zinc-700 text-zinc-300",
  confirmed: "bg-emerald-700 text-emerald-100",
  rejected: "bg-red-800 text-red-100",
};

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

interface Props {
  c: Candidate;
  selected: boolean;
  onSelect: (c: Candidate) => void;
  onPatch: (id: string, patch: Partial<Candidate>) => void;
  onReset: (id: string) => void;
}

export default function CandidateCard(props: Props) {
  const { c } = props;
  const num =
    "w-16 bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-xs font-mono";

  return (
    <div
      onClick={() => props.onSelect(c)}
      className={`rounded-lg border p-2.5 cursor-pointer transition-colors ${
        props.selected ? "border-amber-400 bg-zinc-800" : "border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
      }`}
    >
      <div className="flex gap-2.5">
        <img
          src={`/api/candidates/${c.id}/thumb.jpg`}
          alt=""
          className="w-24 rounded bg-zinc-800 object-cover"
          loading="lazy"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap text-[10px]">
            <span className="bg-zinc-700 rounded px-1.5 py-0.5 font-mono">#{c.rank}</span>
            <span
              className="rounded px-1.5 py-0.5 font-semibold uppercase"
              style={{ backgroundColor: `${TYPE_COLORS[c.type]}33`, color: TYPE_COLORS[c.type] }}
            >
              {c.type}
            </span>
            <span className={`rounded px-1.5 py-0.5 ${XV_STYLE[c.cross_validation]}`}>{c.cross_validation}</span>
            <span className={`rounded px-1.5 py-0.5 ${STATUS_STYLE[c.status]}`}>{c.status}</span>
            <span className="ml-auto font-mono text-xs text-zinc-300">{fmt(c.t)}</span>
          </div>
          <div className="mt-1.5 flex items-center gap-1.5">
            <div className="flex-1 h-1.5 bg-zinc-700 rounded">
              <div
                className="h-1.5 rounded bg-amber-400"
                style={{ width: `${Math.round(c.confidence * 100)}%` }}
              />
            </div>
            <span className="text-[10px] font-mono text-zinc-400">{Math.round(c.confidence * 100)}%</span>
          </div>
          {c.notes && <p className="mt-1 text-[11px] text-zinc-400 line-clamp-2">{c.notes}</p>}
        </div>
      </div>
      <div
        className="mt-2 flex items-center gap-2 text-[10px] text-zinc-400"
        onClick={(e) => e.stopPropagation()}
      >
        <span>IN</span>
        <input
          className={num}
          type="number"
          step={0.1}
          value={c.clip_start}
          onChange={(e) => props.onPatch(c.id, { clip_start: parseFloat(e.target.value) })}
        />
        <span>OUT</span>
        <input
          className={num}
          type="number"
          step={0.1}
          value={c.clip_end}
          onChange={(e) => props.onPatch(c.id, { clip_end: parseFloat(e.target.value) })}
        />
        <button
          className="p-1 rounded hover:bg-zinc-700"
          title="Reset window"
          onClick={() => props.onReset(c.id)}
        >
          <RotateCcw size={12} />
        </button>
        <span className="flex-1" />
        <button
          className={`flex items-center gap-1 rounded px-2 py-0.5 text-xs ${
            c.status === "confirmed" ? "bg-emerald-700" : "bg-zinc-700 hover:bg-emerald-800"
          }`}
          onClick={() => props.onPatch(c.id, { status: c.status === "confirmed" ? "pending" : "confirmed" })}
        >
          <Check size={12} /> Confirm
        </button>
        <button
          className={`flex items-center gap-1 rounded px-2 py-0.5 text-xs ${
            c.status === "rejected" ? "bg-red-800" : "bg-zinc-700 hover:bg-red-900"
          }`}
          onClick={() => props.onPatch(c.id, { status: c.status === "rejected" ? "pending" : "rejected" })}
        >
          <X size={12} /> Reject
        </button>
      </div>
    </div>
  );
}
