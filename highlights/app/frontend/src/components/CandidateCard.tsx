import { Check, RotateCcw, X } from "lucide-react";
import { useEffect, useState } from "react";
import { useProjectApi } from "../api";
import { fmtClock, parseClock } from "../lib/time";
import type { Candidate, Team } from "../types";
import { TYPE_COLORS, TYPE_LABEL } from "./Timeline";

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

const fmt = fmtClock;

interface Props {
  c: Candidate;
  selected: boolean;
  thumbV: string | undefined;
  onSelect: (c: Candidate) => void;
  onPatch: (id: string, patch: Partial<Candidate> & { team?: Team }) => Promise<boolean>;
  onReset: (id: string) => void;
}

export default function CandidateCard(props: Props) {
  const api = useProjectApi();
  const { c } = props;
  const [inStr, setInStr] = useState(fmtClock(c.clip_start));
  const [outStr, setOutStr] = useState(fmtClock(c.clip_end));
  const [thumbErr, setThumbErr] = useState(false);

  // a new thumb key (new video / candidates reload) gets a fresh try
  useEffect(() => setThumbErr(false), [props.thumbV]);

  // keep local drafts in sync when the server value changes
  useEffect(() => setInStr(fmtClock(c.clip_start)), [c.clip_start]);
  useEffect(() => setOutStr(fmtClock(c.clip_end)), [c.clip_end]);

  const num = "w-16 bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-xs font-mono";

  const commit = async (field: "clip_start" | "clip_end", raw: string, serverVal: number) => {
    const v = parseClock(raw);
    // unchanged, empty or unparseable -> revert to the server value
    if (v === null || v === serverVal) {
      if (field === "clip_start") setInStr(fmtClock(serverVal));
      else setOutStr(fmtClock(serverVal));
      return;
    }
    const ok = await props.onPatch(c.id, { [field]: v });
    if (!ok) {
      // rejected by server (e.g. 422): revert to the server value
      if (field === "clip_start") setInStr(fmtClock(serverVal));
      else setOutStr(fmtClock(serverVal));
    }
  };

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
  };

  return (
    <div
      onClick={() => props.onSelect(c)}
      className={`rounded-lg border p-2.5 cursor-pointer transition-colors ${
        props.selected ? "border-amber-400 bg-zinc-800" : "border-zinc-800 bg-zinc-900 hover:bg-zinc-800"
      }`}
    >
      <div className="flex gap-2.5">
        {props.thumbV === undefined || thumbErr ? (
          <div className="w-24 aspect-video rounded bg-zinc-800 flex items-center justify-center text-[9px] text-zinc-600">
            no frame
          </div>
        ) : (
          <img
            src={api.thumbUrl(c.id, props.thumbV)}
            alt=""
            className="w-24 aspect-video rounded bg-zinc-800 object-cover"
            loading="lazy"
            onError={() => setThumbErr(true)}
          />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap text-[10px]">
            <span className="bg-zinc-700 rounded px-1.5 py-0.5 font-mono">#{c.rank}</span>
            <span
              className="rounded px-1.5 py-0.5 font-semibold uppercase"
              style={{ backgroundColor: `${TYPE_COLORS[c.type]}33`, color: TYPE_COLORS[c.type] }}
            >
              {TYPE_LABEL[c.type] ?? c.type}
            </span>
            <span className={`rounded px-1.5 py-0.5 ${XV_STYLE[c.cross_validation]}`}>{c.cross_validation}</span>
            {Array.isArray(c.signals.angles) &&
              ((c.signals.angles as unknown[]).length >= 2 ? (
                <span
                  className="rounded px-1.5 py-0.5 bg-emerald-800 text-emerald-200"
                  title={`seen by angles ${(c.signals.angles as number[]).join(", ")}`}
                >
                  cross-confirmed
                </span>
              ) : (
                <span
                  className="rounded px-1.5 py-0.5 bg-zinc-700 text-zinc-300"
                  title={`seen by angle ${(c.signals.angles as number[]).join(", ")}`}
                >
                  single-angle
                </span>
              ))}
            {c.signals.disputed === true && (
              <span
                className="rounded px-1.5 py-0.5 bg-orange-900/60 text-orange-200"
                title={`disputed type: ${(c.signals.types as string[] | undefined)?.join(" / ") ?? ""}`}
              >
                disputed
              </span>
            )}
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
          type="text"
          value={inStr}
          onChange={(e) => setInStr(e.target.value)}
          onBlur={(e) => void commit("clip_start", e.target.value, c.clip_start)}
          onKeyDown={onKey}
        />
        <span>OUT</span>
        <input
          className={num}
          type="text"
          value={outStr}
          onChange={(e) => setOutStr(e.target.value)}
          onBlur={(e) => void commit("clip_end", e.target.value, c.clip_end)}
          onKeyDown={onKey}
        />
        <span className="text-zinc-500">({Math.round(c.clip_end - c.clip_start)} s)</span>
        <button
          className="p-1 rounded hover:bg-zinc-700"
          title="Reset window"
          onClick={() => props.onReset(c.id)}
        >
          <RotateCcw size={12} />
        </button>
        <span className="flex-1" />
        {c.type === "goal" && Array.isArray(c.signals.angles) && (
          <select
            className="bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-[10px]"
            title="Team (multi-angle score)"
            value={(c.signals.team as Team | undefined) ?? ""}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "home" || v === "away") void props.onPatch(c.id, { team: v });
            }}
          >
            <option value="">—</option>
            <option value="home">Home</option>
            <option value="away">Away</option>
          </select>
        )}
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
