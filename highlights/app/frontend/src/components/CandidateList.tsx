import { useState } from "react";
import type { Candidate, Status } from "../types";
import CandidateCard from "./CandidateCard";

interface Props {
  candidates: Candidate[];
  selectedId: string | null;
  thumbV: number | undefined;
  sort: "confidence" | "time";
  onSort: (s: "confidence" | "time") => void;
  onSelect: (c: Candidate) => void;
  onPatch: (id: string, patch: Partial<Candidate>) => Promise<boolean>;
  onReset: (id: string) => void;
}

const FILTERS: (Status | "all")[] = ["all", "pending", "confirmed", "rejected"];

export default function CandidateList(props: Props) {
  const [filter, setFilter] = useState<Status | "all">("all");
  const shown = props.candidates.filter((c) => filter === "all" || c.status === filter);
  const chip = (active: boolean) =>
    `rounded px-2 py-0.5 text-xs ${active ? "bg-amber-500 text-zinc-900 font-semibold" : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"}`;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-1.5 px-1 pb-2 flex-wrap">
        <span className="text-xs text-zinc-400 mr-1">{shown.length} events</span>
        <button className={chip(props.sort === "confidence")} onClick={() => props.onSort("confidence")}>
          confidence
        </button>
        <button className={chip(props.sort === "time")} onClick={() => props.onSort("time")}>
          time
        </button>
        <span className="w-2" />
        {FILTERS.map((f) => (
          <button key={f} className={chip(filter === f)} onClick={() => setFilter(f)}>
            {f}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-0">
        {shown.map((c) => (
          <CandidateCard
            key={c.id}
            c={c}
            selected={c.id === props.selectedId}
            thumbV={props.thumbV}
            onSelect={props.onSelect}
            onPatch={props.onPatch}
            onReset={props.onReset}
          />
        ))}
      </div>
    </div>
  );
}

