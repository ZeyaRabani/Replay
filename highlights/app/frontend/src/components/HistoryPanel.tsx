import { History, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { historyApi } from "../api";
import type { HistoryEvent } from "../types";

const KIND_LABEL: Record<string, string> = {
  created: "Created",
  source_added: "Source added",
  queued: "Run queued",
  stage_start: "Stage started",
  stage_done: "Stage finished",
  stage_failed: "Stage failed",
  zones_saved: "Zones saved",
  window_set: "Match window set",
  recut_queued: "Re-cut queued",
  cut_activated: "Cut activated",
  cut_deleted: "Cut deleted",
  candidate_confirmed: "Candidate confirmed",
  candidate_rejected: "Candidate rejected",
  trimmed: "Video trimmed",
  rendered_reel: "Reel rendered",
  sources_purged: "Sources purged",
  paused: "Paused",
  resumed: "Resumed",
  renamed: "Renamed",
  deleted: "Deleted",
  restarted: "Restarted",
};

function describe(e: HistoryEvent): string {
  const base = KIND_LABEL[e.kind] ?? e.kind;
  const bits: string[] = [];
  if (e.stage) bits.push(e.stage);
  if (e.detail?.title) bits.push(`“${e.detail.title}”`);
  if (e.detail?.style) bits.push(`style ${e.detail.style}`);
  if (e.detail?.freed_bytes) bits.push(`${(Number(e.detail.freed_bytes) / 1e9).toFixed(1)} GB freed`);
  if (e.detail?.type) bits.push(String(e.detail.type));
  return bits.length ? `${base} — ${bits.join(" · ")}` : base;
}

const fmt = (ts: number) =>
  new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

export default function HistoryPanel({ projectId }: { projectId: string }) {
  const [events, setEvents] = useState<HistoryEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setEvents(null);
    historyApi.events(projectId).then(setEvents).catch((e) => setError(String(e)));
  }, [projectId]);

  if (error) return <div className="p-4 text-sm text-red-300">{error}</div>;
  if (!events)
    return (
      <div className="p-4 text-sm text-zinc-500 flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" /> Loading…
      </div>
    );
  if (!events.length)
    return <div className="p-4 text-sm text-zinc-500">No history recorded for this project yet.</div>;
  return (
    <div className="p-3 mob:p-2 overflow-y-auto flex-1">
      <div className="flex items-center gap-2 mb-3 text-zinc-300">
        <History size={14} /> <span className="text-sm font-medium">History</span>
      </div>
      <ol className="relative border-l border-zinc-800 ml-2 space-y-2">
        {events.map((e) => (
          <li key={e.id} className="ml-3">
            <span className="absolute -left-[5px] mt-1.5 h-2.5 w-2.5 rounded-full bg-zinc-700 border border-zinc-600" />
            <div className="text-xs text-zinc-200">{describe(e)}</div>
            <div className="text-[10px] text-zinc-600">{fmt(e.ts)}</div>
          </li>
        ))}
      </ol>
    </div>
  );
}
