import { AlertTriangle, CheckCircle2, Clock, Loader2 } from "lucide-react";
import type { PipelineState } from "../types";

const STYLE: Record<PipelineState, string> = {
  none: "bg-zinc-800 text-zinc-300",
  queued: "bg-zinc-700 text-zinc-200",
  running: "bg-amber-900/60 text-amber-200",
  done: "bg-emerald-900/60 text-emerald-200",
  failed: "bg-red-900/60 text-red-200",
};

export default function StatusPill({ state, progress }: { state: PipelineState; progress?: number }) {
  const icon =
    state === "running" ? (
      <Loader2 size={12} className="animate-spin" />
    ) : state === "done" ? (
      <CheckCircle2 size={12} />
    ) : state === "failed" ? (
      <AlertTriangle size={12} />
    ) : (
      <Clock size={12} />
    );
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ${STYLE[state]}`}>
      {icon}
      {state}
      {state === "running" && progress !== undefined && <span>{Math.round(progress * 100)}%</span>}
    </span>
  );
}
