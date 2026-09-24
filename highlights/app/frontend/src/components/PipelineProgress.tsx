import { AlertTriangle, Check, Circle, Info, Loader2, RotateCcw, XCircle } from "lucide-react";
import type { PipelineStatus, ProjectSummary } from "../types";

export const STAGES: { key: string; label: string }[] = [
  { key: "download", label: "Download best quality" },
  { key: "probe", label: "Probe video" },
  { key: "audio", label: "Audio features & whistles" },
  { key: "motion", label: "Motion features" },
  { key: "features", label: "Build feature table" },
  { key: "score", label: "Score moments" },
  { key: "candidates", label: "Pick candidates" },
  { key: "stats", label: "Match stats" },
];

interface Props {
  project: ProjectSummary;
  status: PipelineStatus | null;
  onRerun: () => void;
  onCancel: () => void;
  busy: boolean;
}

export default function PipelineProgress({ project, status, onRerun, onCancel, busy }: Props) {
  const stages = project.source.kind === "youtube" ? STAGES : STAGES.filter((s) => s.key !== "download");
  const curIdx = status ? stages.findIndex((s) => s.key === status.stage) : -1;
  const state = status?.state ?? project.pipeline_state;
  const failed = state === "failed";
  const running = state === "running" || state === "queued";

  return (
    <div className="flex-1 min-h-0 overflow-auto p-4">
      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-[320px_1fr] gap-4">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 mb-3">
            {failed ? (
              <AlertTriangle size={18} className="text-red-400" />
            ) : (
              <Loader2 size={18} className="animate-spin text-amber-400" />
            )}
            <div className="font-semibold text-sm">{failed ? "Processing failed" : "Processing match"}</div>
          </div>
          <div className="h-2 bg-zinc-800 rounded mb-1">
            <div
              className={`h-2 rounded transition-all ${failed ? "bg-red-500" : "bg-amber-400"}`}
              style={{ width: `${Math.round((status?.progress ?? project.progress ?? 0) * 100)}%` }}
            />
          </div>
          <div className="text-xs text-zinc-400 mb-4">
            {Math.round((status?.progress ?? project.progress ?? 0) * 100)}% ·{" "}
            {failed
              ? (status?.error ?? project.message ?? "failed")
              : (status?.message ?? project.message)}
          </div>
          <ol className="flex flex-col gap-1.5">
            {stages.map((s, i) => {
              const done = status?.state === "done" || (curIdx >= 0 && i < curIdx);
              const cur = i === curIdx && running;
              const isFailedHere = i === curIdx && failed;
              return (
                <li
                  key={s.key}
                  className={`flex items-center gap-2 text-xs ${
                    done ? "text-emerald-300" : cur ? "text-amber-200" : isFailedHere ? "text-red-300" : "text-zinc-500"
                  }`}
                >
                  {done ? (
                    <Check size={13} />
                  ) : cur ? (
                    <Loader2 size={13} className="animate-spin" />
                  ) : isFailedHere ? (
                    <XCircle size={13} />
                  ) : (
                    <Circle size={13} />
                  )}
                  <span className="flex-1">{s.label}</span>
                  {cur && status && <span className="text-zinc-500">{Math.round(status.stage_progress * 100)}%</span>}
                </li>
              );
            })}
          </ol>
          {status?.download && (
            <div className="mt-3 text-[11px] text-zinc-500">
              Source: {status.download.resolution} ({status.download.format})
            </div>
          )}
          {failed && status?.error && (
            <div className="mt-3 rounded bg-red-950/60 border border-red-900 text-red-200 text-xs p-2 whitespace-pre-wrap">
              {status.error}
            </div>
          )}
          <div className="mt-4 flex gap-2">
            {running ? (
              <button
                disabled={busy}
                onClick={onCancel}
                className="flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2.5 py-1 text-xs disabled:opacity-40"
              >
                <XCircle size={13} /> Cancel
              </button>
            ) : (
              <button
                disabled={busy}
                onClick={onRerun}
                className="flex items-center gap-1 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-2.5 py-1 text-xs disabled:opacity-40"
              >
                <RotateCcw size={13} /> Re-run
              </button>
            )}
          </div>
          <div className="mt-4 flex items-start gap-1.5 text-[11px] text-zinc-500">
            <Info size={12} className="mt-0.5 shrink-0" />
            You can close this tab; processing continues in the background and this page updates when you return.
          </div>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3 flex flex-col min-h-[320px]">
          <div className="text-[11px] uppercase tracking-wide text-zinc-500 mb-2">Log</div>
          <pre className="flex-1 overflow-auto text-[11px] leading-relaxed font-mono text-zinc-300 whitespace-pre-wrap">
            {status?.log && status.log.length > 0 ? status.log.join("\n") : "waiting for log output…"}
          </pre>
        </div>
      </div>
    </div>
  );
}
