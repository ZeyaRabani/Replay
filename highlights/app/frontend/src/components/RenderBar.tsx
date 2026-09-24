import { Download, Loader2, PlayCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Candidate, RenderJob } from "../types";

interface Props {
  candidates: Candidate[];
  onError: (msg: string) => void;
}

const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

export default function RenderBar(props: Props) {
  const [overlay, setOverlay] = useState(true);
  const [job, setJob] = useState<RenderJob | null>(null);
  const timer = useRef<number | null>(null);

  const confirmed = props.candidates.filter((c) => c.status === "confirmed");
  const selected = confirmed.length > 0 ? confirmed : props.candidates.filter((c) => c.status !== "rejected");
  const totalS = selected.reduce((s, c) => s + (c.clip_end - c.clip_start), 0);

  useEffect(() => () => {
    if (timer.current) window.clearInterval(timer.current);
  }, []);

  const start = async () => {
    try {
      const { job_id } = await api.startRender({ overlay, reencode: false });
      if (timer.current) window.clearInterval(timer.current);
      timer.current = window.setInterval(async () => {
        try {
          const j = await api.renderJob(job_id);
          setJob(j);
          if (j.state === "done" || j.state === "error") {
            if (timer.current) window.clearInterval(timer.current);
            timer.current = null;
            if (j.state === "error") props.onError(j.error ?? "render failed");
          }
        } catch {
          /* transient poll failure */
        }
      }, 1000);
    } catch (e) {
      props.onError(e instanceof Error ? e.message : String(e));
    }
  };

  const running = job && (job.state === "queued" || job.state === "running");

  return (
    <div className="sticky bottom-0 z-10 bg-zinc-900/95 border-t border-zinc-700 px-4 py-2.5 backdrop-blur">
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-sm text-zinc-300">
          <b className="text-amber-400">{selected.length}</b> {confirmed.length > 0 ? "confirmed" : "non-rejected"}{" "}
          · <b className="text-amber-400">{fmt(totalS)}</b>
        </span>
        <label className="flex items-center gap-1.5 text-xs text-zinc-300">
          <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
          overlay
        </label>
        <button
          className="flex items-center gap-1.5 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5 text-sm disabled:opacity-40"
          disabled={selected.length === 0 || !!running}
          onClick={start}
        >
          {running ? <Loader2 size={15} className="animate-spin" /> : <PlayCircle size={15} />} Render reel
        </button>
        {job && (
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-40 h-2 bg-zinc-700 rounded">
              <div
                className={`h-2 rounded ${job.state === "error" ? "bg-red-500" : "bg-amber-400"}`}
                style={{ width: `${Math.round(job.progress * 100)}%` }}
              />
            </div>
            <span className="text-xs text-zinc-400">{job.message}</span>
            {job.state === "done" && (
              <span className="flex items-center gap-2 text-xs">
                {job.reel_url && (
                  <a className="flex items-center gap-1 text-amber-400 hover:underline" href={job.reel_url}>
                    <Download size={12} /> reel.mp4
                  </a>
                )}
                {job.stats_url && (
                  <a className="flex items-center gap-1 text-amber-400 hover:underline" href={job.stats_url}>
                    <Download size={12} /> stats.json
                  </a>
                )}
                {job.clips.map((c) => (
                  <a key={c.id} className="text-zinc-300 hover:underline" href={c.url} title={c.id}>
                    {c.id}.mp4
                  </a>
                ))}
              </span>
            )}
          </div>
        )}
        <span className="flex-1" />
        <a className="text-xs text-zinc-400 hover:text-amber-400 hover:underline" href="/api/stats" target="_blank">
          Export stats
        </a>
      </div>
    </div>
  );
}
