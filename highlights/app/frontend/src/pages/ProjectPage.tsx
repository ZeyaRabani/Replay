import { BarChart3, ChevronLeft, ListVideo, Loader2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ProjectApiContext, projectApi } from "../api";
import PipelineProgress from "../components/PipelineProgress";
import StatusPill from "../components/StatusPill";
import TopBar from "../components/TopBar";
import type { PipelineStatus, ProjectDetail } from "../types";
import ProjectReview from "./ProjectReview";
import ProjectStats from "./ProjectStats";

type Tab = "review" | "stats";

export default function ProjectPage() {
  const { id = "" } = useParams();
  const api = useMemo(() => projectApi(id), [id]);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("review");
  const [seekRequest, setSeekRequest] = useState<{ t: number; n: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const p = await api.get();
      setProject(p);
      if (p.pipeline_state === "queued" || p.pipeline_state === "running" || p.pipeline_state === "failed") {
        setStatus(await api.pipeline());
      } else {
        setStatus(p.pipeline);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [api]);

  useEffect(() => {
    setProject(null);
    setStatus(null);
    void refresh();
  }, [refresh]);

  const live = project?.pipeline_state === "queued" || project?.pipeline_state === "running";
  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    if (!live) return;
    timer.current = window.setInterval(() => void refresh(), 2000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [live, refresh]);

  const act = async (fn: () => Promise<PipelineStatus>) => {
    setBusy(true);
    try {
      setStatus(await fn());
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const showPipeline = project && project.pipeline_state !== "done" && project.pipeline_state !== "none";

  const tabCls = (on: boolean) =>
    `flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded ${
      on ? "bg-zinc-700 text-amber-300" : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800"
    }`;

  return (
    <ProjectApiContext.Provider value={api}>
      <div className="flex flex-col h-screen">
        <TopBar>
          <Link to="/projects" className="text-zinc-400 hover:text-zinc-200 flex items-center" title="All projects">
            <ChevronLeft size={16} />
          </Link>
          <span className="text-sm truncate">{project?.title ?? "…"}</span>
          {project && <StatusPill state={project.pipeline_state} progress={project.progress} />}
          {project && !showPipeline && (
            <div className="flex items-center gap-1 ml-3 bg-zinc-800/60 rounded p-0.5">
              <button className={tabCls(tab === "review")} onClick={() => setTab("review")}>
                <ListVideo size={13} /> Review
              </button>
              <button className={tabCls(tab === "stats")} onClick={() => setTab("stats")}>
                <BarChart3 size={13} /> Stats
              </button>
            </div>
          )}
        </TopBar>
        {error && <div className="bg-red-900/80 text-red-100 text-sm px-4 py-2 border-b border-red-700">{error}</div>}
        {!project ? (
          <div className="p-6 text-sm text-zinc-500 flex items-center gap-2">
            {!error && <Loader2 size={14} className="animate-spin" />} {error ? "Project unavailable" : "Loading…"}
          </div>
        ) : showPipeline ? (
          <PipelineProgress
            project={project}
            status={status}
            busy={busy}
            onRerun={() => void act(() => api.runPipeline({}))}
            onCancel={() => void act(() => api.cancelPipeline())}
          />
        ) : (
          <>
            {/* keep Review mounted so the video/playhead survive tab switches */}
            <div className={tab === "review" ? "flex flex-col flex-1 min-h-0" : "hidden"}>
              <ProjectReview seekRequest={seekRequest} />
            </div>
            {tab === "stats" && (
              <ProjectStats
                onSeek={(t) => {
                  setSeekRequest((s) => ({ t, n: (s?.n ?? 0) + 1 }));
                  setTab("review");
                }}
              />
            )}
          </>
        )}
      </div>
    </ProjectApiContext.Provider>
  );
}
