import { BarChart3, ChevronLeft, Clapperboard, ListVideo, Loader2, PieChart, XCircle } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ProjectApiContext, projectApi, projectsApi } from "../api";
import DirectorCut from "../components/DirectorCut";
import MatchAnalysis from "../components/MatchAnalysis";
import PipelineProgress from "../components/PipelineProgress";
import StatusPill from "../components/StatusPill";
import TitleEdit from "../components/TitleEdit";
import TopBar from "../components/TopBar";
import type { PipelineStatus, ProjectDetail } from "../types";
import ProjectReview from "./ProjectReview";
import ProjectStats from "./ProjectStats";

type Tab = "review" | "stats" | "director" | "analysis";

export default function ProjectPage() {
  const { id = "" } = useParams();
  const api = useMemo(() => projectApi(id), [id]);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("review");
  const [seekRequest, setSeekRequest] = useState<{ t: number; n: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [gen, setGen] = useState(0);
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

  // multi-angle projects waiting for sync offsets default to the director tab
  useEffect(() => {
    if (project?.pipeline_state === "needs_input") setTab("director");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project?.id]);

  const live = project?.pipeline_state === "queued" || project?.pipeline_state === "running";
  // a re-run on a project that already has output keeps the old cut viewable;
  // remount the tabs when it finishes so they pick up the new files
  const wasLive = useRef(false);
  useEffect(() => {
    if (wasLive.current && !live) setGen((g) => g + 1);
    wasLive.current = live;
  }, [live]);
  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    // slow poll when idle so a re-cut started elsewhere (another tab) shows up
    timer.current = window.setInterval(() => void refresh(), live ? 2000 : 15000);
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

  const inlineProgress = !!project && live && project.video !== null;
  const showPipeline =
    project &&
    project.pipeline_state !== "done" &&
    project.pipeline_state !== "none" &&
    project.pipeline_state !== "needs_input" &&
    !inlineProgress;
  const pct = Math.round((status?.progress ?? project?.progress ?? 0) * 100);

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
          {project && (
            <TitleEdit
              value={project.title}
              onSave={async (t) => {
                try {
                  await projectsApi.rename(project.id, t);
                  setProject({ ...project, title: t });
                } catch (e) {
                  setError(e instanceof Error ? e.message : String(e));
                }
              }}
            />
          )}
          {project && <StatusPill state={project.pipeline_state} progress={project.progress} />}
          {project && !showPipeline && (
            <div className="flex items-center gap-1 ml-3 bg-zinc-800/60 rounded p-0.5 overflow-x-auto whitespace-nowrap min-w-0">
              <button className={tabCls(tab === "review")} onClick={() => setTab("review")}>
                <ListVideo size={13} /> Review
              </button>
              <button className={tabCls(tab === "stats")} onClick={() => setTab("stats")}>
                <BarChart3 size={13} /> Stats
              </button>
              {project.mode === "multiangle" && (
                <>
                  <button className={tabCls(tab === "director")} onClick={() => setTab("director")}>
                    <Clapperboard size={13} /> Director cut
                  </button>
                  <button className={tabCls(tab === "analysis")} onClick={() => setTab("analysis")}>
                    <PieChart size={13} /> Analysis
                  </button>
                </>
              )}
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
            {inlineProgress && (
              <div className="border-b border-amber-900/60 bg-amber-950/30 px-4 py-2 flex items-center gap-3 text-xs">
                <Loader2 size={14} className="animate-spin text-amber-400 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-amber-200 truncate">
                      Re-cutting in the background — {status?.message ?? project.message ?? "working"}
                    </span>
                    <span className="text-zinc-400 shrink-0">{pct}%</span>
                  </div>
                  <div className="h-1.5 bg-zinc-800 rounded mt-1">
                    <div className="h-1.5 rounded bg-amber-400 transition-all" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="text-[11px] text-zinc-500 mt-0.5">
                    The current cut below stays available; the page switches to the new cut when it finishes. You can close this tab.
                  </div>
                </div>
                <button
                  disabled={busy}
                  onClick={() => void act(() => api.cancelPipeline())}
                  className="flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 disabled:opacity-40 shrink-0"
                >
                  <XCircle size={12} /> Cancel
                </button>
              </div>
            )}
            {/* keep Review mounted so the video/playhead survive tab switches */}
            <div key={gen} className={tab === "review" ? "flex flex-col flex-1 min-h-0" : "hidden"}>
              <ProjectReview seekRequest={seekRequest} />
            </div>
            {tab === "stats" && (
              <ProjectStats
                key={gen}
                onSeek={(t) => {
                  setSeekRequest((s) => ({ t, n: (s?.n ?? 0) + 1 }));
                  setTab("review");
                }}
              />
            )}
            {tab === "analysis" && (
              <MatchAnalysis
                key={gen}
                onSeek={(t) => {
                  setSeekRequest((s) => ({ t, n: (s?.n ?? 0) + 1 }));
                  setTab("review");
                }}
              />
            )}
            {tab === "director" && (
              <DirectorCut
                key={gen}
                onSeek={(t) => {
                  setSeekRequest((s) => ({ t, n: (s?.n ?? 0) + 1 }));
                  setTab("review");
                }}
                onCutsChanged={() => {
                  setGen((g) => g + 1);
                  void refresh();
                }}
              />
            )}
          </>
        )}
      </div>
    </ProjectApiContext.Provider>
  );
}
