import { Film, Link2, Loader2, Plus, Trash2, Upload, Youtube } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { mediaUrl, projectsApi } from "../api";
import StatusPill from "../components/StatusPill";
import TopBar from "../components/TopBar";
import type { ProjectSummary } from "../types";

const fmtDate = (v: number | string) =>
  new Date(typeof v === "number" ? v * 1000 : v).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
const fmtDur = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const isLive = (p: ProjectSummary) => p.pipeline_state === "queued" || p.pipeline_state === "running";

function SourceBadge({ p }: { p: ProjectSummary }) {
  const k = p.source.kind;
  const cls = "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide";
  if (k === "youtube")
    return (
      <span className={`${cls} bg-red-900/50 text-red-200`}>
        <Youtube size={11} /> YouTube
      </span>
    );
  if (k === "upload")
    return (
      <span className={`${cls} bg-sky-900/50 text-sky-200`}>
        <Upload size={11} /> Upload
      </span>
    );
  return (
    <span className={`${cls} bg-zinc-700 text-zinc-200`}>
      <Film size={11} /> Local
    </span>
  );
}

function ProjectCard({ p, onDelete }: { p: ProjectSummary; onDelete: (p: ProjectSummary) => void }) {
  const [thumbErr, setThumbErr] = useState(false);
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 hover:border-zinc-700 transition-colors flex gap-3 p-3">
      <Link to={`/projects/${p.id}`} className="shrink-0">
        {p.thumb_url && !thumbErr ? (
          <img
            src={mediaUrl(p.thumb_url)}
            alt=""
            onError={() => setThumbErr(true)}
            className="w-40 aspect-video rounded bg-zinc-800 object-cover"
          />
        ) : (
          <div className="w-40 aspect-video rounded bg-zinc-800 flex items-center justify-center text-zinc-600">
            {isLive(p) ? <Loader2 size={20} className="animate-spin text-amber-400" /> : <Film size={20} />}
          </div>
        )}
      </Link>
      <div className="flex-1 min-w-0 flex flex-col gap-1.5">
        <div className="flex items-start gap-2">
          <Link to={`/projects/${p.id}`} className="font-medium text-sm truncate hover:text-amber-300 flex-1">
            {p.title}
          </Link>
          <button
            className="text-zinc-500 hover:text-red-300 p-1"
            title="Delete project"
            onClick={() => onDelete(p)}
          >
            <Trash2 size={14} />
          </button>
        </div>
        <div className="flex items-center gap-2 flex-wrap text-xs text-zinc-400">
          <SourceBadge p={p} />
          <StatusPill state={p.pipeline_state} progress={p.progress} />
          <span>{fmtDate(p.created_at)}</span>
          {p.video && (
            <span>
              {fmtDur(p.video.duration_s)} · {p.video.width}×{p.video.height}
            </span>
          )}
        </div>
        {isLive(p) ? (
          <div className="mt-1">
            <div className="h-1.5 bg-zinc-800 rounded">
              <div
                className="h-1.5 rounded bg-amber-400 transition-all"
                style={{ width: `${Math.round((p.progress ?? 0) * 100)}%` }}
              />
            </div>
            <div className="text-[11px] text-zinc-400 mt-1 truncate">
              {p.stage && <span className="text-amber-300 font-medium">{p.stage}</span>}
              {p.stage && p.message ? " — " : ""}
              {p.message}
            </div>
          </div>
        ) : p.pipeline_state === "failed" ? (
          <div className="text-[11px] text-red-300 truncate" title={p.message ?? ""}>
            {p.message}
          </div>
        ) : (
          <div className="text-xs text-zinc-400">
            <b className="text-zinc-200">{p.n_candidates}</b> candidates ·{" "}
            <b className="text-emerald-300">{p.n_confirmed}</b> confirmed
          </div>
        )}
        {p.source.url && (
          <div className="text-[11px] text-zinc-500 truncate flex items-center gap-1">
            <Link2 size={10} /> {p.source.url}
          </div>
        )}
        {p.source.filename && <div className="text-[11px] text-zinc-500 truncate">{p.source.filename}</div>}
      </div>
    </div>
  );
}

function NewProject({ onCreated, onError }: { onCreated: (p: ProjectSummary) => void; onError: (m: string) => void }) {
  const [tab, setTab] = useState<"youtube" | "upload">("youtube");
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [cookies, setCookies] = useState("");
  const [showCookies, setShowCookies] = useState(false);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    try {
      const p =
        tab === "youtube"
          ? await projectsApi.createYoutube(url.trim(), title.trim(), cookies.trim() || undefined)
          : await projectsApi.createUpload(file as File, title.trim());
      onCreated(p);
      setUrl("");
      setTitle("");
      setFile(null);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const tabCls = (on: boolean) =>
    `flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-t border-b-2 ${
      on ? "border-amber-400 text-amber-300" : "border-transparent text-zinc-400 hover:text-zinc-200"
    }`;
  const input =
    "bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm placeholder:text-zinc-500 focus:outline-none focus:border-amber-400 w-full";
  const canSubmit = tab === "youtube" ? url.trim().length > 0 : !!file;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <div className="flex items-center gap-2 font-semibold text-sm mb-3">
        <Plus size={16} className="text-amber-400" /> New project
      </div>
      <div className="flex gap-1 border-b border-zinc-800 mb-3">
        <button className={tabCls(tab === "youtube")} onClick={() => setTab("youtube")}>
          <Youtube size={13} /> YouTube link
        </button>
        <button className={tabCls(tab === "upload")} onClick={() => setTab("upload")}>
          <Upload size={13} /> Upload file
        </button>
      </div>
      <div className="flex flex-col gap-2">
        {tab === "youtube" ? (
          <input
            className={input}
            placeholder="https://www.youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        ) : (
          <label className="flex items-center gap-2 bg-zinc-800 border border-dashed border-zinc-600 hover:border-amber-400 rounded px-3 py-3 text-sm cursor-pointer">
            <Upload size={15} className="text-zinc-400" />
            <span className="truncate text-zinc-300">{file ? file.name : "Choose a video file (mp4/mkv)…"}</span>
            <input
              type="file"
              accept="video/*,.mkv"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
        )}
        <input
          className={input}
          placeholder="Title (optional)"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        {tab === "youtube" && (
          <div className="text-xs">
            <button
              type="button"
              onClick={() => setShowCookies(!showCookies)}
              className="text-zinc-400 hover:text-zinc-200"
            >
              {showCookies ? "▾" : "▸"} Advanced: YouTube cookies (Netscape cookies.txt)
            </button>
            {showCookies && (
              <div className="mt-1">
                <textarea
                  className={`${input} font-mono text-[11px] h-24`}
                  placeholder="# Netscape HTTP Cookie File — paste cookies.txt contents here"
                  value={cookies}
                  onChange={(e) => setCookies(e.target.value)}
                />
                <p className="text-[11px] text-zinc-500 mt-1">
                  Needed only if YouTube blocks the server (bot check). Export with a
                  &apos;Get cookies.txt&apos; browser extension.
                </p>
              </div>
            )}
          </div>
        )}
        <button
          disabled={busy || !canSubmit}
          onClick={() => void submit()}
          className="flex items-center justify-center gap-2 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-2 text-sm disabled:opacity-40"
        >
          {busy ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />}
          {tab === "youtube" ? "Download & analyse" : "Upload & analyse"}
        </button>
        <p className="text-[11px] text-zinc-500">
          Best available quality is downloaded and analysed on this machine. Processing runs in the background —
          you can close the tab and come back.
        </p>
      </div>
    </div>
  );
}

export default function Projects() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const showError = (m: string) => {
    setError(m);
    window.setTimeout(() => setError(null), 8000);
  };

  const refresh = useCallback(async () => {
    try {
      setProjects(await projectsApi.list());
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const anyLive = projects?.some(isLive) ?? false;
  useEffect(() => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
    if (!anyLive) return;
    timer.current = window.setInterval(() => void refresh(), 2000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [anyLive, refresh]);

  const del = async (p: ProjectSummary) => {
    if (!window.confirm(`Delete "${p.title}"? This removes the downloaded video and all renders.`)) return;
    try {
      await projectsApi.remove(p.id);
      setProjects((ps) => ps?.filter((x) => x.id !== p.id) ?? null);
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <TopBar />
      {error && <div className="bg-red-900/80 text-red-100 text-sm px-4 py-2 border-b border-red-700">{error}</div>}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-4 p-4 max-w-7xl w-full mx-auto">
        <div className="flex flex-col gap-3 min-w-0">
          <div className="flex items-center justify-between">
            <h1 className="font-semibold">Your projects</h1>
            <span className="text-xs text-zinc-500">{projects ? `${projects.length} total` : ""}</span>
          </div>
          {projects === null ? (
            <div className="text-sm text-zinc-500 flex items-center gap-2">
              <Loader2 size={14} className="animate-spin" /> Loading…
            </div>
          ) : projects.length === 0 ? (
            <div className="rounded-lg border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-500">
              No projects yet. Paste a YouTube link or upload a match video to get started.
            </div>
          ) : (
            projects.map((p) => <ProjectCard key={p.id} p={p} onDelete={del} />)
          )}
        </div>
        <div>
          <NewProject onCreated={(p) => setProjects((ps) => [p, ...(ps ?? [])])} onError={showError} />
        </div>
      </div>
    </div>
  );
}
