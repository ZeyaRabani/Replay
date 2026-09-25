import { Film, Layers, Link2, Loader2, Plus, RotateCcw, Trash2, Upload, X, Youtube } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { configApi, meApi, mediaUrl, projectApi, projectsApi } from "../api";
import type { CookieStatus } from "../api";
import StatusPill from "../components/StatusPill";
import TopBar from "../components/TopBar";
import type { ProjectMeta, ProjectSummary } from "../types";

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
  if (k === "multiangle")
    return (
      <span className={`${cls} bg-purple-900/50 text-purple-200`}>
        <Layers size={11} /> Multi-angle · {p.n_angles} angles
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

const isBotBlock = (msg?: string | null) =>
  !!msg && /bot check|sign in/i.test(msg);

function ProjectCard({
  p,
  onDelete,
  cookiesSaved,
  onRetry,
  onSetupCookies,
  onUploadInstead,
}: {
  p: ProjectSummary;
  onDelete: (p: ProjectSummary) => void;
  cookiesSaved: boolean;
  onRetry: (p: ProjectSummary) => void;
  onSetupCookies: () => void;
  onUploadInstead: () => void;
}) {
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
          {p.meta?.pitch_type && (
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
              {p.meta.pitch_type}-a-side
            </span>
          )}
          {p.meta?.camera && (
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
              {p.meta.camera}
            </span>
          )}
          {p.meta?.cut_style && (
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
              {p.meta.cut_style === "fast" ? "fast cuts" : "normal cuts"}
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
          isBotBlock(p.message) ? (
            <div className="text-[11px]" title={p.message ?? ""}>
              <span className="text-red-300">YouTube blocked this server&apos;s download.</span>
              <span className="flex gap-3 mt-1">
                {cookiesSaved ? (
                  <button
                    className="inline-flex items-center gap-1 text-amber-300 hover:text-amber-200"
                    onClick={() => onRetry(p)}
                  >
                    <RotateCcw size={11} /> Retry
                  </button>
                ) : (
                  <button
                    className="text-amber-300 hover:text-amber-200"
                    onClick={onSetupCookies}
                  >
                    Set up YouTube cookies
                  </button>
                )}
                <button
                  className="inline-flex items-center gap-1 text-sky-300 hover:text-sky-200"
                  onClick={onUploadInstead}
                >
                  <Upload size={11} /> Upload the file instead
                </button>
              </span>
            </div>
          ) : (
            <div className="text-[11px] text-red-300 truncate" title={p.message ?? ""}>
              {p.message}
            </div>
          )
        ) : p.pipeline_state === "needs_input" ? (
          <Link to={`/projects/${p.id}`} className="text-[11px] text-orange-300 hover:text-orange-200 truncate" title={p.message ?? ""}>
            Needs sync offsets — {p.message}
          </Link>
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

function YouTubeAccess({ status, onChanged, onError, innerRef }: {
  status: CookieStatus | null;
  onChanged: () => void;
  onError: (m: string) => void;
  innerRef: React.Ref<HTMLDivElement>;
}) {
  const saved = status?.saved ?? false;
  const isAdmin = status?.is_admin ?? false;
  const sharedAvailable = status?.shared_available ?? false;
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [share, setShare] = useState(false);
  const [busy, setBusy] = useState(false);
  const input =
    "bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm placeholder:text-zinc-500 focus:outline-none focus:border-amber-400 w-full";

  const save = async () => {
    setBusy(true);
    try {
      await meApi.saveCookies(text, share);
      setText("");
      setOpen(false);
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    try {
      await meApi.deleteCookies();
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div ref={innerRef} className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 mb-3">
      <div className="flex items-center justify-between gap-2">
        <div className="font-semibold text-sm flex items-center gap-2">
          <Youtube size={15} className="text-red-400" /> YouTube access
        </div>
        {saved ? (
          <span className="text-xs text-zinc-400">
            Cookies saved ✓{" "}
            <button className="text-zinc-500 hover:text-red-300 underline" onClick={() => void remove()}>
              Remove
            </button>
          </span>
        ) : sharedAvailable && !isAdmin ? (
          <span className="text-xs text-zinc-400">
            YouTube access provided by the server admin ✓{" "}
            <button
              className="text-zinc-500 hover:text-amber-300 underline"
              onClick={() => setOpen(!open)}
            >
              {open ? "Close" : "paste your own anyway"}
            </button>
          </span>
        ) : (
          <button
            className="text-xs text-amber-300 hover:text-amber-200 underline"
            onClick={() => setOpen(!open)}
          >
            {open ? "Close" : "Not set — Set up"}
          </button>
        )}
      </div>
      {open && !saved && (
        <div className="mt-3 text-xs text-zinc-300 flex flex-col gap-2">
          <ol className="list-decimal pl-4 text-zinc-400 space-y-1">
            <li>
              Install the Chrome extension{" "}
              <a
                href="https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc"
                target="_blank"
                rel="noreferrer"
                className="text-sky-300 underline"
              >
                Get cookies.txt LOCALLY
              </a>
            </li>
            <li>Open youtube.com while signed in, click the extension, then Copy.</li>
            <li>Paste below and Save.</li>
          </ol>
          <textarea
            className={`${input} font-mono text-[11px] h-24`}
            placeholder="# Netscape HTTP Cookie File — paste cookies.txt contents here"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          {isAdmin && (
            <label className="flex items-center gap-1.5 text-xs text-zinc-400">
              <input
                type="checkbox"
                checked={share}
                onChange={(e) => setShare(e.target.checked)}
              />
              Share with all users of this server
            </label>
          )}
          <button
            disabled={busy || !text.trim()}
            onClick={() => void save()}
            className="self-start bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5 disabled:opacity-40"
          >
            {busy ? <Loader2 size={13} className="animate-spin" /> : "Save"}
          </button>
        </div>
      )}
    </div>
  );
}

function NewProject({ onCreated, onError, tab, setTab }: {
  onCreated: (p: ProjectSummary) => void;
  onError: (m: string) => void;
  tab: "youtube" | "upload";
  setTab: (t: "youtube" | "upload") => void;
}) {
  const [url, setUrl] = useState("");
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [uploadOrigin, setUploadOrigin] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"single" | "multi">("single");
  const [maTab, setMaTab] = useState<"links" | "upload">("links");
  const [maRows, setMaRows] = useState<{ url: string; label: string }[]>([
    { url: "", label: "" },
    { url: "", label: "" },
  ]);
  const [maFiles, setMaFiles] = useState<{ file: File | null; label: string }[]>([
    { file: null, label: "" },
    { file: null, label: "" },
  ]);
  const [pitchType, setPitchType] = useState("");
  const [camera, setCamera] = useState("");
  const [cutStyle, setCutStyle] = useState<"normal" | "fast">("normal");

  useEffect(() => {
    configApi
      .get()
      .then((c) => setUploadOrigin(c.upload_origin))
      .catch(() => setUploadOrigin(null));
  }, []);
  const foreignUpload =
    !!uploadOrigin && uploadOrigin.replace(/\/$/, "") !== window.location.origin.replace(/\/$/, "");

  const submit = async () => {
    setBusy(true);
    try {
      const meta: ProjectMeta = {};
      if (pitchType) meta.pitch_type = pitchType as ProjectMeta["pitch_type"];
      if (camera) meta.camera = camera as ProjectMeta["camera"];
      if (mode === "multi") meta.cut_style = cutStyle;
      let p: ProjectSummary;
      if (mode === "multi") {
        if (maTab === "links") {
          const angles = maRows.map((r, i) => ({ url: r.url.trim(), label: r.label.trim() || `Angle ${i + 1}` }));
          p = await projectsApi.createMultiangle(title.trim() || undefined, angles, undefined, meta);
        } else {
          p = await projectsApi.createMultiangleUpload(
            maFiles.map((r) => r.file as File),
            maFiles.map((r, i) => r.label.trim() || `Angle ${i + 1}`),
            title.trim() || undefined,
            meta,
          );
        }
      } else {
        p =
          tab === "youtube"
            ? await projectsApi.createYoutube(url.trim(), title.trim(), meta)
            : await projectsApi.createUpload(file as File, title.trim(), meta);
      }
      onCreated(p);
      setUrl("");
      setTitle("");
      setFile(null);
      setPitchType("");
      setCamera("");
      setCutStyle("normal");
      setMaRows([{ url: "", label: "" }, { url: "", label: "" }]);
      setMaFiles([{ file: null, label: "" }, { file: null, label: "" }]);
    } catch (e) {
      let m = e instanceof Error ? e.message : String(e);
      if (m.startsWith("413")) {
        m = `Upload failed: file too large for this link.${
          foreignUpload ? ` Open ${uploadOrigin}/projects and upload there.` : ""
        }`;
      }
      onError(m);
    } finally {
      setBusy(false);
    }
  };

  const tabCls = (on: boolean) =>
    `flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-t border-b-2 ${
      on ? "border-amber-400 text-amber-300" : "border-transparent text-zinc-400 hover:text-zinc-200"
    }`;
  const modeCls = (on: boolean) =>
    `flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded ${
      on ? "bg-zinc-700 text-amber-300" : "bg-zinc-800/60 text-zinc-400 hover:text-zinc-200"
    }`;
  const input =
    "bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm placeholder:text-zinc-500 focus:outline-none focus:border-amber-400 w-full";
  const canSubmit =
    mode === "multi"
      ? maTab === "links"
        ? maRows.every((r) => r.url.trim().length > 0)
        : maFiles.every((r) => r.file !== null)
      : tab === "youtube"
        ? url.trim().length > 0
        : !!file;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <div className="flex items-center gap-2 font-semibold text-sm mb-3">
        <Plus size={16} className="text-amber-400" /> New project
      </div>
      <div className="flex gap-1.5 mb-3">
        <button className={modeCls(mode === "single")} onClick={() => setMode("single")}>
          <Film size={13} /> Single camera
        </button>
        <button className={modeCls(mode === "multi")} onClick={() => setMode("multi")}>
          <Layers size={13} /> Multi-angle (director cut)
        </button>
      </div>
      {mode === "multi" ? (
        <div className="flex flex-col gap-2">
          <div className="flex gap-1 border-b border-zinc-800 mb-1">
            <button className={tabCls(maTab === "links")} onClick={() => setMaTab("links")}>
              <Youtube size={13} /> YouTube links
            </button>
            <button className={tabCls(maTab === "upload")} onClick={() => setMaTab("upload")}>
              <Upload size={13} /> Upload files
            </button>
          </div>
          {maTab === "links" ? (
            <>
              {maRows.map((r, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center gap-1.5">
                    <input
                      className={input}
                      placeholder={`Angle ${i + 1} YouTube URL${i === 0 ? " — reference angle (timeline zero)" : ""}`}
                      value={r.url}
                      onChange={(e) =>
                        setMaRows((rs) => rs.map((x, j) => (j === i ? { ...x, url: e.target.value } : x)))
                      }
                    />
                    {maRows.length > 2 && (
                      <button
                        className="text-zinc-500 hover:text-red-300 p-1 shrink-0"
                        title="Remove angle"
                        onClick={() => setMaRows((rs) => rs.filter((_, j) => j !== i))}
                      >
                        <X size={13} />
                      </button>
                    )}
                  </div>
                  <input
                    className={input}
                    placeholder={`Label (optional, e.g. "Main", "Far side")`}
                    value={r.label}
                    onChange={(e) =>
                      setMaRows((rs) => rs.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))
                    }
                  />
                </div>
              ))}
              {maRows.length < 4 && (
                <button
                  className="self-start text-xs text-amber-300 hover:text-amber-200"
                  onClick={() => setMaRows((rs) => [...rs, { url: "", label: "" }])}
                >
                  + Add angle
                </button>
              )}
            </>
          ) : foreignUpload ? (
            <div className="text-xs text-zinc-300 rounded border border-zinc-700 bg-zinc-800/60 p-3">
              Large uploads must go directly to your server:
              <a
                href={`${uploadOrigin}/projects`}
                className="mt-2 inline-block bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5"
              >
                Open {uploadOrigin}/projects
              </a>
            </div>
          ) : (
            <>
              {maFiles.map((r, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center gap-1.5">
                    <label className="flex items-center gap-2 bg-zinc-800 border border-dashed border-zinc-600 hover:border-amber-400 rounded px-3 py-2 text-sm cursor-pointer flex-1 min-w-0">
                      <Upload size={14} className="text-zinc-400 shrink-0" />
                      <span className="truncate text-zinc-300">
                        {r.file ? r.file.name : `Angle ${i + 1} video${i === 0 ? " (reference)" : ""}…`}
                      </span>
                      <input
                        type="file"
                        accept="video/*,.mkv"
                        className="hidden"
                        onChange={(e) =>
                          setMaFiles((rs) =>
                            rs.map((x, j) => (j === i ? { ...x, file: e.target.files?.[0] ?? null } : x)),
                          )
                        }
                      />
                    </label>
                    {maFiles.length > 2 && (
                      <button
                        className="text-zinc-500 hover:text-red-300 p-1 shrink-0"
                        title="Remove angle"
                        onClick={() => setMaFiles((rs) => rs.filter((_, j) => j !== i))}
                      >
                        <X size={13} />
                      </button>
                    )}
                  </div>
                  <input
                    className={input}
                    placeholder="Label (optional)"
                    value={r.label}
                    onChange={(e) =>
                      setMaFiles((rs) => rs.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))
                    }
                  />
                </div>
              ))}
              {maFiles.length < 4 && (
                <button
                  className="self-start text-xs text-amber-300 hover:text-amber-200"
                  onClick={() => setMaFiles((rs) => [...rs, { file: null, label: "" }])}
                >
                  + Add angle
                </button>
              )}
            </>
          )}
          <input
            className={input}
            placeholder="Title (optional)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <div className="flex gap-2">
            <select className={input} value={pitchType} onChange={(e) => setPitchType(e.target.value)}>
              <option value="">Pitch — not set</option>
              <option value="11">11-a-side</option>
              <option value="9">9-a-side</option>
              <option value="7">7-a-side</option>
              <option value="5">5-a-side</option>
              <option value="other">Other</option>
            </select>
            <select className={input} value={camera} onChange={(e) => setCamera(e.target.value)}>
              <option value="">Camera — not set</option>
              <option value="normal">Normal 1×</option>
              <option value="ultrawide">Ultrawide 0.5×</option>
              <option value="zoom">Zoomed</option>
              <option value="other">Other</option>
            </select>
            <select
              className={input}
              value={cutStyle}
              onChange={(e) => setCutStyle(e.target.value as "normal" | "fast")}
            >
              <option value="normal">Cuts: Normal — broadcast-style holds</option>
              <option value="fast">Cuts: Fast — follow the ball, quick cuts</option>
            </select>
          </div>
          <button
            disabled={busy || !canSubmit}
            onClick={() => void submit()}
            className="flex items-center justify-center gap-2 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-2 text-sm disabled:opacity-40"
          >
            {busy ? <Loader2 size={15} className="animate-spin" /> : <Layers size={15} />}
            Assemble director cut
          </button>
          <p className="text-[11px] text-zinc-500">
            2–4 angles of the same match. The first angle is the reference (timeline zero). Angles are
            downloaded, synced by audio, and a director picks the best shot second-by-second.
          </p>
        </div>
      ) : (
        <>
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
        ) : foreignUpload ? (
          <div className="text-xs text-zinc-300 rounded border border-zinc-700 bg-zinc-800/60 p-3">
            Large uploads must go directly to your server:
            <a
              href={`${uploadOrigin}/projects`}
              className="mt-2 inline-block bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-1.5"
            >
              Open {uploadOrigin}/projects
            </a>
          </div>
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
        <div className="flex gap-2">
          <select className={input} value={pitchType} onChange={(e) => setPitchType(e.target.value)}>
            <option value="">Pitch — not set</option>
            <option value="11">11-a-side</option>
            <option value="9">9-a-side</option>
            <option value="7">7-a-side</option>
            <option value="5">5-a-side</option>
            <option value="other">Other</option>
          </select>
          <select className={input} value={camera} onChange={(e) => setCamera(e.target.value)}>
            <option value="">Camera — not set</option>
            <option value="normal">Normal 1×</option>
            <option value="ultrawide">Ultrawide 0.5×</option>
            <option value="zoom">Zoomed</option>
            <option value="other">Other</option>
          </select>
        </div>
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
        </>
      )}
    </div>
  );
}

export default function Projects() {
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cookieStatus, setCookieStatus] = useState<CookieStatus | null>(null);
  const [newTab, setNewTab] = useState<"youtube" | "upload">("youtube");
  const cookiesPanel = useRef<HTMLDivElement | null>(null);
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

  const refreshCookies = useCallback(async () => {
    try {
      setCookieStatus(await meApi.getCookies());
    } catch {
      /* leave as-is */
    }
  }, []);

  useEffect(() => {
    void refreshCookies();
  }, [refreshCookies]);

  const retry = async (p: ProjectSummary) => {
    try {
      await projectApi(p.id).runPipeline({ force: true });
      void refresh();
    } catch (e) {
      showError(e instanceof Error ? e.message : String(e));
    }
  };

  const openCookiesPanel = () => {
    cookiesPanel.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const uploadInstead = () => {
    setNewTab("upload");
    cookiesPanel.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

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
              No projects yet. Paste a YouTube link, upload a match video, or assemble a multi-angle director cut to get started.
            </div>
          ) : (
            projects.map((p) => (
              <ProjectCard
                key={p.id}
                p={p}
                onDelete={del}
                cookiesSaved={!!cookieStatus?.saved || !!cookieStatus?.shared_available}
                onRetry={retry}
                onSetupCookies={openCookiesPanel}
                onUploadInstead={uploadInstead}
              />
            ))
          )}
        </div>
        <div>
          <YouTubeAccess
            status={cookieStatus}
            onChanged={() => void refreshCookies()}
            onError={showError}
            innerRef={cookiesPanel}
          />
          <NewProject
            onCreated={(p) => setProjects((ps) => [p, ...(ps ?? [])])}
            onError={showError}
            tab={newTab}
            setTab={setNewTab}
          />
        </div>
      </div>
    </div>
  );
}
