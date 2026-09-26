import { createContext, useContext } from "react";
import type {
  Candidate,
  CutsList,
  DirectorFull,
  MultiangleInfo,
  PipelineStatus,
  ProjectDetail,
  ProjectMeta,
  ProjectSummary,
  RenderJob,
  Stats,
  Team,
  User,
  VideoInfo,
  ZonePolygon,
} from "./types";

const USER_KEY = "hl_user";

export const getUser = (): string | null => localStorage.getItem(USER_KEY);
export const setUser = (name: string | null) => {
  if (name) localStorage.setItem(USER_KEY, name);
  else localStorage.removeItem(USER_KEY);
};

/** <img>/<video>/<a download> cannot send X-User, so media URLs carry the user as a query param */
export function mediaUrl(path: string, params: Record<string, string | undefined> = {}): string {
  const q = new URLSearchParams();
  const u = getUser();
  if (u) q.set("user", u);
  for (const [k, v] of Object.entries(params)) if (v !== undefined) q.set(k, v);
  const s = q.toString();
  return s ? `${path}?${s}` : path;
}

function userHeaders(): Record<string, string> {
  const u = getUser();
  return u ? { "X-User": u } : {};
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { ...userHeaders(), ...(init?.headers ?? {}) } });
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const j = await r.json();
      msg = j.detail ?? msg;
    } catch {
      /* keep status */
    }
    if (r.status === 413) msg = "413: file too large";
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  if (r.status === 204) return undefined as T;
  return r.json();
}

const json = (body: unknown, method = "POST"): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const usersApi = {
  list: () => req<User[]>("/api/users"),
  create: (name: string) => req<User>("/api/users", json({ name })),
};

export interface CookieStatus {
  saved: boolean;
  updated_at: number | null;
  shared_available: boolean;
  is_admin: boolean;
}

export const meApi = {
  getCookies: () => req<CookieStatus>("/api/me/youtube-cookies"),
  saveCookies: (cookies_text: string, share = false) =>
    req<{ saved: boolean; updated_at: number }>("/api/me/youtube-cookies", json({ cookies_text, share }, "PUT")),
  deleteCookies: () => req<void>("/api/me/youtube-cookies", { method: "DELETE" }),
  deleteSharedCookies: () => req<void>("/api/admin/youtube-cookies", { method: "DELETE" }),
};

export const configApi = {
  get: () => req<{ upload_origin: string | null }>("/api/config"),
};

export const projectsApi = {
  list: () => req<ProjectSummary[]>("/api/projects"),
  createYoutube: (youtube_url: string, title?: string, meta?: ProjectMeta) =>
    req<ProjectSummary>("/api/projects", json({ youtube_url, title: title || undefined, ...meta })),
  createPath: (path: string, title?: string, meta?: ProjectMeta) =>
    req<ProjectSummary>("/api/projects", json({ path, title: title || undefined, ...meta })),
  createUpload: (file: File, title?: string, meta?: ProjectMeta) => {
    const fd = new FormData();
    fd.append("file", file);
    if (title) fd.append("title", title);
    if (meta?.pitch_type) fd.append("pitch_type", meta.pitch_type);
    if (meta?.camera) fd.append("camera", meta.camera);
    if (meta?.cut_style) fd.append("cut_style", meta.cut_style);
    return req<ProjectSummary>("/api/projects", { method: "POST", body: fd });
  },
  createMultiangle: (title: string | undefined, angles: { url: string; label: string }[], cookies_text?: string, meta?: ProjectMeta, match_window?: [number, number]) =>
    req<ProjectSummary>("/api/projects/multiangle", json({ title: title || undefined, angles, cookies_text, ...meta, match_window })),
  createMultiangleUpload: (files: File[], labels: string[], title?: string, meta?: ProjectMeta) => {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    for (const l of labels) fd.append("labels", l);
    if (title) fd.append("title", title);
    if (meta?.pitch_type) fd.append("pitch_type", meta.pitch_type);
    if (meta?.camera) fd.append("camera", meta.camera);
    if (meta?.cut_style) fd.append("cut_style", meta.cut_style);
    return req<ProjectSummary>("/api/projects/multiangle/upload", { method: "POST", body: fd });
  },
  remove: (id: string) => req<void>(`/api/projects/${id}`, { method: "DELETE" }),
  storage: () =>
    req<{
      total_bytes: number;
      used_bytes: number;
      free_bytes: number;
      projects_bytes: number;
      per_project: { id: string; title: string; owner: string; bytes: number; sources_bytes: number }[];
    }>("/api/storage"),
  purgeSources: (id: string) =>
    req<{ freed_bytes: number; sources_purged: boolean }>(`/api/projects/${id}/purge-sources`, { method: "POST" }),
};

export function projectApi(id: string) {
  const base = `/api/projects/${id}`;
  return {
    id,
    base,
    get: () => req<ProjectDetail>(base),
    pipeline: () => req<PipelineStatus>(`${base}/pipeline`),
    runPipeline: (body: { stages?: string[]; force?: boolean } = {}) =>
      req<PipelineStatus>(`${base}/pipeline/run`, json(body)),
    cancelPipeline: () => req<PipelineStatus>(`${base}/pipeline/cancel`, { method: "POST" }),
    stats: () => req<Stats>(`${base}/stats`),

    registerVideo: (path: string) => req<VideoInfo>(`${base}/video`, json({ path })),
    getVideo: () => req<VideoInfo>(`${base}/video`),
    buildProxy: () => req<{ status: string }>(`${base}/video/proxy`, { method: "POST" }),
    proxyStatus: () => req<{ ready: boolean; progress: number }>(`${base}/video/proxy/status`),
    videoUrl: (kind: "proxy" | "source", v: string) =>
      mediaUrl(`${base}/video/${kind === "proxy" ? "proxy.mp4" : "source.mp4"}`, { v }),

    getMatchWindow: () =>
      req<{ match_window: [number, number]; halves: { start: number; end: number }[] | null; warning?: string; duration: number }>(
        `${base}/match-window`),
    putMatchWindow: (start_s: number, end_s: number) =>
      req<{ match_window: [number, number] }>(`${base}/match-window`, json({ start_s, end_s }, "PUT")),
    startTrim: (start_s: number, end_s: number) =>
      req<{ ready: boolean; progress: number }>(`${base}/video/trim`, json({ start_s, end_s })),
    trimStatus: (start_s: number, end_s: number) =>
      req<{ ready: boolean; progress: number; error?: string }>(
        `${base}/video/trim/status?start=${start_s}&end=${end_s}`),
    trimmedUrl: (start_s: number, end_s: number, v: string) =>
      mediaUrl(`${base}/video/trimmed.mp4`, { start: String(start_s), end: String(end_s), v }),

    loadCandidatesPath: (path: string) =>
      req<Candidate[]>(`${base}/candidates/load`, json({ path })).then(() => undefined),
    loadCandidatesFile: (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return req<Candidate[]>(`${base}/candidates/load`, { method: "POST", body: fd }).then(() => undefined);
    },
    listCandidates: (sort: "confidence" | "time") => req<Candidate[]>(`${base}/candidates?sort=${sort}`),
    patchCandidate: (cid: string, patch: Partial<Candidate> & { team?: Team }) =>
      req<Candidate>(`${base}/candidates/${cid}`, json(patch, "PATCH")),
    resetCandidate: (cid: string) => req<Candidate>(`${base}/candidates/${cid}/reset`, { method: "POST" }),
    thumbUrl: (cid: string, v?: string) => mediaUrl(`${base}/candidates/${cid}/thumb.jpg`, { v }),

    startRender: (body: { ids?: string[]; overlay: boolean; reencode: boolean }) =>
      req<{ job_id: string }>(`${base}/render`, json(body)),
    renderJob: (jobId: string) => req<RenderJob>(`${base}/render/${jobId}`),
    statsUrl: mediaUrl(`${base}/stats`),
    fileUrl: (url: string) => mediaUrl(url),

    multiangle: () => req<MultiangleInfo>(`${base}/multiangle`),
    multiangleDirector: () => req<DirectorFull>(`${base}/multiangle/director`),
    putOffsets: (offsets: number[]) => req<PipelineStatus>(`${base}/multiangle/offsets`, json(offsets, "PUT")),
    angleVideoUrl: (i: number) => mediaUrl(`${base}/multiangle/angle/${i}/video`),
    recut: (style: "normal" | "fast", window?: [number, number] | null) =>
      req<PipelineStatus>(`${base}/multiangle/recut`, json({ style, window })),
    putMatchWindowMa: (start: number | null, end: number | null) =>
      req<{ match_window: [number, number] | null }>(
        `${base}/multiangle/match-window`, json({ start, end }, "PUT")),
    getZones: () =>
      req<{ angles: ZonePolygon[][]; ref_t?: (number | null)[] }>(
        `${base}/multiangle/zones`),
    putZones: (angles: ZonePolygon[][], ref_t?: (number | null)[]) =>
      req<{ angles: ZonePolygon[][] }>(`${base}/multiangle/zones`,
        json({ angles, ref_t }, "PUT")),
    angleFrameUrl: (i: number, t?: number) =>
      mediaUrl(`${base}/multiangle/angle/${i}/frame.jpg`,
        t == null ? {} : { t: String(t) }),
    listCuts: () => req<CutsList>(`${base}/multiangle/cuts`),
    activateCut: (id: string) =>
      req<CutsList>(`${base}/multiangle/cuts/${id}/activate`, json({})),
    deleteCut: (id: string) =>
      req<CutsList>(`${base}/multiangle/cuts/${id}`, { method: "DELETE" }),
    cutDownloadUrl: (id: string) =>
      mediaUrl(`${base}/multiangle/cuts/${id}/match.mp4`),

    project: () =>
      req<{ video: VideoInfo | null; candidates_version: number; proxy_ready: boolean; mode?: "single" | "multiangle" }>(`${base}/project`),
  };
}

export type ProjectApi = ReturnType<typeof projectApi>;

export const ProjectApiContext = createContext<ProjectApi | null>(null);

export function useProjectApi(): ProjectApi {
  const api = useContext(ProjectApiContext);
  if (!api) throw new Error("useProjectApi outside ProjectApiContext");
  return api;
}
