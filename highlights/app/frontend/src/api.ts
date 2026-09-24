import { createContext, useContext } from "react";
import type {
  Candidate,
  PipelineStatus,
  ProjectDetail,
  ProjectSummary,
  RenderJob,
  Stats,
  User,
  VideoInfo,
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

export const meApi = {
  getCookies: () => req<{ saved: boolean; updated_at: number | null }>("/api/me/youtube-cookies"),
  saveCookies: (cookies_text: string) =>
    req<{ saved: boolean; updated_at: number }>("/api/me/youtube-cookies", json({ cookies_text }, "PUT")),
  deleteCookies: () => req<void>("/api/me/youtube-cookies", { method: "DELETE" }),
};

export const configApi = {
  get: () => req<{ upload_origin: string | null }>("/api/config"),
};

export const projectsApi = {
  list: () => req<ProjectSummary[]>("/api/projects"),
  createYoutube: (youtube_url: string, title?: string) =>
    req<ProjectSummary>("/api/projects", json({ youtube_url, title: title || undefined })),
  createPath: (path: string, title?: string) =>
    req<ProjectSummary>("/api/projects", json({ path, title: title || undefined })),
  createUpload: (file: File, title?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (title) fd.append("title", title);
    return req<ProjectSummary>("/api/projects", { method: "POST", body: fd });
  },
  remove: (id: string) => req<void>(`/api/projects/${id}`, { method: "DELETE" }),
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

    loadCandidatesPath: (path: string) =>
      req<Candidate[]>(`${base}/candidates/load`, json({ path })).then(() => undefined),
    loadCandidatesFile: (file: File) => {
      const fd = new FormData();
      fd.append("file", file);
      return req<Candidate[]>(`${base}/candidates/load`, { method: "POST", body: fd }).then(() => undefined);
    },
    listCandidates: (sort: "confidence" | "time") => req<Candidate[]>(`${base}/candidates?sort=${sort}`),
    patchCandidate: (cid: string, patch: Partial<Candidate>) =>
      req<Candidate>(`${base}/candidates/${cid}`, json(patch, "PATCH")),
    resetCandidate: (cid: string) => req<Candidate>(`${base}/candidates/${cid}/reset`, { method: "POST" }),
    thumbUrl: (cid: string, v?: string) => mediaUrl(`${base}/candidates/${cid}/thumb.jpg`, { v }),

    startRender: (body: { ids?: string[]; overlay: boolean; reencode: boolean }) =>
      req<{ job_id: string }>(`${base}/render`, json(body)),
    renderJob: (jobId: string) => req<RenderJob>(`${base}/render/${jobId}`),
    statsUrl: mediaUrl(`${base}/stats`),
    fileUrl: (url: string) => mediaUrl(url),

    project: () =>
      req<{ video: VideoInfo | null; candidates_version: number; proxy_ready: boolean }>(`${base}/project`),
  };
}

export type ProjectApi = ReturnType<typeof projectApi>;

export const ProjectApiContext = createContext<ProjectApi | null>(null);

export function useProjectApi(): ProjectApi {
  const api = useContext(ProjectApiContext);
  if (!api) throw new Error("useProjectApi outside ProjectApiContext");
  return api;
}
