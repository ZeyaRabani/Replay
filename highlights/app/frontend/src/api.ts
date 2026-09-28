import { createContext, useContext } from "react";
import type {
  AnalysisResponse,
  AnalysisStatus,
  CalibCamera,
  CalibLandmarks,
  CalibPoint,
  CalibResponse,
  Candidate,
  CutsList,
  DirectLearnResult,
  DirectSession,
  DirectSuggest,
  DirectSuggest3,
  DirectorFull,
  DownloadInfo,
  HistoryEvent,
  HistoryMatch,
  MultiangleInfo,
  PipelineStatus,
  PitchDims,
  PlayerGroup,
  PlayersPaths,
  PlayersResponse,
  RadarPitch,
  PlayersRoster,
  PlayersStats,
  PlayersV2Tracks,
  ProjectDetail,
  ProjectMeta,
  ProjectSummary,
  RenderJob,
  Stats,
  Team,
  User,
  VideoInfo,
  LearnedZones,
  ZoneKeyframe,
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
  createMultiangle: (title: string | undefined, angles: { url: string; label: string }[], cookies_text?: string, meta?: ProjectMeta, match_window?: [number, number], match_window_angle?: number) =>
    req<ProjectSummary>("/api/projects/multiangle", json({ title: title || undefined, angles, cookies_text, ...meta, match_window, match_window_angle })),
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
  rename: (id: string, title: string) =>
    req<{ id: string; title: string }>(`/api/projects/${id}`, json({ title }, "PATCH")),
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

    analysis: () => req<AnalysisResponse>(`${base}/analysis`),
    analyse: (force = false) => req<AnalysisStatus>(`${base}/analyse`, json({ force })),
    players: () => req<PlayersResponse>(`${base}/analysis/players`),
    analysePlayers: (force = false) => req<AnalysisStatus>(`${base}/analyse/players`, json({ force })),
    putRoster: (roster: PlayersRoster) =>
      req<{ roster: PlayersRoster; players_stats: PlayersStats }>(`${base}/analysis/players/roster`, json(roster, "PUT")),
    cropUrl: (name: string) => mediaUrl(`${base}/analysis/players/crops/${name}`),
    rebuildGroups: () =>
      req<{ groups: PlayerGroup[]; n_tracklets: number; n_grouped: number }>(
        `${base}/analysis/players/groups/rebuild`, json({})),
    playerPaths: () => req<PlayersPaths>(`${base}/analysis/players/paths`),
    calibLandmarks: () => req<CalibLandmarks>(`${base}/analysis/calib/landmarks`),
    calib: () => req<CalibResponse>(`${base}/analysis/calib`),
    putCalib: (angles: Record<string, { pts: CalibPoint[] }>) =>
      req<CalibResponse>(`${base}/analysis/calib`, json({ angles }, "PUT")),
    putCalibCameras: (cameras: Record<string, CalibCamera>) =>
      req<CalibResponse>(`${base}/analysis/calib/cameras`,
                         json({ cameras }, "PUT")),
    putCalibPitch: (pitch: PitchDims) =>
      req<CalibResponse>(`${base}/analysis/calib`, json({ pitch }, "PUT")),
    runPlayersV2: () => req<AnalysisStatus>(`${base}/analysis/players/v2/run`, json({})),
    playersV2Tracks: () => req<PlayersV2Tracks>(`${base}/analysis/players/v2/tracks`),
    radarPitch: () => req<RadarPitch>(`${base}/analysis/radar/pitch`),
    putRadarPitch: (corners: [number, number][], t: number | null) =>
      req<RadarPitch>(`${base}/analysis/radar/pitch`,
                      json({ corners, t }, "PUT")),

    multiangle: () => req<MultiangleInfo>(`${base}/multiangle`),
    multiangleDirector: () => req<DirectorFull>(`${base}/multiangle/director`),
    putOffsets: (offsets: number[]) => req<PipelineStatus>(`${base}/multiangle/offsets`, json(offsets, "PUT")),
    angleVideoUrl: (i: number) => mediaUrl(`${base}/multiangle/angle/${i}/video`),
    recut: (style: "normal" | "fast", window?: [number, number] | null,
            preview = false) =>
      req<PipelineStatus>(`${base}/multiangle/recut`,
                          json({ style, window, preview })),
    putMatchWindowMa: (start: number | null, end: number | null, angle = 0) =>
      req<{ match_window: [number, number] | null;
            match_window_src: { angle: number; start: number; end: number } | null }>(
        `${base}/multiangle/match-window`, json({ start, end, angle }, "PUT")),
    getZones: () =>
      req<{ version: 2; angles: ZoneKeyframe[][] }>(
        `${base}/multiangle/zones`),
    learnedZones: () =>
      req<LearnedZones>(`${base}/multiangle/zones/learned`),
    putZones: (angles: ZoneKeyframe[][]) =>
      req<{ version: 2; angles: ZoneKeyframe[][] }>(
        `${base}/multiangle/zones`, json({ angles }, "PUT")),
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
    directSuggest: (t?: [number, number]) =>
      req<DirectSuggest>(`${base}/multiangle/direct/suggest` +
        (t ? `?t_start=${t[0]}&t_end=${t[1]}` : "")),
    directSuggest3: () =>
      req<DirectSuggest3>(`${base}/multiangle/direct/suggest3`),
    directSessions: () =>
      req<{ sessions: DirectSession[] }>(`${base}/multiangle/direct/sessions`),
    saveDirectSession: (t_start: number, t_end: number,
                        choices: { t: number; angle: number }[]) =>
      req<DirectSession>(`${base}/multiangle/direct/sessions`,
                         json({ t_start, t_end, choices })),
    learnDirect: (recut: boolean) =>
      req<{ learn: DirectLearnResult; job: unknown }>(
        `${base}/multiangle/direct/learn`, json({ recut })),
    getDirectLearn: () =>
      req<DirectLearnResult>(`${base}/multiangle/direct/learn`),

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

export const historyApi = {
  list: (includeDeleted = true) =>
    req<HistoryMatch[]>(`/api/history?include_deleted=${includeDeleted ? 1 : 0}`),
  events: (id: string) => req<HistoryEvent[]>(`/api/history/${id}/events`),
  artefactUrl: (id: string, name: string) =>
    mediaUrl(`/api/history/${id}/artefacts/${encodeURIComponent(name)}`),
  restart: (id: string) =>
    req<ProjectSummary>(`/api/history/${id}/restart`, json({})),
  remove: (id: string) =>
    req<void>(`/api/history/${id}`, { method: "DELETE" }),
};

export const downloadsApi = {
  get: () => req<Record<string, DownloadInfo>>("/api/history/downloads"),
};

/** Canonical YouTube video id (same forms as backend history.video_id). */
export function ytId(url: string | null | undefined): string | null {
  if (!url) return null;
  let u = url.trim();
  if (!u.includes("://")) u = "https://" + u;
  let p: URL;
  try {
    p = new URL(u);
  } catch {
    return null;
  }
  const host = p.hostname.toLowerCase();
  const path = p.pathname.replace(/^\/+/, "");
  if (host.includes("youtu.be")) return path.split("/")[0] || null;
  if (host.includes("youtube.com") || host.includes("youtube-nocookie.com")) {
    const v = p.searchParams.get("v");
    if (v) return v;
    const [head, ...rest] = path.split("/");
    if (["shorts", "embed", "live", "v"].includes(head) && rest[0]) return rest[0];
  }
  return null;
}
