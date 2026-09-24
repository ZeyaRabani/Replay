import type { Candidate, RenderJob, VideoInfo } from "./types";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const j = await r.json();
      msg = j.detail ?? msg;
    } catch {
      /* keep status */
    }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

export const api = {
  registerVideo: (path: string) =>
    req<VideoInfo>("/api/video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),
  getVideo: () => req<VideoInfo>("/api/video"),
  buildProxy: () => req<{ status: string }>("/api/video/proxy", { method: "POST" }),
  proxyStatus: () => req<{ ready: boolean; progress: number }>("/api/video/proxy/status"),
  loadCandidatesPath: (path: string) =>
    req<Candidate[]>("/api/candidates/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),
  loadCandidatesFile: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<Candidate[]>("/api/candidates/load", { method: "POST", body: fd });
  },
  listCandidates: (sort: "confidence" | "time") => req<Candidate[]>(`/api/candidates?sort=${sort}`),
  patchCandidate: (id: string, patch: Partial<Candidate>) =>
    req<Candidate>(`/api/candidates/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  resetCandidate: (id: string) => req<Candidate>(`/api/candidates/${id}/reset`, { method: "POST" }),
  startRender: (body: { ids?: string[]; overlay: boolean; reencode: boolean }) =>
    req<{ job_id: string }>("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  renderJob: (id: string) => req<RenderJob>(`/api/render/${id}`),
};
