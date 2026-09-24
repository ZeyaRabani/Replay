export type EventType = "goal" | "shot" | "chance" | "excitement" | "other";
export type Status = "pending" | "confirmed" | "rejected";
export type CrossValidation = "confirmed" | "pipeline_only" | "visual_only" | "rejected";

export interface Candidate {
  id: string;
  type: EventType;
  t: number;
  t_start: number;
  t_end: number;
  confidence: number;
  signals: Record<string, unknown>;
  notes: string;
  cross_validation: CrossValidation;
  status: Status;
  clip_start: number;
  clip_end: number;
  rank: number;
}

export interface VideoInfo {
  path: string;
  duration_s: number;
  width: number;
  height: number;
  fps: number;
  proxy_ready: boolean;
  registered_at: number;
}

export interface RenderJob {
  job_id: string;
  state: "queued" | "running" | "done" | "error";
  progress: number;
  message: string;
  clips: { id: string; path: string; url: string; duration: number }[];
  reel_url: string | null;
  stats_url: string | null;
  error: string | null;
}
