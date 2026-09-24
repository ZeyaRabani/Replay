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

export interface User {
  name: string;
  created_at: number;
  n_projects?: number;
}

export type PipelineState = "none" | "queued" | "running" | "done" | "failed";
export type PipelineStage =
  | "download"
  | "probe"
  | "audio"
  | "motion"
  | "features"
  | "score"
  | "candidates"
  | "stats"
  | "done";

export interface ProjectSource {
  kind: "youtube" | "upload" | "path";
  url?: string;
  filename?: string;
}

export interface ProjectVideo {
  duration_s: number;
  width: number;
  height: number;
  fps: number;
}

export interface ProjectSummary {
  id: string;
  title: string;
  created_at: number;
  source: ProjectSource;
  pipeline_state: PipelineState;
  progress: number;
  stage: string | null;
  message: string | null;
  video: ProjectVideo | null;
  n_candidates: number;
  n_confirmed: number;
  thumb_url: string | null;
}

export interface PipelineStatus {
  state: PipelineState;
  stage: PipelineStage | string;
  progress: number;
  stage_progress: number;
  message: string;
  error: string | null;
  started_at: number | null;
  updated_at: number | null;
  finished_at: number | null;
  pid: number | null;
  video_path: string | null;
  video: ProjectVideo | null;
  download: { format: string; resolution: string; filesize: number } | null;
  log?: string[];
}

export interface ProjectDetail extends ProjectSummary {
  pipeline: PipelineStatus | null;
}

export interface TimelineBin {
  t: number;
  motion: number;
  audio: number;
  excitement: number;
  events: number;
}

export interface TopMoment {
  t: number;
  type: EventType;
  confidence: number;
  reason: string;
}

export interface Stats {
  duration_s: number;
  match_window: [number, number];
  halves: { start: number; end: number }[];
  bin_s: number;
  timeline: TimelineBin[];
  events_by_type: Record<string, number>;
  events_per_10min: ({ t: number } & Record<string, number>)[];
  top_moments: TopMoment[];
  whistles: number[];
  activity: {
    mean_motion: number;
    peak_motion_t: number;
    loudest_t: number;
    quietest_stretch: [number, number];
  };
  pipeline: { model: string; auroc_reference: number; notes: string };
}
