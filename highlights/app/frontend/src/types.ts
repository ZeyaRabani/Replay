export type EventType =
  | "goal"
  | "shot"
  | "goalmouth"
  | "crowd"
  | "attack"
  | "chance"   // legacy projects
  | "excitement"
  | "other";
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
  created_at: number | string;
  n_projects?: number;
}

export type PipelineState = "none" | "queued" | "running" | "done" | "failed" | "needs_input";
export type Team = "home" | "away";
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
  kind: "youtube" | "upload" | "path" | "multiangle";
  url?: string | null;
  filename?: string | null;
  angles?: { url: string | null; filename: string | null; label: string }[];
}

export interface ProjectVideo {
  duration_s: number;
  width: number;
  height: number;
  fps: number;
}

export interface ProjectMeta {
  pitch_type?: "11" | "9" | "7" | "5" | "other";
  camera?: "normal" | "ultrawide" | "zoom" | "other";
}

export interface ProjectSummary {
  id: string;
  title: string;
  created_at: number | string;
  source: ProjectSource;
  meta?: ProjectMeta;
  pipeline_state: PipelineState;
  progress: number;
  stage: string | null;
  message: string | null;
  video: ProjectVideo | null;
  n_candidates: number;
  n_confirmed: number;
  thumb_url: string | null;
  mode: "single" | "multiangle";
  n_angles: number;
}

export interface SyncPair {
  a: number;
  b: number;
  offset: number;
  pnr: number;
  r2: number;
  confident: boolean;
  accepted_by?: "pnr" | "triangle" | "manual" | null;
  refine_spread_s?: number;
  manual?: boolean;
  consistent?: boolean;
}

export interface SyncInfo {
  reference: number;
  method: "xcorr" | "xcorr+triangle" | "manual" | string;
  offsets: number[];
  pairs: SyncPair[];
  triangle_residual_s: number | null;
  needs_manual: number[];
  coverage: Record<string, unknown>;
  confidence_note?: string;
}

export interface DirectorSummary {
  per_second_rule: Record<string, number>;
  ratios: Record<string, number>;
  n_cuts: number;
  mean_hold_s: number;
  median_hold_s?: number;
  min_hold_s?: number;
  cuts_per_10min?: number;
  angle_share: Record<string, number>;
  cluster_baseline?: number[];
}

export interface DirectorSegment {
  t_start: number;
  t_end: number;
  angle: number;
  rule: "ball" | "cluster" | "hold" | "coverage" | "start" | string;
  score: number;
  runner_up?: { angle: number; score: number } | null;
}

export interface DirectorFull extends DirectorSummary {
  segments: DirectorSegment[];
}

export interface AngleInfo {
  index: number;
  label: string;
  url: string | null;
  filename: string | null;
  duration: number | null;
  status: string | null;
  has_file: boolean;
}

export interface MultiangleScore {
  home: { label: string; goals: number };
  away: { label: string; goals: number };
  unassigned?: number;
  basis?: string;
}

export interface MultiangleInfo {
  sync: SyncInfo | null;
  director: DirectorSummary | null;
  angles: AngleInfo[];
  score: MultiangleScore;
  status: PipelineStatus | null;
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

export interface MatchStats {
  goals: number;
  shots_on_goal: number;
  goalmouth_actions: number;
  attacks: number;
  crowd_reactions: number;
  big_moments: number;
  territory: { near_goal_pct: number; far_goal_pct: number };
  tempo: { mean_motion_pct: number; high_intensity_pct: number };
  stoppages: {
    whistles: number;
    quiet_stretches: number;
    estimated_stoppage_pct: number;
  };
  halves: {
    start: number;
    end: number;
    goals: number;
    shots_on_goal: number;
    attacks: number;
    mean_motion_pct: number;
  }[];
  peak_minute: { t: number; events: number };
  basis?: string;
}

export interface Stats {
  duration_s: number;
  match_stats?: MatchStats;
  match_window: [number, number];
  halves: { start: number; end: number }[];
  // match-level broadcast stats (new pipeline only)
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
  pipeline: { model: string; auroc_reference: number | null; notes: string };
  multiangle?: {
    sync?: SyncInfo;
    director?: DirectorSummary;
    confirmation?: { cross: number; single: number; disputed: number };
    score?: MultiangleScore;
  };
}
