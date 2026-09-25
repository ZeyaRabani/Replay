"""Pydantic models shared by the API, store and CLI."""

from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal["goal", "shot", "chance", "excitement", "tackle", "other"]
CandidateStatus = Literal["pending", "confirmed", "rejected"]
Team = Literal["home", "away"]
CrossValidation = Literal["confirmed", "pipeline_only", "visual_only", "rejected"]


class RawEvent(BaseModel):
    """One event as it appears in a candidates.json input file."""

    type: EventType = "other"
    t: float
    t_start: float
    t_end: float
    confidence: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    cross_validation: CrossValidation = "pipeline_only"


class CandidatesFile(BaseModel):
    """Top-level candidates.json schema."""

    source: str = ""
    video_duration_s: float = 0.0
    events: list[RawEvent] = Field(default_factory=list)


class Candidate(BaseModel):
    """Internal candidate: raw event + review/edit state."""

    id: str
    type: EventType = "other"
    t: float
    t_start: float
    t_end: float
    confidence: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    cross_validation: CrossValidation = "pipeline_only"
    status: CandidateStatus = "pending"
    clip_start: float
    clip_end: float
    rank: int = 0


class CandidatePatch(BaseModel):
    status: CandidateStatus | None = None
    clip_start: float | None = None
    clip_end: float | None = None
    type: EventType | None = None
    notes: str | None = None
    team: Team | None = None


class VideoInfo(BaseModel):
    path: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    proxy_ready: bool = False
    registered_at: float = 0.0


class VideoRegisterRequest(BaseModel):
    path: str


class CandidatesLoadRequest(BaseModel):
    path: str


class RenderRequest(BaseModel):
    ids: list[str] | None = None
    overlay: bool = True
    reencode: bool = False


class ClipResult(BaseModel):
    id: str
    path: str
    url: str
    duration: float


class RenderJob(BaseModel):
    job_id: str
    state: Literal["queued", "running", "done", "error"] = "queued"
    progress: float = 0.0
    message: str = ""
    clips: list[ClipResult] = Field(default_factory=list)
    reel_url: str | None = None
    stats_url: str | None = None
    error: str | None = None
