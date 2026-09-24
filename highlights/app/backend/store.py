"""Project state: video info, candidates, persisted to a JSON workdir file."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .schemas import Candidate, CandidatesFile, VideoInfo

PAD_GOAL = 5.0
PAD_DEFAULT = 3.0


def workdir() -> Path:
    env = os.environ.get("HL_WORKDIR")
    root = Path(env) if env else Path(__file__).resolve().parents[1] / "workdir"
    root.mkdir(parents=True, exist_ok=True)
    (root / "thumbs").mkdir(exist_ok=True)
    (root / "renders").mkdir(exist_ok=True)
    return root


def default_clip_window(t: float, event_type: str, duration: float) -> tuple[float, float]:
    pad = PAD_GOAL if event_type == "goal" else PAD_DEFAULT
    start = max(0.0, t - pad)
    end = min(duration, t + pad) if duration > 0 else t + pad
    return start, end


def make_candidates(cf: CandidatesFile, duration: float) -> list[Candidate]:
    dur = duration or cf.video_duration_s
    out: list[Candidate] = []
    for i, ev in enumerate(cf.events):
        status = "rejected" if ev.cross_validation == "rejected" else "pending"
        start, end = default_clip_window(ev.t, ev.type, dur)
        out.append(
            Candidate(
                id=f"c{i + 1:03d}",
                status=status,
                clip_start=start,
                clip_end=end,
                **ev.model_dump(),
            )
        )
    # rank by confidence desc, 1-based
    for rank, c in enumerate(sorted(out, key=lambda c: -c.confidence), start=1):
        c.rank = rank
    return out


class ProjectStore:
    """Single-project state, JSON persisted so restarts survive."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or workdir()
        self.state_path = self.root / "project.json"
        self.lock = threading.RLock()
        self.video: VideoInfo | None = None
        self.source: str = ""
        self.candidates: list[Candidate] = []
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            if data.get("video"):
                self.video = VideoInfo(**data["video"])
            self.source = data.get("source", "")
            self.candidates = [Candidate(**c) for c in data.get("candidates", [])]
        except Exception:
            # corrupt state -> start fresh rather than crash
            self.video = None
            self.candidates = []

    def save(self) -> None:
        with self.lock:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "video": self.video.model_dump() if self.video else None,
                        "source": self.source,
                        "candidates": [c.model_dump() for c in self.candidates],
                    },
                    indent=2,
                )
            )
            tmp.replace(self.state_path)

    def set_video(self, info: VideoInfo) -> None:
        with self.lock:
            self.video = info
            self.save()

    def load_candidates(self, cf: CandidatesFile) -> list[Candidate]:
        with self.lock:
            duration = self.video.duration_s if self.video else cf.video_duration_s
            self.source = cf.source
            self.candidates = make_candidates(cf, duration)
            self.save()
            return self.candidates

    def get(self, cand_id: str) -> Candidate | None:
        with self.lock:
            for c in self.candidates:
                if c.id == cand_id:
                    return c
        return None

    def update(self, cand: Candidate) -> None:
        with self.lock:
            self.save()

    def reset_candidate(self, cand_id: str) -> Candidate | None:
        with self.lock:
            c = self.get(cand_id)
            if c is None:
                return None
            duration = self.video.duration_s if self.video else 0.0
            c.clip_start, c.clip_end = default_clip_window(c.t, c.type, duration)
            self.save()
            return c


STORE = ProjectStore()
