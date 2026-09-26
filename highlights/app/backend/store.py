"""Multi-user project registry + per-project state persisted as JSON."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from .schemas import Candidate, CandidatesFile, VideoInfo

PAD_GOAL = 5.0
PAD_DEFAULT = 3.0

PROJECT_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def workdir() -> Path:
    env = os.environ.get("HL_WORKDIR")
    root = Path(env) if env else Path.home() / ".replay_highlights"
    root.mkdir(parents=True, exist_ok=True)
    (root / "projects").mkdir(exist_ok=True)
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

    def __init__(
        self,
        root: Path,
        *,
        id: str | None = None,
        owner: str = "",
        title: str = "",
        source: dict | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        for d in ("thumbs", "renders", "source", "pipeline"):
            (self.root / d).mkdir(exist_ok=True)
        self.id = id or self.root.name
        self.owner = owner
        self.title = title
        self.created_at = time.time()
        self.source_info: dict = source or {"kind": "path", "url": None, "filename": None}
        if self.is_multiangle:
            self.multiangle_dir.mkdir(exist_ok=True)
            for i in range(len(self.source_info.get("angles") or [])):
                self.angle_dir(i).mkdir(parents=True, exist_ok=True)
        self.pipeline_state: str = "none"
        self.meta: dict = {}
        self.state_path = self.root / "project.json"
        self.lock = threading.RLock()
        self.video: VideoInfo | None = None
        self.source: str = ""  # candidates-file track string
        self.candidates: list[Candidate] = []
        self.proxy_complete: bool = False
        self.proxy_source: str = ""
        self.candidates_version: int = 0
        self._load()

    @property
    def pipeline_dir(self) -> Path:
        return self.root / "pipeline"

    @property
    def is_multiangle(self) -> bool:
        return self.source_info.get("kind") == "multiangle"

    @property
    def multiangle_dir(self) -> Path:
        return self.root / "multiangle"

    @property
    def angles_dir(self) -> Path:
        return self.root / "angles"

    def angle_dir(self, i: int) -> Path:
        return self.angles_dir / f"a{i}"

    def angle_video(self, i: int) -> Path | None:
        d = self.angle_dir(i)
        if not d.is_dir():
            return None
        for f in sorted(d.iterdir()):
            if f.is_file() and f.name.startswith("match."):
                return f
        return None

    @property
    def status_path(self) -> Path:
        if self.is_multiangle:
            return self.multiangle_dir / "status.json"
        return self.pipeline_dir / "status.json"

    @property
    def log_path(self) -> Path:
        if self.is_multiangle:
            return self.multiangle_dir / "log.txt"
        return self.pipeline_dir / "log.txt"

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            self.id = data.get("id", self.id)
            self.owner = data.get("owner", self.owner)
            self.title = data.get("title", self.title)
            self.created_at = float(data.get("created_at", self.created_at))
            src = data.get("source")
            if isinstance(src, dict):
                self.source_info = src
            elif isinstance(src, str):
                # old flat format: "source" was the candidates track string
                self.source = src
            if "candidates_source" in data:
                self.source = data["candidates_source"]
            if data.get("video"):
                self.video = VideoInfo(**data["video"])
            self.candidates = [Candidate(**c) for c in data.get("candidates", [])]
            self.proxy_complete = bool(data.get("proxy_complete", False))
            self.proxy_source = data.get("proxy_source", "")
            self.candidates_version = int(data.get("candidates_version", 0))
            self.pipeline_state = data.get("pipeline_state", "none")
            self.meta = data.get("meta") or {}
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
                        "id": self.id,
                        "owner": self.owner,
                        "title": self.title,
                        "created_at": self.created_at,
                        "source": self.source_info,
                        "video": self.video.model_dump() if self.video else None,
                        "candidates_version": self.candidates_version,
                        "candidates": [c.model_dump() for c in self.candidates],
                        "proxy_complete": self.proxy_complete,
                        "proxy_source": self.proxy_source,
                        "pipeline_state": self.pipeline_state,
                        "meta": self.meta,
                        "candidates_source": self.source,
                    },
                    indent=2,
                )
            )
            tmp.replace(self.state_path)

    def set_pipeline_state(self, state: str) -> None:
        with self.lock:
            self.pipeline_state = state
            self.save()

    def set_video(self, info: VideoInfo) -> None:
        with self.lock:
            self.video = info
            self.revalidate_windows(info.duration_s)
            self.save()

    def revalidate_windows(self, duration: float) -> None:
        """Reset candidate clip windows that are invalid for `duration`.

        Candidates whose event t lies beyond the new duration get a short
        valid window at the end of the video (t itself is preserved).
        """
        if duration <= 0:
            return
        for c in self.candidates:
            ok = 0 <= c.clip_start < c.clip_end <= duration
            if ok:
                continue
            if c.t <= duration:
                c.clip_start, c.clip_end = default_clip_window(
                    c.t, c.type, duration)
            else:
                pad = PAD_GOAL if c.type == "goal" else PAD_DEFAULT
                c.clip_end = duration
                c.clip_start = max(0.0, duration - 2 * pad)

    def load_candidates(self, cf: CandidatesFile) -> list[Candidate]:
        with self.lock:
            duration = self.video.duration_s if self.video else cf.video_duration_s
            self.source = cf.source
            self.candidates = make_candidates(cf, duration)
            self.revalidate_windows(duration)
            self.candidates_version += 1
            self.save()
            return sorted(self.candidates, key=lambda c: -c.confidence)

    def set_proxy_complete(self, source: str) -> None:
        with self.lock:
            self.proxy_complete = True
            self.proxy_source = source
            self.save()

    def invalidate_video(self) -> None:
        """Drop proxy artifacts + thumbnail cache for a previous video."""
        with self.lock:
            for name in ("proxy.mp4", "proxy.part.mp4"):
                (self.root / name).unlink(missing_ok=True)
            shutil.rmtree(self.root / "thumbs", ignore_errors=True)
            (self.root / "thumbs").mkdir(exist_ok=True)
            self.proxy_complete = False
            self.proxy_source = ""
            self.save()

    def thumb_dir(self) -> Path:
        """Thumbnail cache dir keyed by video path + mtime (never stale)."""
        key = "none"
        if self.video is not None:
            p = Path(self.video.path)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            key = hashlib.md5(f"{self.video.path}|{mtime}".encode()).hexdigest()[:10]
        d = self.root / "thumbs" / key
        d.mkdir(parents=True, exist_ok=True)
        return d

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


MERGE_USERS = {"demo": "admin", "john": "admin", "yusuf": "zeya"}
DROP_USERS = {"watch", "qa", "testing"}


class Registry:
    """Users + projects rooted at a workdir. Thread-safe."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.projects_dir = root / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.users_path = root / "users.json"
        self.lock = threading.RLock()
        self._projects: dict[str, ProjectStore] = {}
        self._migrate_users()

    def _migrate_users(self) -> None:
        """One-off profile cleanup: demo/john profiles merge into admin
        (their projects keep ownership via reassignment); throwaway test
        profiles are dropped unless they still own projects."""
        with self.lock:
            users = self._read_users()
            projects = self.list_projects()
            if (not any(u["name"] in MERGE_USERS or u["name"] in DROP_USERS
                        for u in users)
                    and not any(p.owner in MERGE_USERS for p in projects)):
                return
            for p in projects:
                if p.owner in MERGE_USERS:
                    p.owner = MERGE_USERS[p.owner]
                    p.save()
            owners = {p.owner for p in self.list_projects()}
            have = {u["name"] for u in users}
            keep = []
            for u in users:
                n = u["name"]
                if n in MERGE_USERS:
                    if MERGE_USERS[n] not in have:
                        keep.append({"name": MERGE_USERS[n],
                                     "created_at": u.get("created_at",
                                                         time.time())})
                        have.add(MERGE_USERS[n])
                    continue
                if n in DROP_USERS and n not in owners:
                    continue
                keep.append(u)
            self._write_users(keep)

    # ---------- users ----------

    def _read_users(self) -> list[dict]:
        try:
            return json.loads(self.users_path.read_text()).get("users", [])
        except Exception:
            return []

    def _write_users(self, users: list[dict]) -> None:
        tmp = self.users_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"users": users}, indent=2))
        tmp.replace(self.users_path)

    def list_users(self) -> list[dict]:
        with self.lock:
            return self._read_users()

    def get_user(self, name: str) -> dict | None:
        with self.lock:
            for u in self._read_users():
                if u["name"] == name:
                    return u
            return None

    def add_user(self, name: str) -> dict:
        with self.lock:
            users = self._read_users()
            for u in users:
                if u["name"] == name:
                    return u
            u = {"name": name, "created_at": time.time()}
            users.append(u)
            self._write_users(users)
            return u

    # ---------- projects ----------

    @staticmethod
    def new_project_id() -> str:
        return uuid.uuid4().hex[:12]

    def create_project(self, owner: str, title: str, source: dict,
                       meta: dict | None = None) -> ProjectStore:
        with self.lock:
            pid = self.new_project_id()
            p = ProjectStore(
                self.projects_dir / pid,
                id=pid, owner=owner, title=title, source=source,
            )
            p.meta = {k: v for k, v in (meta or {}).items() if v is not None}
            p.save()
            self._projects[pid] = p
            return p

    def get(self, project_id: str) -> ProjectStore | None:
        if not PROJECT_ID_RE.match(project_id or ""):
            return None
        with self.lock:
            if project_id in self._projects:
                return self._projects[project_id]
            state = self.projects_dir / project_id / "project.json"
            if not state.is_file():
                return None
            p = ProjectStore(self.projects_dir / project_id)
            self._projects[project_id] = p
            return p

    def list_projects(self, owner: str | None = None) -> list[ProjectStore]:
        with self.lock:
            out: list[ProjectStore] = []
            if not self.projects_dir.is_dir():
                return []
            for d in self.projects_dir.iterdir():
                if not d.is_dir() or not (d / "project.json").is_file():
                    continue
                p = self.get(d.name)
                if p is None:
                    continue
                if owner is not None and p.owner != owner:
                    continue
                out.append(p)
            out.sort(key=lambda p: p.created_at, reverse=True)
            return out

    def newest_for(self, owner: str) -> ProjectStore | None:
        projects = self.list_projects(owner)
        return projects[0] if projects else None

    def delete(self, project_id: str) -> None:
        with self.lock:
            self._projects.pop(project_id, None)
            shutil.rmtree(self.projects_dir / project_id, ignore_errors=True)
