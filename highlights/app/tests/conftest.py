"""Shared fixtures for backend tests."""

import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
SAMPLE = APP / "sample"
FAKE_PIPELINE = Path(__file__).resolve().parent / "fake_pipeline.py"
FAKE_MULTIANGLE = Path(__file__).resolve().parent / "fake_multiangle.py"


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "sample.mp4"
    subprocess.run(["bash", str(SAMPLE / "make_sample_video.sh"), str(out)], check=True)
    return out


@pytest.fixture(scope="session")
def short_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "short.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         "testsrc2=duration=6:size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True)
    return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_WORKDIR", str(tmp_path / "wd"))
    monkeypatch.setenv("HL_PIPELINE_CMD", f"{sys.executable} {FAKE_PIPELINE}")
    monkeypatch.setenv("HL_MULTIANGLE_CMD", f"{sys.executable} {FAKE_MULTIANGLE}")
    monkeypatch.setenv("HL_DEMO_VIDEO", "/nonexistent")
    from fastapi.testclient import TestClient

    import highlights.app.backend.main as m

    m.reset_registry()
    c = TestClient(m.app)
    r = c.post("/api/users", json={"name": "tester"})
    assert r.status_code == 200, r.text
    c.headers["X-User"] = "tester"
    return c


def new_project(client, video) -> str:
    """Create a project on a local file without running the pipeline."""
    r = client.post("/api/projects", json={"path": str(video), "run_pipeline": False})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def scoped(pid: str, path: str) -> str:
    return f"/api/projects/{pid}{path}"
