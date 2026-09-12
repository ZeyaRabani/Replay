"""End-to-end ``pitchworld run`` on synthetic clips with a fake tracker (no ultralytics / Modal)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pitchworld import __version__, modal_app
from pitchworld.cli import main
from pitchworld.sync import probe

# Synthetic 50x30 m pitch seen by a physical camera (x=25, y=-28, z=9 m, f=170 px, 18 deg down) behind the S
# touchline, 320x180 frame. Landmark pixels were generated with pitchworld.posefit.homography_world_to_pixel.
LANDMARKS = {"corner_A_S": [15.5, 89.5], "corner_B_S": [304.5, 89.5], "corner_B_N": [233.3, 62.5],
             "corner_A_N": [86.7, 62.5], "half_S": [160.0, 89.5], "half_N": [160.0, 62.5]}
PITCH = {"preset": "small_sided", "length": 50, "width": 30}


def _fake_track_local(video_path: str, model_name: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 1280,
                      classes=(0,), ball_imgsz: int = 1920) -> dict:
    """Two 'players' walking across the frame; feet stay inside the pitch quadrilateral."""
    info = probe(Path(video_path))
    n = round(info["duration"] * info["fps"])
    frames = []
    for f in range(n):
        s = f / max(1, n - 1)
        dets = []
        for tid, (x0, x1, foot_y) in enumerate(((60, 260, 85), (220, 110, 68)), start=1):
            xc = x0 + (x1 - x0) * s
            h = 30 if foot_y > 80 else 20
            dets.append({"id": tid, "cls": 0, "conf": 0.9, "box": [xc - 6, foot_y - h, xc + 6, foot_y]})
        frames.append(dets)
    return {"fps": info["fps"], "width": info["width"], "height": info["height"], "model": model_name,
            "frames": frames}


@pytest.fixture
def fake_tracker(monkeypatch):
    monkeypatch.setattr(modal_app, "track_local", _fake_track_local)


@pytest.fixture
def calib_files(tmp_path):
    calib = tmp_path / "calib.json"
    entry = {"points": [{"world": k, "pixel": v} for k, v in LANDMARKS.items()]}
    calib.write_text(json.dumps({"cameras": {"0": entry, "1": entry}}))
    pitch = tmp_path / "pitch.json"
    pitch.write_text(json.dumps(PITCH))
    return calib, pitch


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_sync_subcommand_writes_sync_json(make_clips, tmp_path):
    clips = make_clips([0.0, 0.5], seed=10)
    out = tmp_path / "out"
    assert main(["sync", *map(str, clips), "--out", str(out)]) == 0
    data = json.loads((out / "sync.json").read_text())
    assert set(data) >= {"offsets_s", "confidences", "method", "common_start_s", "common_duration_s", "low_confidence"}
    assert abs(data["offsets_s"][1] - 0.5) <= 0.02
    assert data["low_confidence"] is False


def test_run_rejects_bad_clip_count(tmp_path, capsys):
    assert main(["run", "only_one.mp4", "--out", str(tmp_path)]) == 2
    assert "need 2-4 clips" in capsys.readouterr().err
    assert main(["run", "a.mp4", "b.mp4", "--out", str(tmp_path)]) == 2
    assert "missing" in capsys.readouterr().err


def test_run_full_pipeline_writes_tracking_json(make_clips, calib_files, fake_tracker, tmp_path):
    calib, pitch = calib_files
    clips = make_clips([0.0, 0.5], seed=11)
    out = tmp_path / "out"
    rc = main(["run", *map(str, clips), "--out", str(out), "--local", "--calib", str(calib), "--pitch", str(pitch),
               "--no-viz"])
    assert rc == 0

    assert (out / "sync.json").exists()
    assert (out / "synced" / "cam0.mp4").exists() and (out / "synced" / "cam1.mp4").exists()
    assert (out / "debug" / "sync_check.jpg").exists()
    assert (out / "debug" / "calib_cam0.jpg").exists() and (out / "debug" / "calib_cam1.jpg").exists()
    assert (out / "debug" / "raw_tracks.json").exists()
    assert not (out / "debug" / "pitch_map.mp4").exists()

    tr = json.loads((out / "tracking.json").read_text())
    assert set(tr) == {"version", "pitch", "fps", "num_frames", "duration_s", "cameras", "quality", "frames"}
    assert tr["version"] == __version__
    assert tr["pitch"]["length"] == 50 and tr["pitch"]["width"] == 30
    assert tr["fps"] == 24.0
    assert tr["num_frames"] == len(tr["frames"]) > 0
    assert tr["duration_s"] == pytest.approx(tr["num_frames"] / tr["fps"], abs=1e-3)

    assert len(tr["cameras"]) == 2
    for i, cam in enumerate(tr["cameras"]):
        assert cam["index"] == i
        assert Path(cam["source"]).exists() and Path(cam["synced_clip"]).exists()
        assert cam["sync_method"] == ("reference" if i == 0 else "audio_xcorr")
        assert np.asarray(cam["homography_px_to_m"]).shape == (3, 3)
        assert set(cam["calibration"]) >= {"method", "confidence", "reproj_error_m", "reproj_error_px", "n_points"}
        assert cam["calibration"]["method"].startswith("manual")
        assert cam["calibration"]["n_points"] == len(LANDMARKS)
        assert cam["calibration"]["confidence"] >= 0.5
        assert cam["calibration"]["reproj_error_m"] < 0.1
        assert cam["frame_size"] == [320, 180]
    assert abs(tr["cameras"][1]["sync_offset_s"] - 0.5) <= 0.02

    q = tr["quality"]
    assert set(q) == {"sync_low_confidence", "calibration_low_confidence", "cross_camera_disagreement_m",
                      "warnings", "stats"}
    assert q["sync_low_confidence"] is False
    assert isinstance(q["calibration_low_confidence"], bool)
    assert isinstance(q["warnings"], list)
    assert "0-1" in q["cross_camera_disagreement_m"]
    # identical calibrations + identical fake detections -> cameras agree exactly
    assert q["cross_camera_disagreement_m"]["0-1"] == pytest.approx(0.0, abs=1e-6)
    assert q["calibration_low_confidence"] is False

    seen_players = 0
    for k, fr in enumerate(tr["frames"]):
        assert fr["frame"] == k
        assert fr["t"] == pytest.approx(k / tr["fps"], abs=1e-3)
        for p in fr["players"]:
            assert set(p) >= {"id", "x", "y", "conf", "cameras", "detections"}
            assert -3 <= p["x"] <= 53 and -3 <= p["y"] <= 33
            assert 0 < p["conf"] <= 1
            assert set(p["cameras"]) <= {0, 1}
            for d in p["detections"]:
                assert set(d) >= {"camera", "track_id", "box"} and len(d["box"]) == 4
            seen_players += 1
    assert seen_players > 0
    # both fake players seen by both cameras -> fused into 2 ids, not 4
    assert q["stats"]["unique_ids"] == 2
    for fr in tr["frames"][1:-1]:
        assert len(fr["players"]) == 2
        assert all(sorted(p["cameras"]) == [0, 1] for p in fr["players"])


def test_run_reuse_skips_tracking(make_clips, calib_files, fake_tracker, tmp_path, monkeypatch):
    calib, pitch = calib_files
    clips = make_clips([0.0, 0.5], seed=11)
    out = tmp_path / "out"
    args = ["run", *map(str, clips), "--out", str(out), "--local", "--calib", str(calib), "--pitch", str(pitch),
            "--no-viz"]
    assert main(args) == 0
    first = (out / "tracking.json").read_text()

    def boom(*a, **k):
        raise AssertionError("tracker must not run with --reuse")

    monkeypatch.setattr(modal_app, "track_local", boom)
    assert main([*args, "--reuse"]) == 0
    assert (out / "tracking.json").read_text() == first


def test_module_entrypoint_help():
    res = subprocess.run([sys.executable, "-m", "pitchworld.cli", "run", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    for flag in ("--local", "--calib", "--pitch", "--offsets", "--reuse", "--no-viz"):
        assert flag in res.stdout
