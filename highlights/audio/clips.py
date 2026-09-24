"""Cut short clips + contact-sheet frames for the top-N excitement events (for manual review)."""
import argparse
import json
import subprocess
from pathlib import Path


def main(video: Path, events: Path, out_dir: Path, n: int = 8, clip_s: float = 6.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = [e for e in json.loads(events.read_text())["events"] if e["type"] == "excitement"]
    ev.sort(key=lambda e: -e["confidence"])
    for k, e in enumerate(ev[:n], 1):
        t0 = max(0.0, e["t"] - clip_s / 2)
        tag = f"{k:02d}_{int(e['t'])}s"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", str(video), "-t", f"{clip_s}",
                        "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac",
                        str(out_dir / f"clip_{tag}.mp4")], check=True)
        # 6 frames, 1 s apart, tiled 3x2 at full resolution for inspection
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", str(video), "-t", f"{clip_s}",
                        "-vf", "fps=1,tile=3x2", "-frames:v", "1", str(out_dir / f"frames_{tag}.jpg")], check=True)
        print(tag, e["notes"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("events", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    main(a.video, a.events, a.out_dir, a.n)
