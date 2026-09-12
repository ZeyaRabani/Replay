"""Generate a persistent HappyOyster world on Reactor from Stage 1 output.

Generic over any number of cameras: reads tracking.json for pitch dimensions and
player counts, picks the seed frame from the camera that sees the most players
(or --camera), letterboxes/crops it to a landscape aspect HappyOyster accepts
(1.5-2.0, <= 2 MB), builds the prompt, then talks to Reactor headlessly with the
Python SDK: connect -> upload_file -> create_world -> wait for world_state ready
-> save encrypted_world_id.

    REACTOR_API_KEY=rk_... python reactor-app/scripts/gen_world.py \
        --tracking out/tracking.json --frames full0.jpg full1.jpg full2.jpg \
        --out out/reactor [--camera 0] [--mode adventure|directing] [--dry-run]

Frames may also be extracted from the synced clips (--clips cam0.mp4 ...) at --t0.
Never put the API key in a file that is committed; the script only reads the env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

MODEL = {"adventure": "reactor/happy-oyster-adventure", "directing": "reactor/happy-oyster-director"}
TARGET_ASPECT = 16 / 9  # inside HappyOyster's 1.5-2.0 window
MAX_BYTES = 2 * 1024 * 1024
SEED_WIDTH = 1664


def load_tracking(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def best_camera(tracking: dict) -> int:
    """Camera whose detections cover the most frames -> most informative seed."""
    counts: dict[int, int] = {}
    for fr in tracking.get("frames", []):
        for p in fr.get("players", []):
            for c in p.get("cameras", p.get("sources", [])) or []:
                idx = c if isinstance(c, int) else c.get("camera", c.get("index"))
                if isinstance(idx, int):
                    counts[idx] = counts.get(idx, 0) + 1
    if not counts:
        return 0
    return max(counts, key=counts.get)


def extract_frame(clip: Path, t0: float, out: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", str(clip), "-frames:v", "1", str(out)],
        check=True,
    )
    return out


def build_seed(frame: Path, out: Path) -> Path:
    im = Image.open(frame).convert("RGB")
    w, h = im.size
    if w / h > TARGET_ASPECT:  # too wide -> centre crop
        tw = int(h * TARGET_ASPECT)
        x0 = (w - tw) // 2
        im = im.crop((x0, 0, x0 + tw, h))
    elif w / h < 1.5:  # too tall -> letterbox with dark bars
        th = int(w / TARGET_ASPECT)
        canvas = Image.new("RGB", (w, th), (20, 24, 20))
        canvas.paste(im, (0, (th - h) // 2))
        im = canvas
    im = im.resize((SEED_WIDTH, round(SEED_WIDTH / (im.size[0] / im.size[1]))))
    q = 92
    while True:
        im.save(out, "JPEG", quality=q)
        if out.stat().st_size <= MAX_BYTES or q <= 50:
            break
        q -= 8
    return out


def build_prompt(tracking: dict, n_cams: int) -> str:
    pitch = tracking.get("pitch", {})
    stats = tracking.get("quality", {}).get("stats", {})
    n_players = round(stats.get("mean_players_per_frame", 10))
    return (
        "Photorealistic third-person view of a real small-sided football pitch, "
        f"about {pitch.get('length', 50):.0f} by {pitch.get('width', 30):.0f} metres, "
        "artificial turf with bright green synthetic grass, crisp white boundary lines and a blue "
        "painted D-shaped arc in front of each goal, small white portable aluminium goals with white nets "
        "and rubber wheels, low metal fence and brick sports-hall buildings and tall floodlight masts around "
        "the pitch, overcast evening daylight with soft shadows. "
        f"About {n_players} amateur footballers playing five-a-side in two kits: orange bibs versus "
        "yellow-green bibs, running, passing and shouting, kit bags dumped along the touchline. "
        "The camera is a handheld broadcast camera about 2 metres above the ground that glides smoothly "
        f"around the pitch; the pitch, lines, goals and buildings stay fixed and consistent from every angle "
        f"(the scene was filmed by {n_cams} fixed cameras). Realistic proportions, no fantasy elements."
    )


def publish_seed(seed: Path) -> str:
    """HappyOyster's backend fetches the first frame by URL; a session FileRef is
    rejected with 400001. Push the seed to a public temp host and return the URL."""
    import requests

    r = requests.post(
        "https://tmpfiles.org/api/v1/upload",
        files={"file": (seed.name, seed.read_bytes(), "image/jpeg")},
        headers={"User-Agent": "curl/8.0 replay-gen-world"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")


async def create_world(model: str, seed: Path, seed_url: str | None, prompt: str, mode: str,
                       timeout_s: float, log) -> dict:
    from reactor_sdk import Reactor

    api_key = os.environ.get("REACTOR_API_KEY")
    if not api_key:
        sys.exit("REACTOR_API_KEY not set")
    loop = asyncio.get_running_loop()
    ready: asyncio.Future = loop.create_future()
    states: list[dict] = []

    reactor = Reactor(model_name=model, api_key=api_key, max_session_duration_seconds=600)

    @reactor.on_message
    def _msg(m):
        t = m.get("type")
        d = m.get("data", m)
        log(f"  <- {t}: {json.dumps(d)[:300]}")
        if t == "world_state":
            states.append(d)
            if d.get("phase") in ("ready", "failed") and not ready.done():
                ready.set_result(d)
        elif t == "action_error" and not ready.done():
            ready.set_exception(RuntimeError(f"action_error {d}"))

    @reactor.on_error
    def _err(e):
        log(f"  !! error {e}")

    t0 = time.time()
    await reactor.connect()
    log(f"connected, session={reactor.session_id} ({time.time() - t0:.1f}s)")
    if seed_url:
        payload = {"prompt": prompt, "first_frame_image_url": seed_url}
        log(f"seed url -> {seed_url}")
    else:
        ref = await reactor.upload_file(str(seed), mime_type="image/jpeg")
        log(f"uploaded seed -> {ref}")
        payload = {"prompt": prompt, "first_frame_image": ref}
    if mode == "adventure":
        payload["perspective"] = "third_person"
    else:
        payload["resolution"] = "720p"
    reply = await reactor.send_command("create_world", payload)
    log(f"create_world reply: {reply}")
    try:
        state = await asyncio.wait_for(ready, timeout_s)
    finally:
        await reactor.disconnect()
    state["_session_seconds"] = round(time.time() - t0, 1)
    state["_all_states"] = states
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking", required=True, type=Path)
    ap.add_argument("--frames", nargs="*", type=Path, default=[], help="one full frame per camera (any count)")
    ap.add_argument("--clips", nargs="*", type=Path, default=[], help="synced clips, one per camera (any count)")
    ap.add_argument("--t0", type=float, default=0.0, help="time in the synced clips to grab the seed frame")
    ap.add_argument("--camera", type=int, default=None, help="force the seed camera index")
    ap.add_argument("--mode", choices=list(MODEL), default="adventure")
    ap.add_argument("--prompt-file", type=Path, default=None, help="override the generated prompt")
    ap.add_argument("--seed-url", default=None, help="public URL of the seed (skips upload)")
    ap.add_argument("--upload", choices=["tmpfiles", "reactor"], default="tmpfiles",
                    help="how the seed reaches HappyOyster (reactor FileRef is rejected as of 2026-09-12)")
    ap.add_argument("--out", type=Path, default=Path("out/reactor"))
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--dry-run", action="store_true", help="write seed + prompt only, no API call")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    tracking = load_tracking(a.tracking)
    n_cams = len(tracking.get("cameras", [])) or max(len(a.frames), len(a.clips))
    cam = a.camera if a.camera is not None else best_camera(tracking)

    if a.frames:
        frame = a.frames[min(cam, len(a.frames) - 1)]
    elif a.clips:
        frame = extract_frame(a.clips[min(cam, len(a.clips) - 1)], a.t0, a.out / f"frame_cam{cam}.jpg")
    else:
        sys.exit("give --frames or --clips")

    seed = build_seed(frame, a.out / "seed.jpg")
    prompt = a.prompt_file.read_text().strip() if a.prompt_file else build_prompt(tracking, n_cams)
    (a.out / "prompt.txt").write_text(prompt + "\n")
    im = Image.open(seed)
    print(f"seed: {seed} {im.size} {seed.stat().st_size} bytes (camera {cam}); prompt {len(prompt)} chars")
    if a.dry_run:
        return 0

    log_lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        log_lines.append(s)

    seed_url = a.seed_url
    if seed_url is None and a.upload == "tmpfiles":
        seed_url = publish_seed(seed)
        log(f"published seed -> {seed_url}")
    state = asyncio.run(create_world(MODEL[a.mode], seed, seed_url, prompt, a.mode, a.timeout, log))
    record = {
        "model": MODEL[a.mode],
        "mode": a.mode,
        "encrypted_world_id": state.get("encrypted_world_id"),
        "phase": state.get("phase"),
        "world_status": state.get("world_status"),
        "first_frame": state.get("first_frame"),
        "prompt": prompt,
        "seed": str(seed),
        "seed_url": seed_url,
        "seed_camera": cam,
        "tracking": str(a.tracking),
        "quality": tracking.get("quality", {}),
        "session_seconds": state.get("_session_seconds"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = a.out / f"world_{a.mode}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    (a.out / f"gen_world_{a.mode}.log").write_text("\n".join(log_lines) + "\n")
    print(f"\nworld -> {out}\n  phase={record['phase']} id={record['encrypted_world_id']}")
    return 0 if record["phase"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
