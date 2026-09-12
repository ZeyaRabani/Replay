#!/usr/bin/env python3
"""Record a stand-in screen capture of the Stage 3 viewer and write jumps.json.

    python demo/record_viewer.py --url http://localhost:8765/ --out demo/out/rec.mp4 \
        --seconds 30 [--cdp http://localhost:29229] [--display :0 --size 1600x1200]

Drives the viewer through Chrome DevTools (Playwright over CDP): loads the
tracking sample, presses play, then jumps the camera between the busiest
players and the free anchors while ffmpeg x11grab records the screen. Every
jump is logged with its time offset into the recording as
`{"t": 4.2, "target": "player:35", "label": "Player 35"}` in `jumps.json`
next to the MP4 - this sidecar is what demo/make_demo.py times the captions to.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765/")
    ap.add_argument("--out", type=Path, default=Path("demo/out/rec.mp4"))
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--cdp", default="http://localhost:29229")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument("--size", default="1600x1200")
    ap.add_argument("--jump-every", type=float, default=4.0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(a.cdp)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        page.goto(a.url)
        page.wait_for_selector("#load-sample")
        page.click("#load-sample")
        page.wait_for_function("document.querySelectorAll('#players .player').length > 0", timeout=15000)
        page.bring_to_front()
        # players with the most tracked frames first (sidebar shows e.g. "281f")
        ids = page.eval_on_selector_all(
            "#players .player",
            "els => els.map(e => [e.dataset.id, parseInt((e.textContent.match(/(\\d+)f/) || [0, 0])[1])])"
            ".sort((a, b) => b[1] - a[1]).map(x => x[0])",
        )[:6]
        anchors = page.eval_on_selector_all(".anchor", "els => els.map(e => e.dataset.anchor).filter(x => x !== 'orbit')")
        plan = []
        for i in range(int(a.seconds // a.jump_every)):
            if i % 3 == 2 and anchors:
                plan.append(("anchor", anchors[(i // 3) % len(anchors)]))
            else:
                plan.append(("player", ids[i % len(ids)]))

        rec = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "x11grab", "-framerate", "30", "-video_size", a.size,
             "-i", a.display, "-t", str(a.seconds), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
             "-pix_fmt", "yuv420p", str(a.out)]
        )
        t0 = time.time()
        time.sleep(1.0)
        page.keyboard.press("Space")  # play
        jumps = [{"t": round(time.time() - t0, 2), "target": "orbit", "label": "Overview (orbit)"}]
        for i, (kind, target) in enumerate(plan):
            due = t0 + (i + 1) * a.jump_every
            time.sleep(max(0.0, due - time.time()))
            if time.time() - t0 > a.seconds - 1:
                break
            if kind == "player":
                page.click(f"#players .player[data-id='{target}']")
                label = f"Player {target}"
            else:
                page.click(f".anchor[data-anchor='{target}']")
                label = target.replace("-", " ").replace("_", " ").title()
            jumps.append({"t": round(time.time() - t0, 2), "target": f"{kind}:{target}", "label": label})
            # look around a little
            page.mouse.move(800, 500)
            page.mouse.down()
            page.mouse.move(900, 480, steps=15)
            page.mouse.up()
        rec.wait()
        sidecar = a.out.with_name("jumps.json")
        sidecar.write_text(json.dumps({"recording": a.out.name, "jumps": jumps}, indent=2))
        print(f"wrote {a.out} and {sidecar} ({len(jumps)} jumps)")


if __name__ == "__main__":
    main()
