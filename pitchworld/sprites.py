"""Cut one representative image crop per unified player id from the synced clips -> viewer/sprites/<id>.png + index.json."""
import json
import sys
from pathlib import Path

import cv2

tracking = json.load(open(sys.argv[1]))
clips = sys.argv[2:-1]
out = Path(sys.argv[-1])
out.mkdir(parents=True, exist_ok=True)

# best detection per id: prefer high conf * area, mid-timeline
best = {}
for fr in tracking["frames"]:
    for p in fr["players"]:
        for d in p.get("detections", []):
            x0, y0, x1, y1 = d["box"]
            area = (x1 - x0) * (y1 - y0)
            score = area * p.get("conf", 1.0)
            if p["id"] not in best or score > best[p["id"]][0]:
                best[p["id"]] = (score, fr["frame"], d["camera"], d["box"])

by_cam = {}
for pid, (_, frame, cam, box) in best.items():
    by_cam.setdefault(cam, []).append((frame, pid, box))

index = {}
for cam, items in by_cam.items():
    cap = cv2.VideoCapture(clips[cam])
    for frame, pid, box in sorted(items):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        x0 = max(0, int(x0 - 0.15 * bw)); x1 = min(w, int(x1 + 0.15 * bw))
        y0 = max(0, int(y0 - 0.08 * bh)); y1 = min(h, int(y1 + 0.03 * bh))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = 256 / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), 256), interpolation=cv2.INTER_AREA)
        # soft oval alpha mask so the billboard reads as a cut-out
        import numpy as np
        ch, cw = crop.shape[:2]
        yy, xx = np.mgrid[0:ch, 0:cw]
        m = ((xx - cw / 2) / (cw / 2)) ** 2 + ((yy - ch / 2) / (ch / 2)) ** 2
        alpha = np.clip((1.15 - m) / 0.3, 0, 1) * 255
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha.astype("uint8")
        cv2.imwrite(str(out / f"{pid}.png"), rgba)
        index[str(pid)] = {"aspect": cw / ch, "camera": cam, "frame": frame}
    cap.release()

json.dump(index, open(out / "index.json", "w"))
print(len(index), "sprites")
