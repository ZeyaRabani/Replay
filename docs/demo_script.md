# Replay — 3-minute live demo script (judges)

Target: 3:00 talking, leave 1–2 min for questions. One presenter drives, one
narrates. Everything referenced below is pre-rendered; nothing is computed live
except the final `python -m pitchworld.demo_assets` call if the room's Wi-Fi is fine.

Prep checklist (before walking on):

- `out/` from the real 3-camera run open in a file browser; `out/demo/demo_preview.mp4` loaded in a player, paused on frame 0.
- Terminal at the repo root with the command typed but not executed: `python -m pitchworld.demo_assets out/ --highlight 8 16`.
- Stage 3 viewer tab open (if the viewer session shipped); otherwise `out/debug/pitch_map.mp4` open.
- Know the numbers: offsets 0 / −0.480 s / +0.350 s, loop-closure residual 20 ms, ~700 frames x 3 cams on Modal (yolov8m), cross-camera disagreement 1.7–2.6 m median, calibration flagged LOW.

---

## 0:00 — Hook (20 s)

> "Every weekend thousands of amateur matches are filmed on three or four phones from the touchline — and none of that footage ever becomes *data*. Replay takes 2–4 unsynchronised phone clips of one moment and rebuilds the moment: who was where, in metres, on the pitch — and hands it to a world model so you can watch it from anywhere."

*(Show the three raw clips side by side — shot 2 of the preview, paused.)*

## 0:20 — What we built, in one breath (25 s)

> "Four stages. Stage 1 is a Python pipeline, `pitchworld`: it syncs the clips by audio, calibrates each camera to a shared pitch frame, runs YOLOv8 + ByteTrack on Modal GPUs, and fuses the detections into one `tracking.json` in pitch coordinates. Stage 2 feeds that to Reactor's world-model API. Stage 3 is an interactive 3D viewer with cameras anchored to players. Stage 4 auto-generates this demo video with VEED's OpenEdit."

*(Press play on `demo_preview.mp4`; let it run under the next two sections.)*

## 0:45 — Execution: show it working (60 s)

**Sync.** *(2x2 grid with tile labels `cam1 −0.480 s`, `cam2 +0.350 s`.)*

> "No clapperboard. Audio cross-correlation gives us the offsets — here minus 480 and plus 350 milliseconds — and we check them: going cam0→cam1→cam2→cam0 closes to within 20 ms."

**Tracking.** *(overlay grid.)*

> "Per camera we run YOLOv8m with ByteTrack on Modal — about 700 frames per camera, all three in parallel. Those are the boxes you see."

**Fusion.** *(pitch map.)*

> "Feet go through each camera's homography into the same 50-by-30 pitch. Duplicates across cameras are merged by position and time, so one player has one ID, and every dot remembers which cameras saw it — that's the little `012` under each ID."

**Live command (optional, ~10 s).** Hit Enter on the prepared command.

> "And the whole demo you're watching was generated from that output folder by one command — title cards, grids and the stats card are ffmpeg; the shot list and composition go to VEED OpenEdit for the polished cut."

## 1:45 — World-model use and ambition (40 s)

*(Switch to Stage 3 viewer / Stage 2 output if available; otherwise stay on pitch map.)*

> "Here's why the pitch coordinates matter. `tracking.json` is exactly the conditioning a world model needs: a time series of agents in a metric frame plus the real camera poses. We send it to Reactor to synthesise the views we *didn't* film — a virtual camera on a player's shoulder, a broadcast angle, a bird's eye. Pick a player in the 3D viewer, and the camera anchors to them."

> "The ambition: any grassroots match becomes a navigable, re-watchable 3D moment, from phones people already own."

## 2:25 — Craft: honest confidence flags (25 s)

*(Stats card — shot 6. Don't skip it.)*

> "We want to be straight about where this is fragile. Sync is solid. Calibration is not yet: each camera is fitted from a hand-traced goal line and a blue arc — only about eight constraints for a seven-parameter pose — and we don't know the true pitch dimensions, so we guessed 50 by 30. The pipeline measures this itself: the three cameras disagree on player positions by roughly two metres median, and it stamps `calibration_low_confidence: true` on the output. Every downstream stage — the world model, the viewer, this video — shows that flag rather than hiding it."

> "The fix is known: more traced geometry per camera and a joint multi-camera bundle adjustment — `jointfit.py` already computes the cross-camera residual we'd minimise."

## 2:50 — Close (10 s)

> "Replay: three phones in, one moment out — synced, tracked, in metres, world-model ready, with its own confidence attached. Thanks."

---

## If things break

| Symptom | Do this |
| --- | --- |
| Viewer/Reactor demo not available | Stay on `pitch_map.mp4`; say "Stage 2/3 are in a parallel branch; here is the handoff artefact" and show `tracking.json` `quality` block in the terminal (`python -c "import json;print(json.load(open('out/tracking.json'))['quality'])"`). |
| `demo_assets` command fails live | Play the pre-rendered `demo_preview.mp4`; it is the same output. |
| Asked "why not a full homography from four corners?" | The phones only see one end of the pitch; no four well-spread corners are visible, hence line/arc constraints and a physical pose fit. |
| Asked "why Modal?" | Three cameras x 700 frames of yolov8m is minutes on GPU vs. an hour on a laptop CPU; one function, parallel per camera. |
| Asked about VEED | OpenEdit is agent-driven (no timeline API) and macOS-only today; we generate the shot list, intermediates and an HTML composition, and run the agent skill on a Mac. See `docs/stage4_veed.md`. |

## Judging-criteria map

- **World-model use** — 1:45 section: pitch-coordinate agents + camera poses as conditioning; virtual cameras.
- **Ambition** — hook and 1:45: grassroots footage → navigable 3D moment.
- **Execution** — 0:45 section: sync numbers, Modal tracking, fusion, one-command demo.
- **Craft** — 2:25 section: self-measured confidence flags carried through every stage.
