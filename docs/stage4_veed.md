# Stage 4 — auto-generated demo video with VEED OpenEdit

Stage 4 turns a finished `pitchworld` output directory into a short, judge-facing
demo video. The plan is two layers:

1. **Offline intermediates (this repo, `pitchworld/demo_assets.py`).** Fully
   deterministic ffmpeg renders: title cards, a 2x2 grid of the synced cameras,
   the fused pitch map with a title, and an honest stats/confidence card built
   from `tracking.json["quality"]`. It also writes `shot_list.json`, a stitched
   `demo_preview.mp4` (the fallback deliverable if VEED is unavailable), and a
   serialised VEED request (`veed_request.json` + `veed_composition.html`).
2. **VEED OpenEdit (hackathon sponsor tool).** An agent running the `open-edit`
   skill takes the shot list + intermediates and produces the polished cut
   (motion-graphic titles, captions, transitions) rendered with VEED Engine.

```bash
python -m pitchworld.demo_assets out/ --demo out/demo            # offline assets + preview + VEED request
python -m pitchworld.demo_assets out/ --highlight 6 14           # choose the highlight window (seconds)
```

## What VEED "OpenEdit" actually is (research notes)

Legend: **[verified]** = read directly from the linked source on 2026-09-12;
**[inferred]** = our reading between the lines, not confirmed.

| Fact | Status | Source |
| --- | --- | --- |
| OpenEdit is VEED's open-source, *agent-driven* editing pipeline; "There is no GUI and no timeline." The editor is Apache-2.0; the renderer binaries (VEED Engine, `veed-engine-cli`) are closed but free (PolyForm Shield 1.0.0). | verified | https://github.com/veedstudio/open-edit |
| Install is `npx skills add veedstudio/open-edit`; the agent (Claude Code / Codex / Gemini) then loads the repo's `AGENTS.md` and `.claude/skills/open-edit/SKILL.md`. | verified | https://github.com/veedstudio/open-edit, https://support.veed.io/en/articles/16342833-how-to-use-openedit-veed-s-agent-driven-video-editor |
| Platform: Apple Silicon Mac on macOS 26 (Tahoe) only. "Windows, Linux: planned". | verified | same two links |
| Compositions are authored in **HTML + CSS** and rendered to video by VEED Engine (no headless browser); Chrome can be used as an alternative render backend. | verified | same |
| Capabilities beyond captions: "edit, cut and reframe footage", "layer motion graphics and visual elements (charts, titles, lower thirds, overlays, transitions)", "turn slides or websites into video", "capture web pages". Source video is optional. | verified | support article |
| The pipeline is a set of `npx @veedstudio/openedit-cli` commands: `init` (preflight), `transcribe` (`--provider veed` for the hosted route), `prep` (`meta.json`, `word-timings.json`), `sample-style`, `generate-recipe --run runs/<id> --record` (compiles a recipe into a `.wv` document, lints, `--verify`, renders `out.silent.mp4`), `mux-audio`, `mix-audio`, `stills`, `background-removal`, `lipsync`. Deliverable is `runs/<id>/final/out.mp4`. | verified | https://raw.githubusercontent.com/veedstudio/open-edit/main/AGENTS.md |
| A `.wv` document is an HTML page sized to the canvas (16:9 reference canvas 1280x720, placeholders `{W} {H} {FPS} {DUR} {videoPath}`), with a `.vid` video layer at z0, `.cue` gate elements with `animation-delay` / `animation-duration` in ms, and one span per word/glyph whose `animation-delay` is the word's timing. Every px scales by `SCALE = W / 1280` for 16:9. | verified | https://raw.githubusercontent.com/veedstudio/open-edit/main/docs/recipe-format.md |
| Transcription (captions) needs a VEED login (`veed/.veed-token.json`); only the audio track is uploaded. Compose-and-render-only jobs need no account. | verified | README + support article |
| There is **no hosted REST "timeline" API** for OpenEdit: you cannot POST a shot list and get an MP4 back. Programmatic assembly = write (or have the agent write) the HTML/CSS composition and run the local CLI. VEED's separate hosted APIs are the Subtitles API and the VEED API (Fabric generation, background removal, Lipsync), which are not editors. | inferred from the docs above (no API reference found for a timeline endpoint) | https://www.veed.io/tools/openedit |
| Render time "typically one to three minutes from prompt to finished MP4". | verified | support article |

### Consequences for Replay

* Our CI/dev boxes are Linux, so **OpenEdit cannot run in this repo's pipeline
  today**. Stage 4 therefore ships everything an OpenEdit agent needs, and a
  plain-ffmpeg `demo_preview.mp4` so the demo never depends on a Mac.
* The "programmatic timeline" is the HTML composition. `submit_to_veed()`
  emits `veed_composition.html` in the `.wv` idiom (a `.vid` layer per shot,
  `.cue` captions gated by `animation-delay`) plus a natural-language brief in
  `veed_request.json`. On a Mac the run is:

```bash
npx skills add veedstudio/open-edit          # once
claude                                       # or codex / gemini
> Using /open-edit, build a 16:9 1280x720 demo from out/demo/shot_list.json.
> Use out/demo/veed_composition.html as the starting composition, keep the
> shot order and captions, add animated titles and a subtle wipe between shots,
> and burn in the captions from the "caption" fields. Deliver final/out.mp4.
```

  (The prompt is copied verbatim from `veed_request.json["brief"]`.)

* No captions-from-speech are needed (there is no narration in the source
  clips), so the run does not need a VEED account: it is compose-and-render
  only. If we add a voice-over later, `openedit-cli transcribe --provider veed`
  gives real per-word timings for the caption recipes.

## Files produced by `demo_assets.py`

| File | Purpose |
| --- | --- |
| `demo/00_title.mp4` … `demo/06_next.mp4` | One MP4 per shot, all 1280x720 @ 25 fps, H.264 + silent/AAC audio, so they concat losslessly. |
| `demo/shot_list.json` | Ordered shots with `id`, `kind`, `file`, `duration_s`, `title`, `caption`, `notes`; plus `highlight` window and `quality_flags` copied from `tracking.json`. |
| `demo/demo_preview.mp4` | Offline concatenation of all shots — the fallback demo. |
| `demo/veed_request.json` | Serialised OpenEdit request (dry run; nothing is sent anywhere). |
| `demo/veed_composition.html` | Starting `.wv`-style HTML composition for the agent. |

Shot order: title → synced 2x2 grid (with per-camera audio offsets in the tile
labels) → "calibrate → track → fuse" card → 2x2 grid of per-camera tracking
overlays → fused pitch map → **quality card (confidence flags, cross-camera
disagreement, stats)** → "Stage 2/3" hand-off card.

The quality card is deliberately not optional: it renders
`quality.sync_low_confidence`, `quality.calibration_low_confidence`,
`quality.cross_camera_disagreement_m` and the first warnings, so the demo video
says the same thing as `tracking.json`.

## Not done / open

* No real VEED render has been produced (Linux-only environment). The stub
  `submit_to_veed(dry_run=False)` raises `NotImplementedError` on purpose.
* Transitions and animated titles in the offline preview are hard cuts; the
  polish is expected from OpenEdit's motion-graphics path.
* Voice-over / music: out of scope; `openedit-cli mix-audio` is the intended
  route if we add a bed.
