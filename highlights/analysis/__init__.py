"""Post-pipeline "Analyse match" stage for multi-angle projects.

Runs once tracking/sync are done: a slow YOLO pass over the reference
angle clusters player shirt colours into two teams, then possession /
shots / goals / player-effort stats are derived on the shared-T time
axis and summarised into a short deterministic text. Outputs land in
<project>/analysis/{teams,match_stats,summary}.json plus status/log.
"""
