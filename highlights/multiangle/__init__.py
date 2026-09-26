"""Multi-angle full-match assembly ("Option 2").

Angles are per-camera sub-projects run through the existing Option-1 pipeline
(probe..candidates); this package then syncs them by audio cross-correlation,
extracts per-angle tracking features, picks a broadcast-style director cut,
renders it to match.mp4, and fuses candidates across angles.

Design spec: /home/ubuntu/hl4/MULTIANGLE_SPEC.md
"""
