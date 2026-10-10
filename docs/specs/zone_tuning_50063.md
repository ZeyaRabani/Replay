# Zone tuning — 50063dcf6552 zone recut (director.json, no reprocessing)

Source: `/data/projects/50063dcf6552/multiangle/director.json` (snapshot `20260926-134154`, style "Fast + zones", zones painted on all 3 angles, match window 1660–5560 shared-T → output span 3899 s ≈ 65 min).

## Top line
- **240 cuts → 241 segments**, span 3899 s
- Cut median **9.0 s**, mean **16.2 s**
- Switches (angle changes): **240** = **3.7/min** — effectively every cut boundary is an angle switch
- **Ping-pong**: 41 segments (17%) are ≤3 s inserts between two segments of the *same* angle (A→B→A); 36 of those are zone→X→zone
- Time share per angle: **a0 13.7% / a1 68.2% / a2 18.1%**

## Cut-length distribution
| Length | Count | Share |
|---|---|---|
| <2 s   | 3   | 1.2% |
| 2–4 s  | 51  | 21.2% |
| 4–8 s  | 42  | 17.4% |
| >8 s   | 145 | 60.2% |

## Rule usage (per-second ratios)
- zone 85.1%, event 13.9%, cluster 0.9%, ball 0.1%, hold 0.0%
- Segment rule counts: zone 196, event 39, cluster 5, start 1
- Zone-seg lengths: median 9 s, mean 17.2 s

## Zone internals (director.json fields)
- `zone_ball_share` 6.0% — share of zone-eligible seconds selected via ball-in-zone
- `zone_players_share` 80.4% — via player density in zone
- `zone_suspended_share` per angle: **[0.26%, 0%, 0%]** — angle 0 alone had any suspended (viewcheck-failed) seconds

## Reading for tuning
- The 2–4 s bucket (51 segs, 21%) plus 41 ping-pong flips is the churn the user is complaining about: roughly one short flip per 90 s of output.
- Angle 1 dominates (68%); a0 is nearly viewcheck-suppressed (suspended) yet still takes 13.7%.
- zone→X→zone ping-pongs (36) are the cheapest win — a minimum-hold / hysteresis on zone selection would remove most of them without touching detection.

## Tuned (offline rerun, same inputs — hysteresis: ball hold 2 s / density hold 4 s + 2 s confirm + 6 s no-return)

| Metric | Before | After |
|---|---|---|
| Cuts (segments) | 240 (241) | 212 (213) |
| Median / mean cut | 9.0 s / 16.2 s | 11.0 s / 18.3 s |
| <2 s / 2–4 s / 4–8 s / >8 s | 3 / 51 / 42 / 145 | 5 / 17 / 49 / 142 |
| Switches per minute | 3.69 | 3.26 |
| Ping-pong (≤3 s A→B→A) | 41 | **11** (−73%) |
| zone→X→zone ping-pong | 36 | **8** (−78%) |
| Zone share | 85.1% | 85.1% |
| Angle share a0/a1/a2 | 13.7/68.2/18.1 | 14.5/66.2/19.2 |

Reproduced the production director.json exactly with the old code path first (240 cuts, identical stats) — the harness at `/home/ubuntu/hl4/tune50063/compare.py` replicates `stage_director`'s input assembly on fetched project data. New code passes the ≥60% ping-pong reduction bar. New outputs: `project/director_{old,new}.json`.
