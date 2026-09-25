---
name: replay-live-testing
description: Browser testing of deployed Replay profiles, media, upload handoff and feedback export.
---

# Deployed Replay UI testing

## Access and safety

Use the deployment URL supplied by the user. Live testing needs no local services.
Profiles use display names, not passwords. Origin-specific local storage means
Vercel and direct-backend login are separate. Use existing owners only for
reversible review changes; create/delete testing projects under a QA profile.
Record original candidate numeric trims/statuses and restore them at the end.
Do not start multi-angle assembly or override sync offsets without explicit
authorization: those actions can launch long computations.

## Devin Secrets Needed

None for the name-only profile UI. Do not enter YouTube cookies unless separately
authorized; cookie-free server downloads can fail as expected.

## Media and Stats

Review chooses source.mp4 until a proxy is ready. Read video.currentSrc,
videoWidth/videoHeight/duration as supplemental evidence, then prove moving
frames through visible playback. If a cold deep seek stalls, test fresh playback
from zero and repeat seek; compare direct backend origin without mutating jobs.
Report the initial stall even when retry recovers.

Older projects may lack Match stats; use a newer completed project when requested.
Peak minute and director timeline clicks switch to Review and seek. Check actual
playhead/time, not only tab switching. Check territory bounds0–100 and sums.
The Director tab displays angle metadata; individual media endpoints are:
`/api/projects/{id}/multiangle/angle/{index}/video?user={owner}`.

## Metadata and uploads

Pitch and Camera selectors are shared across single/multi forms. A tagged
YouTube test can validate persistent metadata/chips even when ingestion is
bot-blocked. Without saved cookies the failed card offers cookie setup rather
than Retry, plus Upload the file instead.

The Vercel Upload tab should offer a direct configured backend URL instead of
a file picker. Follow the link and select Upload there to verify the picker;
this proves routing, not successful large-file transfer/absence of413 under load.

## API evidence

For explicit feedback API tests, use same-origin browser fetch with the requested
X-User header; do not extract browser cookies. Check HTTP status, content type,
parse every NDJSON line, compare default versus all=1, and record owner/status
sets. Export may aggregate owners; confirm intended scope rather than assuming
the profile header restricts results.
