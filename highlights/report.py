"""Static HTML review page for highlight candidates."""

from __future__ import annotations

import html
from pathlib import Path


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_review_html(out_dir: Path, candidates, active_windows=None) -> Path:
    win_html = ""
    if active_windows:
        spans = ", ".join(f"{_mmss(a)}-{_mmss(b)}" for a, b in active_windows)
        win_html = f'<p class="win">active play: {html.escape(spans)}</p>'
    cards = []
    for c in candidates:
        bars = "".join(
            f'<div class="sig"><span>{html.escape(k)}</span>'
            f'<div class="bar"><div class="fill" style="width:{int(v * 100)}%"></div></div></div>'
            for k, v in c.signals.items())
        vid = (f'<video controls preload="metadata" src="clips/{html.escape(c.clip)}"></video>'
               if c.clip else '<div class="novid">no clip</div>')
        cards.append(
            f'<div class="card {c.type}"><div class="hdr"><b>#{c.rank}</b> '
            f'<span class="type">{c.type}</span> conf={c.confidence:.2f} '
            f'<span class="t">{_mmss(c.t_event)}</span> goal {c.goal or "?"} '
            f'<span class="anch">{c.anchor}</span></div>{vid}'
            f'<div class="sigs">{bars}</div></div>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>highlights review</title>
<style>
body{{font-family:system-ui,sans-serif;background:#14171c;color:#e8e8e8;max-width:960px;margin:24px auto;padding:0 16px}}
.card{{background:#1e232b;border:1px solid #2c3440;border-radius:10px;padding:14px;margin:14px 0}}
.card.goal{{border-color:#3f9d4d}} .card.chance{{border-color:#8a6d2f}}
.hdr{{font-size:15px;margin-bottom:8px}} .type{{text-transform:uppercase;font-size:12px;padding:1px 6px;border-radius:4px;background:#2c3440}}
.goal .type{{background:#3f9d4d;color:#fff}} .chance .type{{background:#8a6d2f;color:#fff}}
.t{{font-variant-numeric:tabular-nums;color:#9fb4c8}}
video{{width:100%;border-radius:8px;background:#000}}
.novid{{color:#667;font-style:italic}}
.sigs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:6px;margin-top:8px}}
.sig span{{font-size:11px;color:#9fb4c8}} .bar{{height:6px;background:#2c3440;border-radius:3px}}
.fill{{height:100%;background:#4da3ff;border-radius:3px}}
.win{{font-size:13px;color:#9fb4c8}}
.anch{{font-size:11px;padding:1px 6px;border-radius:4px;background:#37404e;color:#9fb4c8}}
</style></head><body><h1>Highlight candidates</h1>{win_html}{''.join(cards) or '<p>No candidates.</p>'}</body></html>"""
    out = Path(out_dir) / "review.html"
    out.write_text(page)
    return out
