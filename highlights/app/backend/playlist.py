"""Per-user YouTube playlist tracker.

Fetches a saved playlist URL via ``yt-dlp --flat-playlist -J`` in a
background thread, caches the grouped result to
``{workdir}/users/<user>/playlist.json``, and answers:

- which matches (grouped by leading dd/mm/yy in the title) exist,
- how many angle uploads each has, whether an "All Angles" upload exists,
- whether any video is already a project/archive source in Replay.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path

from .history import download_counts, video_id
from .store import workdir

FETCH_TIMEOUT = 300
_refreshing: set[str] = set()
_lock = threading.Lock()

_DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*[-–—.:]?\s*(.*)$")  # noqa: RUF001
# trailing angle markers: "… All Angles", "… Green end right", "… Orange end"
_ANGLE_TAIL_RE = re.compile(
    r"\s*[-–—.:]?\s*(all angles|[a-z]+ end(?:\s+\w+)*)\s*$", re.IGNORECASE)  # noqa: RUF001


def _user_dir(user: str) -> Path:
    return workdir() / "users" / user


def _cache_path(user: str) -> Path:
    return _user_dir(user) / "playlist.json"


def parse_title(title: str) -> tuple[str | None, str | None, str]:
    """-> (date 'yyyy-mm-dd' | raw-date-string | None, label, angle-tail?).

    Garbage dates like 15/20/21 group under the raw text instead of a
    normalised date."""
    m = _DATE_RE.match(title or "")
    if not m:
        return None, None, (title or "").strip()
    d, mo, y, rest = m.groups()
    day, month, year = int(d), int(mo), int(y)
    if year < 100:
        year += 2000
    if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
        date = f"{year:04d}-{month:02d}-{day:02d}"
    else:
        date = f"{d}/{mo}/{y}"
    return date, None, rest.strip()


def group_videos(entries: list[dict]) -> list[dict]:
    """yt-dlp flat-playlist entries -> per-match groups, newest first."""
    groups: dict[str, dict] = {}
    for e in entries or []:
        title = str(e.get("title") or "")
        date, _unused, rest = parse_title(title)
        key = date or "no-date"
        label = _ANGLE_TAIL_RE.sub("", rest).strip(" .-–—") or rest  # noqa: RUF001
        g = groups.setdefault(key, {"date": date, "label": label,
                                    "videos": []})
        if len(label) < len(g["label"]):          # shortest label wins
            g["label"] = label
        vid = e.get("id") or video_id(e.get("url"))
        g["videos"].append({
            "id": vid,
            "title": title,
            "duration": e.get("duration"),
            "url": e.get("url") or (f"https://www.youtube.com/watch?v={vid}"
                                    if vid else None),
        })
    out = list(groups.values())
    for g in out:
        g["n_angles"] = sum(
            1 for v in g["videos"]
            if "all angles" not in v["title"].lower())
        g["all_angles_uploaded"] = any(
            "all angles" in v["title"].lower() for v in g["videos"])
        g["done"] = g["all_angles_uploaded"]

    # newest normalised dates first; unparseable/raw groups last
    out.sort(key=lambda g: (
        0 if g["date"] and "-" in str(g["date"]) else 1,
        _desc_date(g["date"]),
        str(g["date"] or ""),
    ))
    return out


def _desc_date(d) -> str:
    """Sort helper: invert a yyyy-mm-dd string so newest sorts first."""
    if d and "-" in str(d):
        y, m, day = str(d).split("-")
        return f"{9999 - int(y):04d}-{99 - int(m):02d}-{99 - int(day):02d}"
    return "9999-99-99"


def _replay_vids(user: str) -> set[str]:
    """YouTube ids the user already has in Replay (live + archived)."""
    return set(download_counts(user).keys())


def fetch_playlist(url: str, cookies: str | None = None) -> list[dict]:
    cmd = ["yt-dlp", "--flat-playlist", "-J", url]
    if cookies:
        cmd[1:1] = ["--cookies", cookies]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=FETCH_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp exited {proc.returncode}: "
                           f"{proc.stderr.strip()[:400]}")
    data = json.loads(proc.stdout)
    return data.get("entries") or []


def get_state(user: str, reg) -> dict:
    url = (reg.get_user(user) or {}).get("playlist_url")
    st = {"url": url, "fetched_at": None, "matches": [], "refreshing": False}
    cp = _cache_path(user)
    if cp.is_file():
        try:
            cached = json.loads(cp.read_text())
            st["fetched_at"] = cached.get("fetched_at")
            st["matches"] = cached.get("matches") or []
        except Exception:
            pass
    with _lock:
        st["refreshing"] = user in _refreshing
    return st


def set_url(user: str, url: str, reg) -> dict:
    reg.update_user(user, playlist_url=url)
    refresh_async(user, reg)
    return get_state(user, reg)


def refresh_async(user: str, reg) -> None:
    with _lock:
        if user in _refreshing:
            return
        _refreshing.add(user)
    threading.Thread(target=_refresh, args=(user, reg), daemon=True).start()


def _refresh(user: str, reg) -> None:
    try:
        url = (reg.get_user(user) or {}).get("playlist_url")
        if not url:
            return
        from .main import _user_cookies_path  # late import, avoids a cycle
        ck = _user_cookies_path(user)
        cookies = str(ck) if ck.is_file() else None
        matches = group_videos(fetch_playlist(url, cookies))
        vids = _replay_vids(user)
        for m in matches:
            m["in_replay"] = any(v["id"] and v["id"] in vids
                                 for v in m["videos"])
        cp = _cache_path(user)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(
            {"fetched_at": time.time(), "matches": matches}, indent=1))
    except Exception:
        pass  # last good cache stays; refreshing flag clears either way
    finally:
        with _lock:
            _refreshing.discard(user)
