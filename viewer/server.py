"""Replay viewer server: static files + server-side Reactor token exchange.

    REACTOR_API_KEY=rk_... python viewer/server.py   # or put it in a .env next to the repo root
    -> http://localhost:8080

Endpoints
    GET  /            static viewer (this directory)
    POST /token       {"model": "<slug>"} -> {"jwt": ...} using REACTOR_API_KEY (never sent to the browser)
    GET  /config      {"reactor": bool, "media": [...], "seed_url": ...}  tells the client whether Reactor is configured
                      (SEED_IMAGE_URL = public URL of the seed image; HappyOyster fetches it server-side)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN_URL = "https://api.reactor.inc/tokens"
DEFAULT_MODEL = "reactor/happy-oyster-adventure"


def load_env() -> None:
    for p in (HERE / ".env", HERE.parent / ".env"):
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


def api_key() -> str | None:
    return os.environ.get("REACTOR_API_KEY") or None


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        if "/media/" not in str(args[0] if args else ""):
            super().log_message(fmt, *args)

    def end_headers(self):  # dev server: always revalidate so edits show up on reload
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/config":
            media = sorted(p.name for p in (HERE / "media").glob("*") if p.is_file()) if (HERE / "media").is_dir() else []
            return self._json(200, {"reactor": api_key() is not None, "media": media,
                                    "seed_url": os.environ.get("SEED_IMAGE_URL") or None})
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/token":
            return self._json(404, {"error": "not found"})
        key = api_key()
        if not key:
            return self._json(503, {"error": "REACTOR_API_KEY not set on the server"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        model = body.get("model") or DEFAULT_MODEL
        payload = json.dumps({
            "authorization_details": [{
                "type": "session",
                "resources": {"models": {"match": [model]}},
                "constraints": {"max_sessions": 5, "max_session_duration_seconds": 3600},
            }]
        }).encode()
        req = urllib.request.Request(TOKEN_URL, data=payload, method="POST",
                                     headers={"Reactor-API-Key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return self._json(200, json.loads(r.read()))
        except urllib.error.HTTPError as e:
            return self._json(e.code, {"error": f"reactor /tokens {e.code}", "detail": e.read().decode(errors="replace")[:500]})
        except (urllib.error.URLError, TimeoutError) as e:
            return self._json(502, {"error": f"reactor /tokens unreachable: {e}"})


def main() -> None:
    load_env()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8080"))
    print(f"Replay viewer  http://localhost:{port}   reactor={'configured' if api_key() else 'NOT configured (Three.js fallback only)'}")
    ThreadingHTTPServer(("", port), partial(Handler, directory=str(HERE))).serve_forever()


if __name__ == "__main__":
    main()
