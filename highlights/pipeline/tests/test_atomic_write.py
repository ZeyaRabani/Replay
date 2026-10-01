"""Atomic write helpers must not clobber hardlinked sibling files."""

import json
import os

from highlights.io import write_json_atomic, write_text_atomic


def test_write_json_atomic_preserves_hardlinked_sibling(tmp_path):
    a = tmp_path / "a" / "director.json"
    b = tmp_path / "b" / "director.json"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text(json.dumps({"style": "normal", "n_cuts": 105}))
    os.link(a, b)
    assert a.stat().st_ino == b.stat().st_ino

    write_json_atomic(a, {"style": "fast", "n_cuts": 196})

    assert json.loads(a.read_text()) == {"style": "fast", "n_cuts": 196}
    assert json.loads(b.read_text()) == {"style": "normal", "n_cuts": 105}
    assert a.stat().st_ino != b.stat().st_ino
    assert not (tmp_path / "a" / "director.json.tmp").exists()


def test_write_text_atomic(tmp_path):
    f = tmp_path / "x.json"
    write_text_atomic(f, "hello")
    write_text_atomic(f, "world")
    assert f.read_text() == "world"
