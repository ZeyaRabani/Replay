"""Atomic file-write helpers.

os.replace() gives the destination a fresh inode, so when project files are
hardlinked between sibling projects (e.g. a shared import), rewriting an
output in one project can never clobber the sibling's copy.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _via_tmp(path: str | Path, write: Callable[[Path], None]) -> None:
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_json_atomic(path: str | Path, obj: Any, **kw: Any) -> None:
    """json.dumps -> tmp file in the same dir -> os.replace."""
    _via_tmp(path, lambda t: t.write_text(json.dumps(obj, **kw)))


def write_text_atomic(path: str | Path, text: str) -> None:
    _via_tmp(path, lambda t: t.write_text(text))


def write_parquet_atomic(df: Any, path: str | Path) -> None:
    _via_tmp(path, lambda t: df.to_parquet(t, index=False))
