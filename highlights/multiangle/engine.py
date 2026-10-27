"""Shared inference-engine selection: NCNN exports when the ncnn wheel
is importable and REPLAY_INFER_ENGINE != "torch", else torch .pt."""
import os
from pathlib import Path


def ncnn_ok() -> bool:
    if os.environ.get("REPLAY_INFER_ENGINE") == "torch":
        return False
    try:
        import ncnn  # noqa: F401
        return True
    except ImportError:
        return False


def prefer_ncnn(path: Path) -> Path:
    """Prefer the sibling <stem>_ncnn_model export dir of a .pt path when
    it exists and ncnn is usable; a direct *_ncnn_model path passes
    through unchanged."""
    if path.name.endswith("_ncnn_model") and path.exists():
        return path
    sib = path.with_name(f"{path.stem}_ncnn_model")
    if sib.exists() and ncnn_ok():
        return sib
    return path
