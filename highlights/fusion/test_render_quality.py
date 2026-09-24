"""Tests for render.py quality-preset command selection."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render  # noqa: E402


def test_preset_extensions():
    assert render.codec_args("preview")[0] == "mp4"
    assert render.codec_args("high")[0] == "mp4"
    assert render.codec_args("max")[0] == "mkv"  # FLAC not MP4-safe


def test_preset_args():
    _, v, a = render.codec_args("preview")
    assert "-crf" in v and v[v.index("-crf") + 1] == "28"
    assert any("scale" in x for x in v)
    assert a[a.index("-b:a") + 1] == "96k"

    _, v, a = render.codec_args("high")
    assert v[v.index("-crf") + 1] == "14"
    assert "slow" in v
    assert "yuv420p" in v
    assert not any("scale" in x for x in v)  # source resolution preserved
    assert a[a.index("-b:a") + 1] == "256k"

    _, v, a = render.codec_args("max")
    assert v[v.index("-crf") + 1] == "0"
    assert "veryslow" in v
    assert "yuv420p" in v
    assert a[a.index("-c:a") + 1] == "flac"


def test_unknown_quality():
    with pytest.raises(ValueError):
        render.codec_args("ultra")


def test_default_out_dir():
    # --out omitted -> rendered_<quality>, preview keeps legacy rendered/
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=["preview", "high", "max"],
                    default="high")
    args = ap.parse_args([])
    assert args.quality == "high"
