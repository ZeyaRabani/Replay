import subprocess
from unittest.mock import patch

import pytest

from highlights.multiangle import trackfeat


def _res(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


def test_probe_dims_retries_empty_stdout_then_valid():
    calls = [
        _res(rc=0, out=""),              # transient empty stdout
        _res(rc=0, out="1920,1080\n"),   # valid on retry
    ]
    with patch.object(trackfeat.subprocess, "run", side_effect=calls) as m, \
         patch.object(trackfeat.time, "sleep") as s:
        w, h = trackfeat._probe_dims("v.mp4", tries=3, delay=0.01)
    assert (w, h) == (1920, 1080)
    assert m.call_count == 2
    assert s.call_count == 1


def test_probe_dims_retries_unparseable_then_valid():
    calls = [
        _res(rc=0, out="1920,\n"),       # non-empty but broken fields
        _res(rc=0, out="1920,1080\n"),
    ]
    with patch.object(trackfeat.subprocess, "run", side_effect=calls) as m, \
         patch.object(trackfeat.time, "sleep"):
        w, h = trackfeat._probe_dims("v.mp4", tries=3, delay=0.01)
    assert (w, h) == (1920, 1080)
    assert m.call_count == 2


def test_probe_dims_retries_nonzero_rc_then_valid():
    calls = [
        _res(rc=1, out="", err="boom"),
        _res(rc=0, out="640,360\n"),
    ]
    with patch.object(trackfeat.subprocess, "run", side_effect=calls), \
         patch.object(trackfeat.time, "sleep"):
        w, h = trackfeat._probe_dims("v.mp4", tries=3, delay=0.01)
    assert (w, h) == (640, 360)


def test_probe_dims_raises_runtimeerror_with_stderr_after_all_fail():
    calls = [_res(rc=1, out="", err="moov atom not found")] * 3
    with patch.object(trackfeat.subprocess, "run", side_effect=calls) as m, \
         patch.object(trackfeat.time, "sleep") as s, \
         pytest.raises(RuntimeError, match="moov atom not found"):
        trackfeat._probe_dims("v.mp4", tries=3, delay=0.01)
    assert m.call_count == 3
    assert s.call_count == 2
