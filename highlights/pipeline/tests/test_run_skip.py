import json

from highlights.pipeline import run


def _make_project(tmp_path):
    proj = tmp_path / "proj"
    pipe = proj / "pipeline"
    (pipe / "audio").mkdir(parents=True)
    (pipe / "motion").mkdir(parents=True)
    (proj / "match.mp4").write_bytes(b"fake")
    (pipe / "probe.json").write_text('{"duration_s": 1200.0}')
    for f in ["features_1s.parquet", "match_window.json",
              "scores.parquet", "candidates.json", "stats.json"]:
        (pipe / f).write_text("{}")
    (pipe / "audio" / "audio.wav").write_bytes(b"")
    (pipe / "audio" / "features_1s.json").write_text("{}")
    (pipe / "audio" / "whistles.json").write_text("{}")
    (pipe / "motion" / "features_1s.json").write_text("{}")
    return proj


def _counter(calls, name):
    def fn(ctx):
        calls.append(name)
    return fn


def test_all_outputs_present_skips_everything(tmp_path, monkeypatch):
    proj = _make_project(tmp_path)
    calls = []
    orig = {n: s.fn for n, s in run.STAGES.items()}
    for n in run.STAGES:
        run.STAGES[n].fn = _counter(calls, n)
    try:
        rc = run.main(["--project-dir", str(proj),
                       "--video", str(proj / "match.mp4")])
    finally:
        for n, fn in orig.items():
            run.STAGES[n].fn = fn
    assert rc == 0
    assert calls == []
    st = json.loads((proj / "pipeline" / "status.json").read_text())
    assert st["state"] == "done"


def test_force_reruns_all(tmp_path, monkeypatch):
    proj = _make_project(tmp_path)
    calls = []
    orig = {n: s.fn for n, s in run.STAGES.items()}
    for n in run.STAGES:
        run.STAGES[n].fn = _counter(calls, n)
    try:
        rc = run.main(["--project-dir", str(proj),
                       "--video", str(proj / "match.mp4"), "--force"])
    finally:
        for n, fn in orig.items():
            run.STAGES[n].fn = fn
    assert rc == 0
    assert calls == list(run.STAGES)


def test_stages_subset_only(tmp_path, monkeypatch):
    proj = _make_project(tmp_path)
    calls = []
    orig = {n: s.fn for n, s in run.STAGES.items()}
    for n in run.STAGES:
        run.STAGES[n].fn = _counter(calls, n)
    try:
        rc = run.main(["--project-dir", str(proj),
                       "--video", str(proj / "match.mp4"),
                       "--force", "--stages", "stats,candidates"])
    finally:
        for n, fn in orig.items():
            run.STAGES[n].fn = fn
    assert rc == 0
    assert calls == ["candidates", "stats"]  # pipeline order, not arg order
