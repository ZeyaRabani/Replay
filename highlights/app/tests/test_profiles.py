"""Profile migration (demo/john -> admin, drop throwaway users) and the
project-title PATCH endpoint."""

import json

from conftest import new_project, scoped

from highlights.app.backend.store import Registry


def _mk(root):
    (root / "projects").mkdir(parents=True, exist_ok=True)
    return root


def test_users_merge_and_drop(tmp_path):
    root = _mk(tmp_path)
    users = [{"name": n, "created_at": 1.0}
             for n in ("demo", "john", "zeya", "watch", "qa", "testing")]
    (root / "users.json").write_text(json.dumps({"users": users}))
    reg = Registry(root)
    # john owns a project -> its owner should be rewritten to admin
    p = reg.create_project(owner="john", title="x", source={})
    reg2 = Registry(root)   # fresh load re-runs the idempotent migration
    names = {u["name"] for u in reg2.list_users()}
    assert "admin" in names
    assert not ({"demo", "john", "watch", "qa", "testing"} & names)
    assert reg2.get(p.id).owner == "admin"


def test_drop_user_kept_when_owning_project(tmp_path):
    root = _mk(tmp_path)
    reg = Registry(root)
    p = reg.create_project(owner="watch", title="x", source={})
    # watch arrives in users.json only after it already owns a project
    (root / "users.json").write_text(
        json.dumps({"users": [{"name": "watch", "created_at": 1.0}]}))
    reg2 = Registry(root)
    assert {u["name"] for u in reg2.list_users()} == {"watch"}
    assert reg2.get(p.id).owner == "watch"


def test_patch_title(client, sample_video):
    pid = new_project(client, sample_video)
    r = client.patch(scoped(pid, ""), json={"title": "  Semi final U12  "})
    assert r.status_code == 200
    assert r.json()["title"] == "Semi final U12"
    assert client.get(scoped(pid, "")).json()["title"] == "Semi final U12"
    for bad in ("", "   ", "x" * 121):
        assert client.patch(
            scoped(pid, ""), json={"title": bad}).status_code == 422
