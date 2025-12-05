import scripts.seed_demo_data as seed
import db
import app
import models


def test_seed_runs_for_user(monkeypatch):
    calls = {"runs": [], "features": [], "embed": []}

    def fake_get_run(con, rid):
        return None

    def fake_insert_run(con, rid, meta, user_id=None):
        calls["runs"].append((rid, meta.get("demo", False)))

    def fake_insert_feature(con, rid, feat):
        calls["features"].append((rid, feat.feature_id, feat.peptide_sequence))

    class Conn:
        def close(self): ...
        def commit(self): ...
        def commit(self): ...

    monkeypatch.setattr(db, "get_db_connection", lambda _: Conn())
    monkeypatch.setattr(db, "get_run", fake_get_run)
    monkeypatch.setattr(db, "insert_run", fake_insert_run)
    monkeypatch.setattr(db, "insert_feature", fake_insert_feature)
    monkeypatch.setattr(db, "get_peptide_embeddings", lambda con, rid: [])
    monkeypatch.setattr(app, "_embed_run", lambda rid, expected_user_id=None: calls["embed"].append(rid))

    seed.seed_runs_for_user("demo-user-id")

    assert calls["runs"], "runs should be inserted"
    assert calls["embed"], "embedding should be triggered"
    assert all(flag for _, flag in calls["runs"]), "demo flag should be true on all runs"
