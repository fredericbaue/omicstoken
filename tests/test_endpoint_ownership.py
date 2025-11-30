import pytest
from fastapi.testclient import TestClient

import app
import db
from auth import current_active_user


class MockUser:
    def __init__(self, user_id: str):
        self.id = user_id


class MockRun:
    def __init__(self, run_id: str, user_id: str):
        self.run_id = run_id
        self.user_id = user_id


client = TestClient(app.app)


def test_export_requires_ownership(monkeypatch):
    run_id = "run-owned"
    owner_id = "user-1"
    other_id = "user-2"

    # Owned case
    monkeypatch.setattr(db, "get_run", lambda con, rid: MockRun(rid, owner_id))
    monkeypatch.setattr(db, "get_peptide_embeddings", lambda con, rid: [{"feature_id": "f1", "sequence": "PEP", "embedding": [0.0]*320, "intensity": 1.0, "length": 3, "charge": 0, "hydrophobicity": 0.0}])
    app.app.dependency_overrides[current_active_user] = lambda: MockUser(owner_id)
    resp = client.get(f"/export/embeddings/{run_id}")
    assert resp.status_code != 404  # authorized path should pass ownership check

    # Unauthorized case
    app.app.dependency_overrides[current_active_user] = lambda: MockUser(other_id)
    resp2 = client.get(f"/export/embeddings/{run_id}")
    assert resp2.status_code == 404

    app.app.dependency_overrides.clear()


def test_embed_requires_ownership(monkeypatch):
    run_id = "run-owned"
    owner_id = "user-1"
    other_id = "user-2"

    def fake_get_run(con, rid):
        return MockRun(rid, owner_id)

    monkeypatch.setattr(db, "get_run", fake_get_run)
    monkeypatch.setattr(db, "get_features_for_run", lambda con, rid: [])
    monkeypatch.setattr(db, "insert_peptide_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(db, "update_user_credits", lambda *args, **kwargs: None)

    app.app.dependency_overrides[current_active_user] = lambda: MockUser(other_id)
    resp = client.post(f"/peptide/embed/{run_id}")
    assert resp.status_code == 404
    app.app.dependency_overrides.clear()
