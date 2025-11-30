import io
import zipfile
import json
import pytest
from fastapi.testclient import TestClient

import app
import db
import export
from auth import current_active_user


client = TestClient(app.app)


class MockUser:
    def __init__(self, user_id: str):
        self.id = user_id


class MockRun:
    def __init__(self, run_id: str, user_id: str):
        self.run_id = run_id
        self.user_id = user_id


def test_export_bundle_unauthorized():
    resp = client.post("/export/bundle", json={"run_ids": ["r1"]})
    assert resp.status_code == 401


def test_export_bundle_success(monkeypatch):
    run_ids = ["r1", "r2"]

    def mock_get_run(con, run_id):
        if run_id in run_ids:
            return MockRun(run_id, "user-1")
        return None

    def mock_get_embeddings(con, run_id):
        return [
            {
                "feature_id": "f1",
                "sequence": "PEPTIDE",
                "intensity": 1.0,
                "length": 7,
                "charge": 1,
                "hydrophobicity": 0.1,
                "embedding": [0.1] * 320,
            }
        ]

    monkeypatch.setattr(db, "get_run", mock_get_run)
    monkeypatch.setattr(db, "get_peptide_embeddings", mock_get_embeddings)

    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.post("/export/bundle", json={"run_ids": run_ids})
    assert resp.status_code == 200

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    assert "bundle.json" in names
    for rid in run_ids:
        assert f"{rid}_omics_export_v1.json" in names
    manifest = json.loads(zf.read("bundle.json"))
    assert manifest["run_count"] == 2
    assert manifest["run_ids"] == run_ids

    app.app.dependency_overrides.clear()


def test_export_bundle_incomplete(monkeypatch):
    def mock_get_run(con, run_id):
        return MockRun(run_id, "user-1")

    def mock_get_embeddings(con, run_id):
        return []  # incomplete

    monkeypatch.setattr(db, "get_run", mock_get_run)
    monkeypatch.setattr(db, "get_peptide_embeddings", mock_get_embeddings)

    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.post("/export/bundle", json={"run_ids": ["r1"]})
    assert resp.status_code == 400
    assert "incomplete" in resp.json().get("detail", "").lower()

    app.app.dependency_overrides.clear()
