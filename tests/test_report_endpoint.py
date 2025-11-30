import pytest
from fastapi.testclient import TestClient
import app
import db
from auth import current_active_user

client = TestClient(app.app)


class MockUser:
    def __init__(self, user_id):
        self.id = user_id


class MockRun:
    def __init__(self, run_id, user_id):
        self.run_id = run_id
        self.user_id = user_id


def test_report_unauthorized():
    resp = client.get("/report/run1")
    assert resp.status_code == 401


def test_report_success(monkeypatch):
    def mock_get_run(con, rid):
        return MockRun(rid, "user-1")

    def mock_get_peptide_embeddings(con, rid):
        # 4 peptides to satisfy clustering
        base_embed = [0.1] * 320
        return [
            {"sequence": "AAA", "intensity": 1.0, "length": 3, "hydrophobicity": 0.1, "embedding": base_embed},
            {"sequence": "BBB", "intensity": 2.0, "length": 3, "hydrophobicity": 0.2, "embedding": base_embed},
            {"sequence": "CCC", "intensity": 3.0, "length": 3, "hydrophobicity": 0.3, "embedding": base_embed},
            {"sequence": "DDD", "intensity": 4.0, "length": 3, "hydrophobicity": 0.4, "embedding": base_embed},
        ]

    monkeypatch.setattr(db, "get_run", mock_get_run)
    monkeypatch.setattr(db, "get_peptide_embeddings", mock_get_peptide_embeddings)
    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.get("/report/run1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == "run1"
    assert data["total_peptides"] == 4
    assert data["total_clusters"] >= 1
    assert data["largest_cluster"] is not None
    assert "plain_language_summary" in data

    app.app.dependency_overrides.clear()


def test_report_incomplete(monkeypatch):
    def mock_get_run(con, rid):
        return MockRun(rid, "user-1")

    def mock_get_peptide_embeddings(con, rid):
        return []  # insufficient

    monkeypatch.setattr(db, "get_run", mock_get_run)
    monkeypatch.setattr(db, "get_peptide_embeddings", mock_get_peptide_embeddings)
    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.get("/report/run1")
    assert resp.status_code == 400

    app.app.dependency_overrides.clear()


def test_report_wrong_owner(monkeypatch):
    def mock_get_run(con, rid):
        return MockRun(rid, "other-user")

    def mock_get_peptide_embeddings(con, rid):
        base_embed = [0.1] * 320
        return [
            {"sequence": "A", "intensity": 1.0, "length": 1, "hydrophobicity": 0.1, "embedding": base_embed}
        ] * 4

    monkeypatch.setattr(db, "get_run", mock_get_run)
    monkeypatch.setattr(db, "get_peptide_embeddings", mock_get_peptide_embeddings)
    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.get("/report/run1")
    assert resp.status_code == 404

    app.app.dependency_overrides.clear()
