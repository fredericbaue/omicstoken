import json
from unittest.mock import MagicMock

import auth
import db
from app import app as fastapi_app
from fastapi.testclient import TestClient
from embeddings import EMBEDDING_DIM


def test_embedding_export_flow(monkeypatch):
    # --- Auth override: act as a logged-in user "u1" ---
    fastapi_app.dependency_overrides[auth.current_active_user] = lambda: auth.User(
        id="u1",
        email="test@test.com",
        is_active=True,
        is_verified=True,
        is_superuser=False,
    )

    # --- DB connection mock (we don't actually touch SQLite here) ---
    mock_con = MagicMock()
    monkeypatch.setattr(db, "get_db_connection", lambda *_args, **_kwargs: mock_con)

    # Mock run row so ownership checks pass
    class MockRun:
        def __init__(self):
            self.run_id = "TEST_EXPORT_RUN"
            self.user_id = "u1"

    monkeypatch.setattr(
        db,
        "get_run",
        lambda con, rid: MockRun() if rid == "TEST_EXPORT_RUN" else None,
    )

    # Mock embeddings: one valid record, correct dimension
    def fake_get_peptide_embeddings(con, rid):
        if rid != "TEST_EXPORT_RUN":
            return []
        return [
            {
                "feature_id": "f1",
                "sequence": "A",
                "intensity": 100.0,
                "length": 1,
                "charge": 1,
                "hydrophobicity": 0.1,
                "embedding": [0.1] * EMBEDDING_DIM,
            }
        ]

    monkeypatch.setattr(db, "get_peptide_embeddings", fake_get_peptide_embeddings)

    client = TestClient(fastapi_app)

    # --- Call export endpoint ---
    res = client.get("/export/embeddings/TEST_EXPORT_RUN")
    assert res.status_code == 200

    body = res.json()
    assert body["run_id"] == "TEST_EXPORT_RUN"
    assert body["total_embeddings"] == 1
    assert len(body["data"]) == 1
    record = body["data"][0]
    assert record["feature_id"] == "f1"
    assert len(record["embedding"]) == EMBEDDING_DIM

    # Clean up overrides
    fastapi_app.dependency_overrides.clear()
