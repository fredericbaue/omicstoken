import json
import pytest
from fastapi.testclient import TestClient

import app
import auth
import db
from auth import current_active_user


client = TestClient(app.app)


class MockUser:
    def __init__(self, user_id):
        self.id = user_id


def test_create_api_token(monkeypatch):
    # Mock creation to avoid touching real DB
    monkeypatch.setattr(auth, "create_api_token_for_user", lambda user_id, label=None: {
        "token": "plain-token",
        "token_id": 123,
        "label": label,
        "created_at": auth.datetime.utcnow(),
    })
    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.post("/auth/api-tokens", json={"label": "lab"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["token"] == "plain-token"
    assert data["token_id"] == 123

    app.app.dependency_overrides.clear()


def test_programmatic_upload_with_token(monkeypatch):
    # Allow token auth to resolve
    monkeypatch.setattr(auth, "_get_user_from_api_token", lambda tok: MockUser("user-1") if tok == "apitoken" else None)
    monkeypatch.setattr(db, "insert_run", lambda con, run_id, meta_dict, user_id=None: None)
    monkeypatch.setattr(db, "insert_feature", lambda con, run_id, feature_data: None)
    monkeypatch.setattr(app, "_embed_run", lambda run_id, expected_user_id=None: None)

    payload = {
        "name": "Run A",
        "features": [
            {
                "feature_id": "f1",
                "sequence": "PEPTIDE",
                "intensity": 1.0,
                "length": 7,
                "charge": 1,
                "hydrophobicity": 0.1
            }
        ],
        "auto_embed": True
    }
    resp = client.post("/api/runs", json=payload, headers={"Authorization": "Bearer apitoken"})
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["auto_embed"] is True


def test_programmatic_upload_requires_features(monkeypatch):
    monkeypatch.setattr(auth, "_get_user_from_api_token", lambda tok: MockUser("user-1"))
    resp = client.post("/api/runs", json={"features": []}, headers={"Authorization": "Bearer apitoken"})
    assert resp.status_code == 400
    assert "Features are required" in resp.json().get("detail", "")
