import pytest
from fastapi.testclient import TestClient
import app
import auth
import db
from auth import current_active_user, current_active_user_or_token

client = TestClient(app.app)


class MockUser:
    def __init__(self, user_id):
        self.id = user_id


def test_create_api_token(monkeypatch):
    # FIX: Make this async because the endpoint awaits it
    async def mock_create(user_id, label=None):
        return {
            "token": "plain-token",
            "token_id": 123,
            "label": label,
            "created_at": auth.datetime.utcnow(),
        }
    monkeypatch.setattr(auth, "create_api_token_for_user", mock_create)
    
    app.app.dependency_overrides[current_active_user] = lambda: MockUser("user-1")

    resp = client.post("/auth/api-tokens", json={"label": "lab"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["token"] == "plain-token"

    app.app.dependency_overrides.clear()

def test_programmatic_upload_with_token(monkeypatch):
    # FIX: Override the actual dependency used by the route
    app.app.dependency_overrides[current_active_user_or_token] = lambda: MockUser("user-1")
    
    monkeypatch.setattr(db, "insert_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(db, "insert_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(app, "_embed_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(db, "update_user_credits", lambda *args, **kwargs: None)

    payload = {
        "name": "Run A",
        "features": [{"feature_id": "f1", "sequence": "PEP", "intensity": 1.0}],
        "auto_embed": True
    }
    resp = client.post("/api/runs", json=payload, headers={"Authorization": "Bearer apitoken"})
    assert resp.status_code == 200
    assert resp.json()["run_id"].startswith("RUN_")

    app.app.dependency_overrides.clear()

def test_programmatic_upload_requires_features(monkeypatch):
    app.app.dependency_overrides[current_active_user_or_token] = lambda: MockUser("user-1")
    resp = client.post("/api/runs", json={"features": []})
    assert resp.status_code == 400
    app.app.dependency_overrides.clear()
