import os
import sys

# Ensure the project root (the parent of the "tests" folder) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from typing import Any, Dict, List
from fastapi.testclient import TestClient

import models
import export
import db
from embeddings import EMBEDDING_DIM
from app import app
from auth import current_active_user


# --- Mock Data and Helpers ---

MOCK_RUN_ID = "RUN_01"
MOCK_USER_ID = "user-for-export-test"


class MockUser:
    """A simple mock user class to satisfy the auth dependency."""
    def __init__(self, user_id: str):
        self.id = user_id


class MockRun:
    """A simple mock run class to satisfy the db.get_run dependency."""
    def __init__(self, run_id: str, user_id: str):
        self.run_id = run_id
        self.user_id = user_id


def get_valid_raw_embedding() -> Dict[str, Any]:
    """Returns a single valid raw embedding record for testing."""
    return {
        "feature_id": "feat1",
        "sequence": "PEPTIDE",
        "intensity": 123.45,
        "length": 7,
        "charge": 2,
        "hydrophobicity": 0.42,
        "embedding": [0.1] * EMBEDDING_DIM,
    }


# --- Unit Tests for normalize_to_omics_token_v1 ---


def test_normalize_to_omics_token_v1_happy_path():
    """
    Tests that a valid raw embedding is correctly normalized.
    """
    # Arrange
    raw_embeddings = [get_valid_raw_embedding()]

    # Act
    result = export.normalize_to_omics_token_v1(MOCK_RUN_ID, raw_embeddings)

    # Assert
    assert len(result) == 1
    record = result[0]
    assert isinstance(record, models.OmicsTokenV1)
    assert record.feature_id == "feat1"
    assert record.sequence == "PEPTIDE"
    assert record.properties.length == 7
    assert record.properties.charge == 2
    assert record.properties.hydrophobicity == pytest.approx(0.42)
    assert len(record.embedding) == EMBEDDING_DIM


def test_normalize_to_omics_token_v1_skips_missing_sequence():
    """
    Tests that records with a missing sequence are skipped.
    """
    # Arrange
    raw_embeddings = [{
        "feature_id": "feat_no_seq",
        "sequence": None,
        "embedding": [0.1] * EMBEDDING_DIM,
    }]

    # Act
    result = export.normalize_to_omics_token_v1(MOCK_RUN_ID, raw_embeddings)

    # Assert
    assert len(result) == 0


def test_normalize_to_omics_token_v1_skips_wrong_embedding_length():
    """
    Tests that records with an incorrect embedding dimension are skipped.
    """
    # Arrange
    raw_embeddings = [{
        "feature_id": "feat_bad_emb",
        "sequence": "PEPTIDE",
        "embedding": [0.1] * (EMBEDDING_DIM - 1),
    }]

    # Act
    result = export.normalize_to_omics_token_v1(MOCK_RUN_ID, raw_embeddings)

    # Assert
    assert len(result) == 0


def test_normalize_to_omics_token_v1_multiple_records_mixed_validity():
    """
    Tests that the function correctly filters a list containing both valid and invalid records.
    """
    # Arrange
    raw_embeddings = [
        {
            "feature_id": "feat_no_seq",
            "sequence": None,
            "embedding": [0.1] * EMBEDDING_DIM,
        },
        get_valid_raw_embedding(),
        {
            "feature_id": "feat_bad_emb",
            "sequence": "BADSEQ",
            "embedding": [0.1],
        },
    ]

    # Act
    result = export.normalize_to_omics_token_v1(MOCK_RUN_ID, raw_embeddings)

    # Assert
    assert len(result) == 1
    assert result[0].feature_id == "feat1"


# --- Integration Test for /export/embeddings/{run_id} ---


client = TestClient(app)


def test_export_embeddings_v1_endpoint(monkeypatch):
    """
    Tests the full export endpoint using mocks for auth and DB calls.
    """

    # Arrange: Mock dependencies
    monkeypatch.setattr(
        db,
        "get_run",
        lambda con, run_id: MockRun(run_id, MOCK_USER_ID),
    )
    monkeypatch.setattr(
        db,
        "get_peptide_embeddings",
        lambda con, run_id: [get_valid_raw_embedding()],
    )

    # Override the authentication dependency to inject a mock user
    app.dependency_overrides[current_active_user] = lambda: MockUser(MOCK_USER_ID)

    # Act
    response = client.get(f"/export/embeddings/{MOCK_RUN_ID}")

    # Assert
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == MOCK_RUN_ID
    assert payload["export_version"] == "omics_export_v1"
    assert payload["total_embeddings"] == 1
    assert len(payload["data"]) == 1
    assert payload["data"][0]["feature_id"] == "feat1"
    assert len(payload["data"][0]["embedding"]) == EMBEDDING_DIM

    # Clean up the override after the test
    app.dependency_overrides.clear()
