import unittest
import sqlite3
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app as fastapi_app
import auth
import db
from embeddings import EMBEDDING_DIM
import export

# --- Mock user and helpers ---

MOCK_USER = auth.User(
    id="test-user-id",
    email="test@example.com",
    is_active=True,
    is_verified=True,
    is_superuser=False,
)

TEST_RUN_ID = "test-export-run"


def get_mock_user():
    """Dependency override to behave like a logged-in user."""
    return MOCK_USER


class TestExportEndpoint(unittest.TestCase):
    def setUp(self):
        """
        Set up an in-memory SQLite DB that is safe to use from FastAPI's
        threadpool, and patch db.get_db_connection so the export endpoint
        uses this DB.
        """
        # IMPORTANT: allow this connection to be used across threads
        self.test_db = sqlite3.connect(":memory:", check_same_thread=False)
        db._create_tables(self.test_db)

        # Insert a run owned by MOCK_USER with required fields populated
        self.test_db.execute(
            """
            INSERT INTO runs (
                run_id,
                user_id,
                instrument,
                method,
                polarity,
                schema_version,
                meta_json,
                n_features_to_embed,
                n_features_embedded
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                TEST_RUN_ID,
                MOCK_USER.id,
                None,
                None,
                None,
                db.SCHEMA_VERSION,
                "{}",
                None,
                None,
            ),
        )

        # Insert one valid peptide embedding (correct dimension)
        valid_embedding = [1.0] * EMBEDDING_DIM
        self.test_db.execute(
            """
            INSERT INTO peptide_embeddings
                (run_id, user_id, feature_id, sequence,
                 intensity, length, charge, hydrophobicity, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                TEST_RUN_ID,
                MOCK_USER.id,
                "valid-feature-1",
                "TESTPEPTIDE",
                12345.6,
                11,
                1,
                -0.5,
                json.dumps(valid_embedding),
            ),
        )

        # Insert one malformed peptide embedding (wrong dimension)
        malformed_embedding = [1.0] * (EMBEDDING_DIM - 1)
        self.test_db.execute(
            """
            INSERT INTO peptide_embeddings
                (run_id, user_id, feature_id, sequence,
                 intensity, length, charge, hydrophobicity, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                TEST_RUN_ID,
                MOCK_USER.id,
                "malformed-feature-1",
                "BADPEPTIDE",
                6789.0,
                10,
                0,
                0.2,
                json.dumps(malformed_embedding),
            ),
        )

        self.test_db.commit()

        # Override Auth: treat every request as MOCK_USER
        fastapi_app.dependency_overrides[auth.current_active_user] = get_mock_user

        # Patch db.get_db_connection so export_embeddings_v1 uses our in-memory DB
        self.get_db_patch = patch("db.get_db_connection", return_value=self.test_db)
        self.get_db_patch.start()

        # Create TestClient AFTER overrides/patches
        self.client = TestClient(fastapi_app)

    def tearDown(self):
        # Stop patch and clean up
        self.get_db_patch.stop()
        self.test_db.close()
        fastapi_app.dependency_overrides.clear()

    def test_export_embeddings_success_and_validation(self):
        """
        Ensure the /export/embeddings/{run_id} endpoint:
        - Returns 200
        - Uses the canonical OmicsToken v1 shape
        - Filters out malformed embeddings
        - Computes basic properties as expected
        """
        response = self.client.get(f"/export/embeddings/{TEST_RUN_ID}")
        self.assertEqual(response.status_code, 200)

        data = response.json()

        # Top-level structure
        self.assertEqual(data["run_id"], TEST_RUN_ID)
        self.assertEqual(data["export_version"], "omics_export_v1")
        self.assertEqual(data["total_embeddings"], 1)  # malformed row filtered
        self.assertIn("data", data)
        self.assertEqual(len(data["data"]), 1)

        # Record structure
        record = data["data"][0]
        self.assertEqual(record["feature_id"], "valid-feature-1")
        self.assertEqual(record["sequence"], "TESTPEPTIDE")
        self.assertEqual(len(record["embedding"]), EMBEDDING_DIM)

        # Calculated properties
        props = record["properties"]
        expected_mw = export.calculate_molecular_weight("TESTPEPTIDE")
        self.assertAlmostEqual(props["molecular_weight"], expected_mw, places=2)
        self.assertEqual(props["length"], 11)

    def test_export_embeddings_run_not_found(self):
        """
        Non-existent run_id should return 404.
        """
        response = self.client.get("/export/embeddings/non-existent-run")
        self.assertEqual(response.status_code, 404)
        detail = response.json().get("detail", "").lower()
        self.assertIn("not found", detail)


if __name__ == "__main__":
    unittest.main()
