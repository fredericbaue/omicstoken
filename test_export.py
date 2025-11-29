import unittest
import sqlite3
import json
import os
from fastapi.testclient import TestClient

from app import app, DATA_DIR
import auth
from auth import User, get_user_db
import db
from embeddings import EMBEDDING_DIM
import export

# --- Mock Data and Helpers ---

MOCK_USER = User(id="test-user-id", email="test@example.com", is_active=True, is_verified=True, is_superuser=False)
TEST_RUN_ID = "test-export-run"
_test_db_connection = None # Module-level variable to hold the connection

def get_mock_user():
    """Dependency override function to return a mock user."""
    return MOCK_USER

def get_test_db_connection_override(data_dir: str = ""):
    """Override for db.get_db_connection to return the test-specific, in-memory DB."""
    return _test_db_connection

class TestExportEndpoint(unittest.TestCase):

    def setUp(self):
        """Set up a temporary in-memory database and a TestClient for each test."""
        global _test_db_connection
        _test_db_connection = sqlite3.connect(":memory:")
        db._create_tables(_test_db_connection)
        
        # Insert mock user and run
        _test_db_connection.execute("INSERT INTO runs (run_id, user_id) VALUES (?, ?)", (TEST_RUN_ID, MOCK_USER.id))

        # Insert one valid peptide embedding
        valid_embedding = [1.0] * EMBEDDING_DIM
        _test_db_connection.execute(
            """INSERT INTO peptide_embeddings 
               (run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (TEST_RUN_ID, MOCK_USER.id, "valid-feature-1", "TESTPEPTIDE", 12345.6, 11, 1, -0.5, json.dumps(valid_embedding))
        )
        
        # Insert one malformed peptide embedding (wrong dimension)
        malformed_embedding = [1.0] * (EMBEDDING_DIM - 1)
        _test_db_connection.execute(
            """INSERT INTO peptide_embeddings
               (run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (TEST_RUN_ID, MOCK_USER.id, "malformed-feature-1", "BADPEPTIDE", 6789.0, 10, 0, 0.2, json.dumps(malformed_embedding))
        )
        _test_db_connection.commit()

        # Override dependencies for this test class
        app.dependency_overrides[auth.current_active_user] = get_mock_user
        app.dependency_overrides[db.get_db_connection] = get_test_db_connection_override
        
        self.client = TestClient(app)

    def tearDown(self):
        """Close the database connection and clear dependency overrides after each test."""
        global _test_db_connection
        if _test_db_connection:
            _test_db_connection.close()
        _test_db_connection = None
        app.dependency_overrides = {}

    def test_export_embeddings_success_and_validation(self):
        """
        Test the successful export of embeddings for a valid run.
        Ensures that malformed records are filtered out.
        """
        # Act
        response = self.client.get(f"/export/embeddings/{TEST_RUN_ID}")
        
        # Assert
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        
        # 1. Check top-level structure
        self.assertEqual(data["run_id"], TEST_RUN_ID)
        self.assertEqual(data["export_version"], "omics_export_v1")
        self.assertEqual(data["total_embeddings"], 1) # Malformed record should be filtered
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)
        self.assertEqual(len(data["data"]), 1)

        # 2. Check the canonical record itself
        record = data["data"][0]
        self.assertEqual(record["run_id"], TEST_RUN_ID)
        self.assertEqual(record["feature_id"], "valid-feature-1")
        self.assertEqual(record["sequence"], "TESTPEPTIDE")
        self.assertEqual(len(record["embedding"]), EMBEDDING_DIM)
        
        # 3. Check calculated properties
        props = record["properties"]
        expected_mw = export.calculate_molecular_weight("TESTPEPTIDE")
        self.assertAlmostEqual(props["molecular_weight"], expected_mw, places=2)
        self.assertEqual(props["length"], 11)

    def test_export_embeddings_run_not_found(self):
        """
        Test that a 404 error is returned for a non-existent run_id.
        """
        # Act
        response = self.client.get("/export/embeddings/non-existent-run")
        
        # Assert
        self.assertEqual(response.status_code, 404)
        self.assertIn("not found", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
