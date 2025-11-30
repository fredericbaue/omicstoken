import json
import os
import sys

import pytest
import faiss

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import db
import search
from embeddings import EMBEDDING_DIM


def _seed_embedding(data_dir: str, run_id: str = "run-1", feature_id: str = "feat-1"):
    """Create minimal run, feature, and embedding rows for testing."""
    con = db.get_db_connection(data_dir)
    cur = con.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO runs(run_id, user_id, instrument, method, polarity, schema_version, meta_json, n_features_to_embed, n_features_embedded) VALUES(?,?,?,?,?,?,?,?,?)",
        (run_id, "user-1", None, None, None, "immuno-0.1.0", "{}", None, None),
    )
    cur.execute(
        "INSERT OR REPLACE INTO features(run_id, feature_id, mz, rt_sec, intensity, adduct, polarity, annotation_name, annotation_score, meta_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (run_id, feature_id, 0.0, 0.0, 1.0, None, None, "PEPTIDE", None, "{}"),
    )
    db.insert_peptide_embedding(
        con=con,
        run_id=run_id,
        user_id="user-1",
        feature_id=feature_id,
        sequence="PEPTIDE",
        intensity=1.0,
        length=7,
        charge=1,
        hydrophobicity=0.1,
        vector=[0.1] * EMBEDDING_DIM,
    )
    con.commit()
    con.close()


def test_rebuild_faiss_index_happy_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_embedding(str(data_dir))

    count = search.rebuild_faiss_index(None, str(data_dir))

    assert count == 1
    index_path = data_dir / "peptides.faiss"
    ids_path = data_dir / "peptides_ids.json"
    assert index_path.exists()
    assert ids_path.exists()
    with open(ids_path, "r") as f:
        ids = json.load(f)
    assert ids == [["run-1", "feat-1"]]


def test_rebuild_faiss_index_failure_leaves_prior_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_embedding(str(data_dir))

    # Pre-create prior index artifacts to ensure they are left intact on failure.
    index_path = data_dir / "peptides.faiss"
    ids_path = data_dir / "peptides_ids.json"
    index_path.write_text("old-index")
    ids_path.write_text("old-ids")

    def boom(*args, **kwargs):
        raise OSError("write failure")

    monkeypatch.setattr(faiss, "write_index", boom)

    count = search.rebuild_faiss_index(None, str(data_dir))

    assert count == 0
    # Prior artifacts remain unchanged; temp files are cleaned up.
    assert index_path.read_text() == "old-index"
    assert ids_path.read_text() == "old-ids"
    assert not (data_dir / "peptides.faiss.tmp").exists()
    assert not (data_dir / "peptides_ids.json.tmp").exists()
