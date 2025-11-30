import json
import threading
import time
from typing import Dict

import faiss

import db
import search
from embeddings import EMBEDDING_DIM


def _seed_embedding(data_dir: str, run_id: str, feature_id: str):
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


def test_rebuild_serialized_with_lock(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _seed_embedding(str(data_dir), "run-1", "feat-1")

    original_write = faiss.write_index
    state: Dict[str, int] = {"active": 0, "max_active": 0}

    def slow_write(index, path):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.1)
        original_write(index, path)
        state["active"] -= 1

    monkeypatch.setattr(faiss, "write_index", slow_write)

    t1 = threading.Thread(target=search.rebuild_faiss_index, args=(None, str(data_dir)))
    t2 = threading.Thread(target=search.rebuild_faiss_index, args=(None, str(data_dir)))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Ensure no overlapping writes occurred due to the lock
    assert state["max_active"] == 1

    # Final index/id map should exist and be readable
    index_path = data_dir / "peptides.faiss"
    ids_path = data_dir / "peptides_ids.json"
    assert index_path.exists()
    assert ids_path.exists()
    ids = json.loads(ids_path.read_text())
    assert ids == [["run-1", "feat-1"]]
