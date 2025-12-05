import sqlite3
import time

import config
import db
from worker import protein_structure_task

RUN_ID = "horizon_verify_run"
FEATURE_ID = "horizon_verify_feat"


def _get_connection() -> sqlite3.Connection:
    path = db.get_db_path(config.DATA_DIR)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def _seed_dummy_rows():
    con = _get_connection()
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO runs(
                run_id, user_id, instrument, method, polarity,
                schema_version, meta_json, n_features_to_embed, n_features_embedded
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (RUN_ID, None, "TestInstrument", "TestMethod", "positive", "horizon-verify", "{}", 0, 0),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO features(
                run_id, feature_id, mz, rt_sec, intensity, adduct,
                polarity, annotation_name, annotation_score, meta_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (RUN_ID, FEATURE_ID, 0.0, 0.0, 1000.0, "[M+H]+", "positive", "PEPTIDESEQ", 1.0, "{}"),
        )
        con.commit()
    finally:
        con.close()


def _cleanup_rows():
    con = _get_connection()
    try:
        con.execute(
            "DELETE FROM protein_structures WHERE run_id=? AND feature_id=?",
            (RUN_ID, FEATURE_ID),
        )
        con.execute(
            "DELETE FROM features WHERE run_id=? AND feature_id=?",
            (RUN_ID, FEATURE_ID),
        )
        con.execute("DELETE FROM runs WHERE run_id=?", (RUN_ID,))
        con.commit()
    finally:
        con.close()


def _poll_for_result(timeout_sec: int = 10) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        con = _get_connection()
        try:
            cur = con.execute(
                "SELECT run_id, feature_id, plddt_score FROM protein_structures WHERE run_id=? AND feature_id=?",
                (RUN_ID, FEATURE_ID),
            )
            row = cur.fetchone()
        finally:
            con.close()
        if row:
            pdb_id = f"{row[0]}:{row[1]}"
            print("SUCCESS: Row found!")
            print(f"  pdb_id={pdb_id}, plddt_score={row[2]}")
            return True
        time.sleep(1)
    print("FAILURE: Timed out waiting for DB write.")
    return False


def main():
    _cleanup_rows()
    _seed_dummy_rows()
    job = protein_structure_task.delay(run_id=RUN_ID, feature_id=FEATURE_ID)
    print("Task triggered... waiting for worker.")
    _poll_for_result()
    _cleanup_rows()
    if job:
        print(f"Celery task id: {job.id}")


if __name__ == "__main__":
    main()
