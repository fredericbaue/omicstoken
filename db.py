import sqlite3
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import models
import config
import logging

# --- Configuration ---
SCHEMA_VERSION = "immuno-0.2.0" # Bumped for Horizon 2

DB_BACKEND = config.DB_BACKEND
DATABASE_URL = (config.DATABASE_URL or "").strip()
USERS_DATABASE_URL = os.getenv("USERS_DATABASE_URL", "").strip()
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "data")

def get_db_path(data_dir: str) -> str:
    """Resolve the primary database location for the active backend (SQLite default)."""
    if DB_BACKEND != "sqlite":
        raise ValueError(f"DB_BACKEND '{DB_BACKEND}' is not supported yet. Set DB_BACKEND=sqlite until Postgres is implemented.")
    return os.path.join(data_dir, "immuno.sqlite")

def get_users_db_path(data_dir: Optional[str] = None) -> str:
    """Resolve the user/credits database path."""
    if USERS_DATABASE_URL:
        return USERS_DATABASE_URL
    base_dir = data_dir or DEFAULT_DATA_DIR
    return os.path.join(base_dir, "users.db")

def get_db_connection(data_dir: str) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database and ensures tables exist."""
    db_path = get_db_path(data_dir)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    _create_tables(con)
    return con

def _create_tables(con: sqlite3.Connection):
    """Creates database tables if they do not already exist."""
    con.execute("""CREATE TABLE IF NOT EXISTS model_versions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        version TEXT NOT NULL,
        embedding_dim INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(name, version, embedding_dim)
    )""")

    con.execute("""CREATE TABLE IF NOT EXISTS runs(
        run_id TEXT PRIMARY KEY,
        user_id TEXT,
        instrument TEXT,
        method TEXT,
        polarity TEXT,
        schema_version TEXT,
        meta_json TEXT,
        n_features_to_embed INTEGER,
        n_features_embedded INTEGER
    )""")
    
    con.execute("""CREATE TABLE IF NOT EXISTS features(
        run_id TEXT,
        feature_id TEXT,
        mz REAL,
        rt_sec REAL,
        intensity REAL,
        adduct TEXT,
        polarity TEXT,
        annotation_name TEXT,
        annotation_score REAL,
        meta_json TEXT,
        PRIMARY KEY(run_id, feature_id),
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )""")

    # Migration: Attempt to add meta_json column if it doesn't exist
    try:
        con.execute("ALTER TABLE features ADD COLUMN meta_json TEXT")
    except sqlite3.OperationalError:
        pass

    con.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        run_id TEXT,
        feature_id TEXT,
        method TEXT,
        polarity TEXT,
        vec_json TEXT,
        PRIMARY KEY(run_id, feature_id),
        FOREIGN KEY(run_id, feature_id) REFERENCES features(run_id, feature_id) ON DELETE CASCADE
    )""")

    con.execute("""CREATE TABLE IF NOT EXISTS peptide_embeddings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        user_id TEXT,
        feature_id TEXT,
        sequence TEXT,
        intensity REAL NOT NULL,
        length INTEGER,
        charge INTEGER,
        hydrophobicity REAL,
        embedding TEXT,
        model_version TEXT DEFAULT 'v2_gemini_768',
        model_version_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(run_id, feature_id),
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
        FOREIGN KEY(run_id, feature_id) REFERENCES features(run_id, feature_id) ON DELETE CASCADE,
        FOREIGN KEY(model_version_id) REFERENCES model_versions(id)
    )""")
    
    # Schema Migration Helpers
    try:
        con.execute("ALTER TABLE peptide_embeddings ADD COLUMN model_version TEXT DEFAULT 'v2_gemini_768'")
    except sqlite3.OperationalError: pass
    try:
        con.execute("ALTER TABLE peptide_embeddings ADD COLUMN model_version_id INTEGER")
    except sqlite3.OperationalError: pass

    # --- HORIZON 2: Protein Structures Table ---
    con.execute("""CREATE TABLE IF NOT EXISTS protein_structures(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        feature_id TEXT,
        sequence TEXT,
        pdb_content TEXT,      -- The actual PDB file content (or path)
        plddt_score REAL,      -- AlphaFold confidence score (0-100)
        pae_json TEXT,         -- Predicted Aligned Error (JSON)
        engine TEXT,           -- 'alphafold_multimer', 'esmfold', 'mock'
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(run_id, feature_id),
        FOREIGN KEY(run_id, feature_id) REFERENCES features(run_id, feature_id) ON DELETE CASCADE
    )""")

    con.commit()

# --- Helper functions for DB operations ---

def get_or_create_model_version(con: sqlite3.Connection, name: str, version: str, embedding_dim: int) -> int:
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM model_versions WHERE name=? AND version=? AND embedding_dim=? LIMIT 1",
        (name, version, embedding_dim),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        "INSERT INTO model_versions(name, version, embedding_dim) VALUES(?,?,?)",
        (name, version, embedding_dim),
    )
    return int(cur.lastrowid)

def insert_run(con: sqlite3.Connection, run_id: str, meta_dict: Dict[str, Any], user_id: Optional[str] = None):
    con.execute(
        "INSERT OR REPLACE INTO runs(run_id, user_id, instrument, method, polarity, schema_version, meta_json) VALUES(?,?,?,?,?,?,?)",
        (run_id, user_id, meta_dict.get("instrument"), meta_dict.get("method"), meta_dict.get("polarity"), SCHEMA_VERSION, json.dumps(meta_dict)),
    )

def insert_feature(con: sqlite3.Connection, run_id: str, feature_data: models.Feature):
    con.execute(
        """INSERT OR REPLACE INTO features(run_id, feature_id, mz, rt_sec, intensity, adduct, polarity, annotation_name, annotation_score, meta_json)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            feature_data.feature_id,
            feature_data.mz,
            feature_data.rt_sec,
            feature_data.intensity,
            feature_data.adduct,
            feature_data.polarity,
            feature_data.peptide_sequence,
            feature_data.annotation_score,
            json.dumps(feature_data.metadata)
        ),
    )

def insert_embedding(con: sqlite3.Connection, run_id: str, feature_id: str, method: str, polarity: str, vector: List[float]):
    con.execute(
        """INSERT OR REPLACE INTO embeddings(run_id, feature_id, method, polarity, vec_json)
           VALUES(?,?,?,?,?)""",
        (run_id, feature_id, method, polarity, json.dumps(vector.tolist())),
    )

def insert_peptide_embedding(
    con: sqlite3.Connection,
    run_id: str,
    user_id: str,
    feature_id: str,
    sequence: str,
    intensity: float, 
    length: int,
    charge: int,
    hydrophobicity: float,
    vector: np.ndarray,
    model_version: str = config.EMBEDDING_MODEL_NAME,
    model_version_id: Optional[int] = None,
):
    vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
    con.execute(
        """INSERT INTO peptide_embeddings(run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, embedding, model_version, model_version_id)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id,
            user_id,
            feature_id,
            sequence,
            intensity,
            length,
            charge,
            hydrophobicity,
            json.dumps(vector_list),
            model_version,
            model_version_id,
        ),
    )

# --- NEW: Horizon 2 Helper ---
def insert_protein_structure(
    con: sqlite3.Connection,
    run_id: str,
    feature_id: str,
    sequence: str,
    pdb_content: str,
    plddt_score: float,
    engine: str = "mock"
):
    """
    Inserts a predicted structure (PDB) into the DB.
    """
    con.execute(
        """INSERT OR REPLACE INTO protein_structures(run_id, feature_id, sequence, pdb_content, plddt_score, engine)
           VALUES(?,?,?,?,?,?)""",
        (run_id, feature_id, sequence, pdb_content, plddt_score, engine)
    )

def update_run_meta(con: sqlite3.Connection, run_id: str, updates: Dict[str, Any]) -> None:
    cur = con.cursor()
    cur.execute("SELECT meta_json FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    existing_meta: Dict[str, Any] = {}
    if row and row[0]:
        try:
            existing_meta = json.loads(row[0]) or {}
        except Exception as e:
            logging.warning("Failed to parse meta_json for run %s; overwriting with updates only (%s)", run_id, e)
            existing_meta = {}
    merged = {**existing_meta, **(updates or {})}
    cur.execute("UPDATE runs SET meta_json=? WHERE run_id=?", (json.dumps(merged), run_id))
    con.commit()

def get_run(con: sqlite3.Connection, run_id: str) -> Optional[models.Run]:
    cur = con.cursor()
    cur.execute("SELECT run_id, user_id, instrument, method, polarity, schema_version, meta_json FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    if row:
        return models.Run(
            run_id=row[0],
            user_id=row[1],
            instrument=row[2],
            method=row[3],
            polarity=row[4],
            schema_version=row[5],
            meta=json.loads(row[6]) if row[6] else {}
        )
    return None

def get_features_for_run(con: sqlite3.Connection, run_id: str) -> List[models.Feature]:
    cur = con.cursor()
    cur.execute("SELECT feature_id, mz, rt_sec, intensity, adduct, polarity, annotation_name, annotation_score, meta_json FROM features WHERE run_id=?", (run_id,))
    results = []
    for row in cur.fetchall():
        meta = json.loads(row[8]) if row[8] else {}
        results.append(models.Feature(
            feature_id=row[0], 
            mz=row[1], 
            rt_sec=row[2], 
            intensity=row[3], 
            adduct=row[4], 
            polarity=row[5], 
            peptide_sequence=row[6], 
            annotation_score=row[7],
            metadata=meta
        ))
    return results

def get_embedding(con: sqlite3.Connection, run_id: str, feature_id: str) -> Optional[List[float]]:
    cur = con.cursor()
    cur.execute("SELECT embedding FROM peptide_embeddings WHERE run_id=? AND feature_id=?", (run_id, feature_id))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None


def get_feature_properties(con: sqlite3.Connection, run_id: str, feature_id: str) -> Optional[Tuple[str, float]]:
    cur = con.cursor()
    cur.execute("SELECT annotation_name, intensity FROM features WHERE run_id=? AND feature_id=? LIMIT 1", (run_id, feature_id))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None

def get_all_embeddings_data(con: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    cur = con.cursor()
    cur.execute("SELECT run_id, feature_id, embedding FROM peptide_embeddings")
    return cur.fetchall()

def get_peptide_embeddings(con: sqlite3.Connection, run_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    cur.execute("""
        SELECT feature_id, sequence, intensity, length, charge, hydrophobicity, embedding 
        FROM peptide_embeddings 
        WHERE run_id=?
    """, (run_id,))
    results = []
    for row in cur.fetchall():
        results.append({
            "feature_id": row[0],
            "sequence": row[1],
            "intensity": row[2],
            "length": row[3],
            "charge": row[4],
            "hydrophobicity": row[5],
            "embedding": json.loads(row[6]) if row[6] else []
        })
    return results

def get_run_summaries(con: sqlite3.Connection, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    cur = con.cursor()
    base_query = """
        SELECT
            r.run_id,
            r.instrument,
            r.method,
            r.polarity,
            r.meta_json,
            (SELECT COUNT(*) FROM features f WHERE f.run_id = r.run_id) as n_features,
            (SELECT COUNT(*) FROM peptide_embeddings pe WHERE pe.run_id = r.run_id) as n_embeddings
        FROM runs r
    """
    params = []
    if user_id:
        base_query += " WHERE r.user_id = ?"
        params.append(user_id)
    cur.execute(base_query, tuple(params))
    results = []
    for row in cur.fetchall():
        meta = json.loads(row[4]) if row[4] else {}
        results.append({
            "run_id": row[0],
            "instrument": row[1],
            "method": row[2],
            "polarity": row[3],
            "original_filename": meta.get("original_filename"),
            "n_features": row[5],
            "n_embeddings": row[6]
        })
    return results

def update_user_credits(con: sqlite3.Connection, user_id: str, amount: int):
    try:
        with sqlite3.connect(get_users_db_path()) as u_con:
            u_con.execute("UPDATE user SET credits = credits + ? WHERE id = ?", (amount, user_id))
            u_con.commit()
    except Exception as e:
        print(f"Error updating credits for user {user_id}: {e}")

def get_user_credits(con: sqlite3.Connection, user_id: str) -> int:
    try:
        with sqlite3.connect(get_users_db_path()) as u_con:
            cur = u_con.cursor()
            cur.execute("SELECT credits FROM user WHERE id = ?", (user_id,))
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception as e:
        print(f"Error fetching credits for user {user_id}: {e}")
        return 0
