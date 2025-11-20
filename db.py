import sqlite3
import os
import json
from typing import List, Dict, Any, Optional, Tuple

import models

# --- Configuration ---
SCHEMA_VERSION = "immuno-0.1.0"

def get_db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "immuno.sqlite")

def get_db_connection(data_dir: str) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database and ensures tables exist."""
    db_path = get_db_path(data_dir)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON") # Enable foreign key constraints
    _create_tables(con)
    return con

def _create_tables(con: sqlite3.Connection):
    """Creates database tables if they do not already exist."""
    con.execute("""CREATE TABLE IF NOT EXISTS runs(
        run_id TEXT PRIMARY KEY,
        instrument TEXT,
        method TEXT,
        polarity TEXT,
        schema_version TEXT,
        meta_json TEXT
    )""")
    
    con.execute("""CREATE TABLE IF NOT EXISTS features(
        run_id TEXT,
        feature_id TEXT,
        mz REAL,
        rt_sec REAL,
        intensity REAL,
        adduct TEXT,
        polarity TEXT,
        annotation_name TEXT, -- Stores peptide sequence
        annotation_score REAL,
        meta_json TEXT,
        PRIMARY KEY(run_id, feature_id),
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    )""")
    
    # Migration: Attempt to add meta_json column if it doesn't exist
    try:
        con.execute("ALTER TABLE features ADD COLUMN meta_json TEXT")
    except sqlite3.OperationalError:
        pass # Column likely already exists

    con.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        run_id TEXT,
        feature_id TEXT,
        method TEXT,
        polarity TEXT,
        vec_json TEXT,
        PRIMARY KEY(run_id, feature_id),
        FOREIGN KEY(run_id, feature_id) REFERENCES features(run_id, feature_id) ON DELETE CASCADE
    )""")
    con.commit()

# --- Helper functions for DB operations ---

def insert_run(con: sqlite3.Connection, run_id: str, meta_dict: Dict[str, Any]):
    """Inserts or replaces a run's metadata into the database."""
    con.execute(
        "INSERT OR REPLACE INTO runs(run_id, instrument, method, polarity, schema_version, meta_json) VALUES(?,?,?,?,?,?)",
        (run_id, meta_dict.get("instrument"), meta_dict.get("method"), meta_dict.get("polarity"), SCHEMA_VERSION, json.dumps(meta_dict)),
    )

def insert_feature(con: sqlite3.Connection, run_id: str, feature_data: models.Feature):
    """Inserts or replaces a feature (peptide) into the database."""
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
            feature_data.peptide_sequence, # Map peptide_sequence to annotation_name DB column
            feature_data.annotation_score,
            json.dumps(feature_data.metadata)
        ),
    )

def insert_embedding(con: sqlite3.Connection, run_id: str, feature_id: str, method: str, polarity: str, vector: List[float]):
    """Inserts or replaces a peptide embedding into the database."""
    con.execute(
        """INSERT OR REPLACE INTO embeddings(run_id, feature_id, method, polarity, vec_json)
           VALUES(?,?,?,?,?)""",
        (run_id, feature_id, method, polarity, json.dumps(vector.tolist())),
    )

def get_run(con: sqlite3.Connection, run_id: str) -> Optional[models.Run]:
    """Retrieves a run's metadata."""
    cur = con.cursor()
    cur.execute("SELECT run_id, instrument, method, polarity, schema_version, meta_json FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    if row:
        return models.Run(run_id=row[0], instrument=row[1], method=row[2], polarity=row[3], schema_version=row[4], meta=json.loads(row[5]))
    return None

def get_features_for_run(con: sqlite3.Connection, run_id: str) -> List[models.Feature]:
    """Retrieves all features (peptides) for a given run."""
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
    """Retrieves a specific peptide embedding."""
    cur = con.cursor()
    cur.execute("SELECT vec_json FROM embeddings WHERE run_id=? AND feature_id=?", (run_id, feature_id))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

def get_feature_properties(con: sqlite3.Connection, run_id: str, feature_id: str) -> Optional[Tuple[str, float]]:
    """Retrieves the peptide sequence and intensity for a given feature."""
    cur = con.cursor()
    cur.execute("SELECT annotation_name, intensity FROM features WHERE run_id=? AND feature_id=? LIMIT 1", (run_id, feature_id))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None

def get_all_embeddings_data(con: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    """Retrieves all embedding data (run_id, feature_id, vec_json) from the database."""
    cur = con.cursor()
    cur.execute("SELECT run_id, feature_id, vec_json FROM embeddings")
    return cur.fetchall()

def get_run_summaries(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Retrieves a summary of all runs with feature and embedding counts."""
    cur = con.cursor()
    cur.execute("""
        SELECT r.run_id, r.instrument, r.method, r.polarity, r.meta_json,
               COUNT(DISTINCT f.feature_id) as n_features,
               COUNT(DISTINCT e.feature_id) as n_embeddings
        FROM runs r
        LEFT JOIN features f ON r.run_id = f.run_id
        LEFT JOIN embeddings e ON r.run_id = e.run_id AND f.feature_id = e.feature_id
        GROUP BY r.run_id
    """)
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