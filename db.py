import sqlite3
import os
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
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

    con.execute("""CREATE TABLE IF NOT EXISTS peptide_embeddings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        user_id TEXT,
        feature_id TEXT, -- Added feature_id
        sequence TEXT,
        intensity REAL NOT NULL,
        length INTEGER,
        charge INTEGER,
        hydrophobicity REAL,
        embedding TEXT, -- JSON list of floats
        model_version TEXT DEFAULT 'v2_gemini_768', -- Track which model generated this embedding
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(run_id, feature_id), -- Ensure unique embedding per run and feature
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
        FOREIGN KEY(run_id, feature_id) REFERENCES features(run_id, feature_id) ON DELETE CASCADE
    )""")
    
    # Migration: Add model_version column if it doesn't exist
    try:
        con.execute("ALTER TABLE peptide_embeddings ADD COLUMN model_version TEXT DEFAULT 'v2_gemini_768'")
    except sqlite3.OperationalError:
        pass  # Column likely already exists
    
    con.commit()

# --- Helper functions for DB operations ---

def insert_run(con: sqlite3.Connection, run_id: str, meta_dict: Dict[str, Any], user_id: Optional[str] = None):
    """Inserts or replaces a run's metadata into the database."""
    con.execute(
        "INSERT OR REPLACE INTO runs(run_id, user_id, instrument, method, polarity, schema_version, meta_json) VALUES(?,?,?,?,?,?,?)",
        (run_id, user_id, meta_dict.get("instrument"), meta_dict.get("method"), meta_dict.get("polarity"), SCHEMA_VERSION, json.dumps(meta_dict)),
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
    """Inserts or replaces a peptide embedding into the database (Old table)."""
    con.execute(
        """INSERT OR REPLACE INTO embeddings(run_id, feature_id, method, polarity, vec_json)
           VALUES(?,?,?,?,?)""",
        (run_id, feature_id, method, polarity, json.dumps(vector.tolist())),
    )

def insert_peptide_embedding(con: sqlite3.Connection, run_id: str, user_id: str, feature_id: str, sequence: str, intensity: float, 
                             length: int, charge: int, hydrophobicity: float, vector: np.ndarray, model_version: str = "v2_gemini_768"):
    """Inserts a row into the new peptide_embeddings table."""
    con.execute(
        """INSERT INTO peptide_embeddings(run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, embedding, model_version)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, json.dumps(vector.tolist()), model_version)
    )

def get_run(con: sqlite3.Connection, run_id: str) -> Optional[models.Run]:
    """Retrieves a run's metadata."""
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

from typing import Optional, List
import json
import sqlite3

def get_embedding(con: sqlite3.Connection, run_id: str, feature_id: str) -> Optional[List[float]]:
    """Retrieves a specific peptide embedding from the new table."""
    cur = con.cursor()
    # FIXED: Changed table from 'embeddings' to 'peptide_embeddings'
    # FIXED: Changed column from 'vec_json' to 'embedding'
    cur.execute("SELECT embedding FROM peptide_embeddings WHERE run_id=? AND feature_id=?", (run_id, feature_id))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None


def get_feature_properties(con: sqlite3.Connection, run_id: str, feature_id: str) -> Optional[Tuple[str, float]]:
    """Retrieves the peptide sequence and intensity for a given feature."""
    cur = con.cursor()
    cur.execute("SELECT annotation_name, intensity FROM features WHERE run_id=? AND feature_id=? LIMIT 1", (run_id, feature_id))
    row = cur.fetchone()
    return (row[0], row[1]) if row else None

def get_all_embeddings_data(con: sqlite3.Connection) -> List[Tuple[str, str, str]]:
    """Retrieves all embedding data (run_id, feature_id, embedding) from the database."""
    cur = con.cursor()
    cur.execute("SELECT run_id, feature_id, embedding FROM peptide_embeddings")
    return cur.fetchall()

def get_peptide_embeddings(con: sqlite3.Connection, run_id: str) -> List[Dict[str, Any]]:
    """Retrieves all peptide embeddings for a specific run."""
    cur = con.cursor()
    cur.execute("""
        SELECT sequence, intensity, length, charge, hydrophobicity, embedding 
        FROM peptide_embeddings 
        WHERE run_id=?
    """, (run_id,))
    
    results = []
    for row in cur.fetchall():
        results.append({
            "sequence": row[0],
            "intensity": row[1],
            "length": row[2],
            "charge": row[3],
            "hydrophobicity": row[4],
            "embedding": json.loads(row[5]) if row[5] else []
        })
    return results

def get_run_summaries(con: sqlite3.Connection, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrievels a summary of all runs with feature and embedding counts. Optionally filtered by user_id."""
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
        FROM
            runs r
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
    """
    Updates the credit balance for a specific user in the separate users.db.
    Note: This function establishes its own connection to users.db.
    """
    try:
        # Connect directly to the user database
        with sqlite3.connect("data/users.db") as u_con:
            u_con.execute("UPDATE user SET credits = credits + ? WHERE id = ?", (amount, user_id))
            u_con.commit()
    except Exception as e:
        # Using logging would be better, but print is fine for now.
        print(f"Error updating credits for user {user_id}: {e}")

def get_user_credits(con: sqlite3.Connection, user_id: str) -> int:
    """
    Fetches the current credit balance for a user from the separate users.db.
    Note: This function establishes its own connection to users.db.
    """
    try:
        with sqlite3.connect("data/users.db") as u_con:
            cur = u_con.cursor()
            cur.execute("SELECT credits FROM user WHERE id = ?", (user_id,))
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception as e:
        print(f"Error fetching credits for user {user_id}: {e}")
        return 0