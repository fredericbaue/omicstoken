"""
Script to safely add model_version column to db.py
"""
import re

# Read the file
with open('db.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add model_version to the CREATE TABLE statement
old_create = '''    con.execute("""CREATE TABLE IF NOT EXISTS peptide_embeddings(
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
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(run_id, feature_id), -- Ensure unique embedding per run and feature
        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
        FOREIGN KEY(run_id, feature_id) REFERENCES features(run_id, feature_id) ON DELETE CASCADE
    )""")
    con.commit()'''

new_create = '''    con.execute("""CREATE TABLE IF NOT EXISTS peptide_embeddings(
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
    
    con.commit()'''

content = content.replace(old_create, new_create)

# 2. Update insert_peptide_embedding function
old_insert = '''def insert_peptide_embedding(con: sqlite3.Connection, run_id: str, user_id: str, feature_id: str, sequence: str, intensity: float, 
                             length: int, charge: int, hydrophobicity: float, vector: np.ndarray):
    """Inserts a row into the new peptide_embeddings table."""
    con.execute(
        """INSERT INTO peptide_embeddings(run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, embedding)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, json.dumps(vector.tolist()))
    )'''

new_insert = '''def insert_peptide_embedding(con: sqlite3.Connection, run_id: str, user_id: str, feature_id: str, sequence: str, intensity: float, 
                             length: int, charge: int, hydrophobicity: float, vector: np.ndarray, model_version: str = "v2_gemini_768"):
    """Inserts a row into the new peptide_embeddings table."""
    con.execute(
        """INSERT INTO peptide_embeddings(run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, embedding, model_version)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (run_id, user_id, feature_id, sequence, intensity, length, charge, hydrophobicity, json.dumps(vector.tolist()), model_version)
    )'''

content = content.replace(old_insert, new_insert)

# Write back
with open('db.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Successfully updated db.py")
