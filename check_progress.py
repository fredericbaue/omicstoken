"""Monitor re-embedding progress"""
import sqlite3
import sys
import os

sys.path.append(os.getcwd())
import db

DATA_DIR = "data"

con = db.get_db_connection(DATA_DIR)
cur = con.cursor()

# Count total features
cur.execute("SELECT COUNT(*) FROM features")
total_features = cur.fetchone()[0]

# Count embedded features
cur.execute("SELECT COUNT(*) FROM peptide_embeddings")
embedded = cur.fetchone()[0]

con.close()

print(f"Progress: {embedded}/{total_features} peptides embedded ({100*embedded/total_features if total_features > 0 else 0:.1f}%)")
