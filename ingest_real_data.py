import pandas as pd
import os
import db
import importers
from embeddings import peptide_to_vector, EMBEDDING_DIM

DATA_DIR = "data"
FILENAME = "Melanoma vs. MPNST.csv"
FILEPATH = os.path.join(DATA_DIR, FILENAME)
RUN_ID = "RUN_MELANOMA_VS_MPNST"

def ingest_real_data():
    print(f"--- Ingesting {FILENAME} ---")
    
    # 1. Read CSV
    try:
        df = pd.read_csv(FILEPATH)
        print(f"Loaded CSV with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Parse Features
    # We force "tumor_csv" format as we know the source
    try:
        features = importers.parse_tumor_csv(df)
        print(f"Parsed {len(features)} features.")
    except Exception as e:
        print(f"Error parsing features: {e}")
        return

    # 3. Insert into DB
    con = db.get_db_connection(DATA_DIR)
    
    # Check if run exists, delete if so to refresh
    existing_run = db.get_run(con, RUN_ID)
    if existing_run:
        print(f"Run {RUN_ID} exists. Cleaning up old data...")
        con.execute("DELETE FROM features WHERE run_id=?", (RUN_ID,))
        con.execute("DELETE FROM embeddings WHERE run_id=?", (RUN_ID,))
        con.execute("DELETE FROM runs WHERE run_id=?", (RUN_ID,))
        con.commit()
        
    # Insert Run
    meta_dict = {"original_filename": FILENAME, "description": "Real data from Simplifi"}
    db.insert_run(con, RUN_ID, meta_dict)
    
    # Insert Features
    print("Inserting features into DB...")
    for feat in features:
        db.insert_feature(con, RUN_ID, feat)
    con.commit()
    
    # 4. Embed Peptides (for search/clustering)
    print("Generating embeddings...")
    count = 0
    for feat in features:
        vec = peptide_to_vector(feat.peptide_sequence)
        if vec is not None and vec.shape[0] == EMBEDDING_DIM:
            db.insert_embedding(con, RUN_ID, feat.feature_id, "unknown", "unknown", vec)
            count += 1
            
    con.commit()
    print(f"Embedded {count} peptides.")
    
    con.close()
    print(f"--- Ingestion Complete. Run ID: {RUN_ID} ---")

if __name__ == "__main__":
    ingest_real_data()
