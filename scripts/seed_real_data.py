import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import db
import importers
import embeddings
import search

TARGETS = [
    ("maxquant_peptides.csv", "RUN_MQ_01"),
    ("maxquant_peptides2.csv", "RUN_MQ_02"),
]

def ingest_file(con, file_name, run_id):
    data_path = Path(config.DATA_DIR) / file_name
    if not data_path.exists():
        print(f"??  File {data_path} not found. Skipping {run_id}.")
        return 0

    print(f"??  Loading {data_path} for {run_id}")
    df = pd.read_csv(data_path)
    features = importers.parse_upload(df, fmt="maxquant")

    meta_dict = {
        "run_id": run_id,
        "original_filename": file_name,
        "instrument": df["Raw file"].iloc[0] if "Raw file" in df.columns and not df.empty else None,
        "method": "maxquant",
    }

    db.insert_run(con, run_id, meta_dict)

    for feature in features:
        feature.run_id = run_id
        db.insert_feature(con, run_id, feature)
        vector = embeddings.peptide_to_vector(feature.peptide_sequence or "")
        db.insert_peptide_embedding(
            con,
            run_id=run_id,
            user_id="demo",
            feature_id=feature.feature_id,
            sequence=feature.peptide_sequence or "",
            intensity=feature.intensity or 0.0,
            length=len(feature.peptide_sequence or ""),
            charge=(feature.charge or 0),
            hydrophobicity=(feature.metadata or {}).get("hydrophobicity", 0.0),
            vector=vector,
        )

    con.commit()
    print(f"? Ingested {len(features)} peptides from {file_name} as {run_id}")
    print(f"?? You can now select {run_id} in the Dashboard.\n")
    return len(features)

def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    con = db.get_db_connection(config.DATA_DIR)
    total = 0
    try:
        for file_name, run_id in TARGETS:
            total += ingest_file(con, file_name, run_id)

        print("?? Rebuilding FAISS index ...")
        search.rebuild_faiss_index(con, config.DATA_DIR)
        print("? Index rebuild complete.")
        print(f"?? Finished seeding {total} peptides across {len(TARGETS)} runs.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
