"""
Re-embed all peptides and rebuild the FAISS index.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

import db
import embeddings
import search
from embeddings import EMBEDDING_DIM
import config

DATA_DIR = config.DATA_DIR


def re_embed_all_peptides():
    """Re-embed all peptides in the database and rebuild the FAISS index."""
    print("=" * 60)
    print("Re-embedding all peptides with current embedder...")
    print("=" * 60)

    con = db.get_db_connection(DATA_DIR)

    cur = con.cursor()
    cur.execute("SELECT run_id, user_id FROM runs")
    runs = cur.fetchall()

    if not runs:
        print("No runs found in database.")
        return

    total_embedded = 0

    for run_id, user_id in runs:
        print(f"\nProcessing run: {run_id}")
        features = db.get_features_for_run(con, run_id)
        print(f"  Found {len(features)} features")

        cur.execute("DELETE FROM peptide_embeddings WHERE run_id=?", (run_id,))
        con.commit()
        print("  Cleared old embeddings")

        embedded_count = 0
        for feature in features:
            if not feature.peptide_sequence:
                continue

            vec = embeddings.peptide_to_vector(feature.peptide_sequence)
            if vec is None or len(vec) != EMBEDDING_DIM:
                continue

            length = len(feature.peptide_sequence)
            charge = (
                feature.peptide_sequence.count("K")
                + feature.peptide_sequence.count("R")
                - feature.peptide_sequence.count("D")
                - feature.peptide_sequence.count("E")
            )
            hydrophobic_aas = set("AVILMFYW")
            hydrophobicity = (
                sum(1 for aa in feature.peptide_sequence if aa in hydrophobic_aas) / length
                if length > 0
                else 0
            )

            db.insert_peptide_embedding(
                con=con,
                run_id=run_id,
                user_id=user_id or "unknown",
                feature_id=feature.feature_id,
                sequence=feature.peptide_sequence,
                intensity=feature.intensity or 0.0,
                length=length,
                charge=charge,
                hydrophobicity=hydrophobicity,
                vector=vec,
            )
            embedded_count += 1

        con.commit()
        print(f"  Embedded {embedded_count} peptides")
        total_embedded += embedded_count

    print(f"\n{'=' * 60}")
    print(f"Total peptides re-embedded: {total_embedded}")
    print(f"{'=' * 60}\n")

    print("Rebuilding FAISS index...")
    n_vectors = search.rebuild_faiss_index(None, DATA_DIR)
    print(f"FAISS index rebuilt with {n_vectors} vectors\n")

    con.close()
    print("Re-embedding complete!")


if __name__ == "__main__":
    re_embed_all_peptides()
