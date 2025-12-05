import logging
import sqlite3
from typing import Optional

from fastapi import HTTPException, status

import config
import db
import search
from embeddings import EMBEDDING_DIM, peptide_to_vector

LOGGER = logging.getLogger(__name__)

HYDROPHOBICITY_SCALE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

AMINO_ACID_WEIGHTS = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}
WATER_MASS = 18.01056


def calculate_molecular_weight(sequence: str) -> float:
    if not sequence:
        return 0.0
    weight = WATER_MASS
    for aa in sequence.upper():
        weight += AMINO_ACID_WEIGHTS.get(aa, 0.0)
    return round(weight, 4)


def calculate_properties(sequence: str):
    if not sequence:
        return {"hydrophobicity": 0, "molecular_weight": 0, "charge": 0, "length": 0}

    valid_seq = [aa for aa in sequence.upper() if aa in HYDROPHOBICITY_SCALE]
    hydro = (
        sum(HYDROPHOBICITY_SCALE.get(aa, 0) for aa in valid_seq) / len(valid_seq)
        if valid_seq
        else 0
    )

    mw = calculate_molecular_weight(sequence)
    pos = sum(sequence.upper().count(aa) for aa in ["K", "R", "H"])
    neg = sum(sequence.upper().count(aa) for aa in ["D", "E"])
    return {
        "hydrophobicity": round(hydro, 2),
        "molecular_weight": mw,
        "charge": pos - neg,
        "length": len(sequence),
    }


def run_embedding_pipeline(
    run_id: str, expected_user_id: Optional[str] = None, trigger_rebuild: bool = True
):
    """Shared embedding worker with ownership enforcement, idempotency, and safe DB handling."""
    con = db.get_db_connection(config.DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if run is None or (
            expected_user_id is not None and str(run.user_id) != str(expected_user_id)
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )
        LOGGER.info("Embedding pipeline start for run %s (user %s)", run_id, run.user_id)

        model_version_id = db.get_or_create_model_version(
            con,
            name=config.EMBEDDER_NAME or "esm2",
            version=config.EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
        )
        features = db.get_features_for_run(con, run_id)
        if not features:
            LOGGER.warning("No features found for run %s; skipping embedding.", run_id)
            return {"run_id": run_id, "peptides_embedded": 0}

        con.execute("DELETE FROM peptide_embeddings WHERE run_id=?", (run_id,))

        count = 0
        total_features = len(features)
        LOGGER.info("Embedding %s peptides for run %s...", total_features, run_id)
        for i, feat in enumerate(features):
            if (i + 1) % 100 == 0:
                LOGGER.info("Processed %s/%s peptides for run %s", i + 1, total_features, run_id)
            vec = peptide_to_vector(feat.peptide_sequence)

            if vec is not None and vec.shape[0] == EMBEDDING_DIM:
                props = calculate_properties(feat.peptide_sequence)
                try:
                    db.insert_peptide_embedding(
                        con,
                        run_id,
                        run.user_id,
                        feat.feature_id,
                        feat.peptide_sequence,
                        feat.intensity or 0.0,
                        props["length"],
                        props["charge"],
                        props["hydrophobicity"],
                        vec,
                        model_version=config.EMBEDDING_MODEL_NAME,
                        model_version_id=model_version_id,
                    )
                    count += 1
                except sqlite3.IntegrityError as ie:
                    LOGGER.warning(
                        "Duplicate embedding skipped for feature %s in run %s (%s)",
                        feat.feature_id,
                        run_id,
                        ie,
                    )
            else:
                LOGGER.warning(
                    "Skipping embedding for feature %s in run %s due to invalid sequence/vector.",
                    feat.feature_id,
                    run_id,
                )

        con.commit()
        LOGGER.info(
            "Embedding pipeline completed for run %s: stored %s/%s embeddings",
            run_id,
            count,
            total_features,
        )
        if trigger_rebuild:
            try:
                search.rebuild_faiss_index(None, config.DATA_DIR, triggered_by_run=run_id)
                LOGGER.info("FAISS index rebuild completed after embedding run %s", run_id)
            except Exception as e:
                LOGGER.exception("FAISS index rebuild failed after embedding run %s: %s", run_id, e)
        return {"run_id": run_id, "peptides_embedded": count}
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        LOGGER.exception("Embedding failed for run %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail="Embedding failed due to an internal error.")
    finally:
        con.close()
