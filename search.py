import json
import logging
import os
import sqlite3
import threading
from typing import List, Optional, Tuple

import faiss
import numpy as np
from fastapi import HTTPException

import db
import embeddings  # noqa: F401  # kept for API symmetry / future use
import models

# --- Configuration ---
FAISS_INDEX_FILENAME = "peptides.faiss"
FAISS_IDS_FILENAME = "peptides_ids.json"
_REBUILD_LOCK = threading.Lock()


def get_faiss_index_path(data_dir: str) -> str:
    """
    Build the full filesystem path to the FAISS index file for a given data_dir.
    """
    return os.path.join(data_dir, FAISS_INDEX_FILENAME)


def get_faiss_ids_path(data_dir: str) -> str:
    """
    Build the full filesystem path to the FAISS ID map JSON for a given data_dir.
    """
    return os.path.join(data_dir, FAISS_IDS_FILENAME)


def rebuild_faiss_index(
    con: Optional[sqlite3.Connection],
    data_dir: str,
) -> int:
    """
    Rebuild the global FAISS index from all peptide embeddings in the database.

    - Uses the provided connection if given; otherwise opens a fresh connection
      (safe for background threads).
    - Reads embeddings via db.get_all_embeddings_data.
    - Normalizes vectors (L2) and builds a faiss.IndexFlatL2.
    - Writes to temporary files then atomically replaces live index/id map.

    If no embeddings exist, any existing index/id files are removed. On failure,
    the previous index is left intact and 0 is returned.

    Returns
    -------
    int
        The number of vectors in the rebuilt index (0 on failure).
    """
    faiss_index_path = get_faiss_index_path(data_dir)
    faiss_ids_path = get_faiss_ids_path(data_dir)
    temp_index_path = f"{faiss_index_path}.tmp"
    temp_ids_path = f"{faiss_ids_path}.tmp"

    if _REBUILD_LOCK.locked():
        logging.info("FAISS rebuild already in progress; waiting for lock.")

    with _REBUILD_LOCK:
        logging.info("Rebuilding FAISS index...")
        created_con = False
        if con is None:
            con = db.get_db_connection(data_dir)
            created_con = True

        try:
            rows = db.get_all_embeddings_data(con)

            if not rows:
                logging.info("No embeddings found to build index. Removing any existing index files.")
                for path in (faiss_index_path, faiss_ids_path, temp_index_path, temp_ids_path):
                    if os.path.exists(path):
                        os.remove(path)
                return 0

            ids: List[Tuple[str, str]] = [(row[0], row[1]) for row in rows]
            try:
                vectors = np.array([json.loads(row[2]) for row in rows], dtype=np.float32)
            except (json.JSONDecodeError, TypeError, ValueError):
                logging.exception("Failed to decode embeddings for FAISS index.")
                return 0

            if vectors.ndim != 2:
                logging.error("Expected 2D vectors array for FAISS index, got shape %s", vectors.shape)
                return 0

            d = vectors.shape[1]
            index = faiss.IndexFlatL2(d)
            faiss.normalize_L2(vectors)
            index.add(vectors)

            # Write to temporary files first, then atomically replace.
            faiss.write_index(index, temp_index_path)
            with open(temp_ids_path, "w") as f:
                json.dump(ids, f)

            os.replace(temp_index_path, faiss_index_path)
            os.replace(temp_ids_path, faiss_ids_path)

            logging.info("FAISS index rebuilt with %s vectors.", index.ntotal)
            return index.ntotal

        except Exception:
            logging.exception("Failed to rebuild FAISS index; leaving prior index intact.")
            for path in (temp_index_path, temp_ids_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            return 0
        finally:
            if created_con:
                con.close()


def search_similar_peptides(
    con: sqlite3.Connection,
    data_dir: str,
    run_id: str,
    feature_id: str,
    k: int = 5,
) -> models.SimilarPeptidesResponse:
    """
    Find peptides with similar biophysical properties using vector similarity.

    This function:
    - Loads the FAISS index and ID map from disk
    - Fetches the query peptide's embedding from the DB
    - Performs a k-NN search in the FAISS index
    - Converts L2 distances back to cosine similarity
    - Fetches metadata for each neighbor from the DB
    - Returns a models.SimilarPeptidesResponse

    On error (missing files, missing embedding, missing feature), it raises
    fastapi.HTTPException, which is expected by the FastAPI layer.
    """
    faiss_index_path = get_faiss_index_path(data_dir)
    faiss_ids_path = get_faiss_ids_path(data_dir)

    if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_ids_path):
        raise HTTPException(
            404,
            "FAISS index not found. Please upload data and ensure embeddings are generated.",
        )

    # Load index
    try:
        index = faiss.read_index(faiss_index_path)
    except Exception as e:
        # Any low-level FAISS error is surfaced as a server error
        raise HTTPException(
            500,
            f"Failed to load FAISS index from disk: {str(e)}",
        )

    # Load ID map
    try:
        with open(faiss_ids_path, "r") as f:
            ids_map: List[Tuple[str, str]] = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(
            500,
            f"Failed to load FAISS ID map from disk: {str(e)}",
        )

    # Fetch query embedding
    query_vector_list = db.get_embedding(con, run_id, feature_id)
    if not query_vector_list:
        raise HTTPException(
            404,
            f"Embedding for feature '{feature_id}' in run '{run_id}' "
            "not found (did you embed it?).",
        )

    query_vector = np.array(query_vector_list, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Fetch original feature properties (sequence, intensity)
    query_feat_props = db.get_feature_properties(con, run_id, feature_id)
    if not query_feat_props:
        raise HTTPException(
            404,
            f"Original feature '{feature_id}' in run '{run_id}' not found in DB.",
        )

    # Perform k-NN search
    try:
        distances, indices = index.search(query_vector, k)
    except Exception as e:
        raise HTTPException(
            500,
            f"FAISS search failed: {str(e)}",
        )

    neighbors: List[models.SimilarPeptide] = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        try:
            neighbor_run_id, neighbor_feature_id = ids_map[idx]
        except (IndexError, ValueError, TypeError):
            # ID map and index are out of sync; skip this neighbor
            continue

        dist = float(distances[0][i])
        # Convert L2 distance back to cosine similarity
        similarity = 1.0 - (dist ** 2) / 2.0

        feat_row = db.get_feature_properties(con, neighbor_run_id, neighbor_feature_id)

        neighbors.append(
            models.SimilarPeptide(
                run_id=neighbor_run_id,
                feature_id=neighbor_feature_id,
                peptide_sequence=feat_row[0] if feat_row else None,
                similarity=float(similarity),
                intensity=feat_row[1] if feat_row else None,
            )
        )

    query_peptide = models.QueryPeptide(
        run_id=run_id,
        feature_id=feature_id,
        peptide_sequence=query_feat_props[0],
        intensity=query_feat_props[1],
    )

    return models.SimilarPeptidesResponse(query=query_peptide, neighbors=neighbors)
