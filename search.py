import json
import os
import sqlite3
from typing import List, Tuple

import faiss
import numpy as np
from fastapi import HTTPException

import db
import embeddings  # noqa: F401  # kept for API symmetry / future use
import models

# --- Configuration ---
FAISS_INDEX_FILENAME = "peptides.faiss"
FAISS_IDS_FILENAME = "peptides_ids.json"


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


def rebuild_faiss_index(con: sqlite3.Connection, data_dir: str) -> int:
    """
    Rebuild the global FAISS index from all peptide embeddings in the database.

    - Reads all embeddings via db.get_all_embeddings_data(con)
    - Normalizes them (L2) and builds a faiss.IndexFlatL2
    - Persists the index to disk along with a JSON ID map

    This is typically run as a background task. If no embeddings exist, any
    previously stored index + id map files are removed.

    Returns
    -------
    int
        The number of vectors in the rebuilt index.
    """
    print("Rebuilding FAISS index...")
    faiss_index_path = get_faiss_index_path(data_dir)
    faiss_ids_path = get_faiss_ids_path(data_dir)

    rows = db.get_all_embeddings_data(con)

    if not rows:
        print("No embeddings found to build index.")
        # Ensure any old index files are removed if no embeddings exist
        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
        if os.path.exists(faiss_ids_path):
            os.remove(faiss_ids_path)
        return 0

    # rows are expected to be (run_id, feature_id, embedding_json)
    ids: List[Tuple[str, str]] = [(row[0], row[1]) for row in rows]
    try:
        vectors = np.array([json.loads(row[2]) for row in rows], dtype=np.float32)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # This should not normally happen; surface clearly for debugging.
        raise RuntimeError(f"Failed to decode embeddings for FAISS index: {e}")

    if vectors.ndim != 2:
        raise RuntimeError(
            f"Expected 2D vectors array for FAISS index, got shape {vectors.shape}"
        )

    d = vectors.shape[1]  # vector dimension
    index = faiss.IndexFlatL2(d)  # Using L2 distance
    faiss.normalize_L2(vectors)   # Normalize vectors to use L2 as cosine similarity
    index.add(vectors)

    faiss.write_index(index, faiss_index_path)
    with open(faiss_ids_path, "w") as f:
        json.dump(ids, f)

    print(f"FAISS index rebuilt with {index.ntotal} vectors.")
    return index.ntotal


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