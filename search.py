import faiss
import numpy as np
import json
import os
import sqlite3
from typing import List, Tuple
from fastapi import HTTPException

import db
import embeddings
import models

# --- Configuration ---
FAISS_INDEX_FILENAME = "peptides.faiss"
FAISS_IDS_FILENAME = "peptides_ids.json"

def get_faiss_index_path(data_dir: str) -> str:
    return os.path.join(data_dir, FAISS_INDEX_FILENAME)

def get_faiss_ids_path(data_dir: str) -> str:
    return os.path.join(data_dir, FAISS_IDS_FILENAME)

def rebuild_faiss_index(con: sqlite3.Connection, data_dir: str):
    """
    Queries all embeddings from the DB, builds a FAISS index, and saves it.
    This is slow and should be run as a background task in a real app.
    """
    print("Rebuilding FAISS index...")
    faiss_index_path = get_faiss_index_path(data_dir)
    faiss_ids_path = get_faiss_ids_path(data_dir)

    rows = db.get_all_embeddings_data(con)

    if not rows:
        print("No embeddings found to build index.")
        # Ensure any old index files are removed if no embeddings exist
        if os.path.exists(faiss_index_path): os.remove(faiss_index_path)
        if os.path.exists(faiss_ids_path): os.remove(faiss_ids_path)
        return 0

    ids = [(row[0], row[1]) for row in rows] # (run_id, feature_id)
    vectors = np.array([json.loads(row[2]) for row in rows], dtype=np.float32)

    d = vectors.shape[1]  # vector dimension
    index = faiss.IndexFlatL2(d)  # Using L2 distance
    faiss.normalize_L2(vectors) # Normalize vectors to use L2 as cosine similarity
    index.add(vectors)

    faiss.write_index(index, faiss_index_path)
    with open(faiss_ids_path, "w") as f:
        json.dump(ids, f)
    print(f"FAISS index rebuilt with {index.ntotal} vectors.")
    return index.ntotal

def search_similar_peptides(con: sqlite3.Connection, data_dir: str, run_id: str, feature_id: str, k: int = 5) -> models.SimilarPeptidesResponse:
    """Finds peptides with similar biophysical properties using vector similarity."""
    faiss_index_path = get_faiss_index_path(data_dir)
    faiss_ids_path = get_faiss_ids_path(data_dir)

    if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_ids_path):
        raise HTTPException(404, "FAISS index not found. Please upload data and ensure embeddings are generated.")

    index = faiss.read_index(faiss_index_path)
    with open(faiss_ids_path, "r") as f:
        ids_map = json.load(f)

    query_vector_list = db.get_embedding(con, run_id, feature_id)
    if not query_vector_list:
        raise HTTPException(404, f"Embedding for feature '{feature_id}' in run '{run_id}' not found (did you embed it?).")
    query_vector = np.array(query_vector_list, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    query_feat_props = db.get_feature_properties(con, run_id, feature_id)
    if not query_feat_props:
        raise HTTPException(404, f"Original feature '{feature_id}' in run '{run_id}' not found in DB.")

    distances, indices = index.search(query_vector, k)

    out = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        neighbor_run_id, neighbor_feature_id = ids_map[idx]
        dist = distances[0][i]
        similarity = 1.0 - (dist**2) / 2.0 # Convert L2 distance back to cosine similarity

        feat_row = db.get_feature_properties(con, neighbor_run_id, neighbor_feature_id)
        out.append(models.SimilarPeptide(
            run_id=neighbor_run_id,
            feature_id=neighbor_feature_id,
            peptide_sequence=feat_row[0] if feat_row else None,
            similarity=float(similarity),
            intensity=feat_row[1] if feat_row else None,
        ))

    query_peptide = models.QueryPeptide(
        run_id=run_id,
        feature_id=feature_id,
        peptide_sequence=query_feat_props[0],
        intensity=query_feat_props[1]
    )

    return models.SimilarPeptidesResponse(query=query_peptide, neighbors=out)