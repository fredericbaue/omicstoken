from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
from fastapi import Request
from pydantic import BaseModel, Field
import uvicorn
import os
import shutil
import pandas as pd
import numpy as np
import asyncio
import random
import inspect
import time
import uuid
import sqlite3
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from typing import List, Optional
import logging
from datetime import datetime
import io
import zipfile
import json

import embeddings
from embeddings import peptide_to_vector, EMBEDDING_DIM
import db
import search
import models
import importers
import summarizer
import export
import auth
import config
from worker import queue_embed_run, queue_generate_summary
from auth import current_active_user, current_active_user_or_token, User

LOGGER = logging.getLogger(__name__)
REQUEST_LOGGER = logging.getLogger("request_logger")
REQUEST_LOGGER.setLevel(logging.getLevelName(config.LOG_LEVEL))

# --- Configuration ---
DATA_DIR = config.DATA_DIR

app = FastAPI(title="Immuno-Peptidomics MVP")

# CORS (Optional, good for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _get_user_identifier(request: Request) -> str:
    """Best-effort extraction of a user identifier for logging."""
    user = getattr(request.state, "user", None)
    if user is None:
        return "unknown"
    return getattr(user, "email", None) or str(getattr(user, "id", "unknown"))

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    correlation_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    request.state.correlation_id = correlation_id

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        status_code = getattr(exc, "status_code", 500)
        if status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN):
            LOGGER.warning(
                "Auth failure on %s %s (status=%s, user=%s, correlation_id=%s)",
                request.method,
                request.url.path,
                status_code,
                _get_user_identifier(request),
                correlation_id,
            )
        payload = {
            "event": "request_log",
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "status": status_code,
            "latency_ms": latency_ms,
            "user": _get_user_identifier(request),
            "error": str(exc),
        }
        REQUEST_LOGGER.exception(json.dumps(payload))
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Correlation-ID"] = correlation_id
    payload = {
        "event": "request_log",
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
        "query": str(request.url.query),
        "status": response.status_code,
        "latency_ms": latency_ms,
        "user": _get_user_identifier(request),
    }
    REQUEST_LOGGER.info(json.dumps(payload))
    return response

@app.on_event("startup")
async def on_startup():
    await auth.create_db_and_tables()
    if config.DEMO_MODE:
        LOGGER.warning("DEMO_MODE is enabled. Seed demo data with scripts/seed_demo_data.py.")

# --- Auth Routes ---
app.include_router(
    auth.fastapi_users.get_auth_router(auth.auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    auth.fastapi_users.get_register_router(auth.UserRead, auth.UserCreate),
    prefix="/auth",
    tags=["auth"],
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=307)

def _embed_run(run_id: str, expected_user_id: Optional[str] = None, trigger_rebuild: bool = True):
    """Shared embedding worker with ownership enforcement, idempotency, and safe DB handling."""
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if run is None or (expected_user_id is not None and str(run.user_id) != str(expected_user_id)):
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

        # Idempotency: remove any existing embeddings for this run before re-embedding.
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
                search.rebuild_faiss_index(None, DATA_DIR, triggered_by_run=run_id)
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


@app.post("/peptide/embed/{run_id}")
def embed(run_id: str, user: User = Depends(current_active_user)):
    """Vectorize each peptide in a run and persist the vectors."""
    return _embed_run(run_id, expected_user_id=str(user.id))

@app.get("/upload", include_in_schema=False)
def upload_page():
    return RedirectResponse(url="/static/upload.html")


@app.post("/upload", include_in_schema=False)
async def upload_handler(request: Request,
                         file: UploadFile = File(...),
                         run_id: str = Form(""),
                         instrument: str = Form(""),
                         method: str = Form(""),
                         format: str = Form("auto"),
                         user: User = Depends(current_active_user)):
    """
    Handles file uploads, detects format, ingests data, and triggers embedding.
    """
    # --- Ingest Logic ---
    name = file.filename or "upload.csv"
    
    # 🚨 FIX: Check for common tab-separated extensions (.txt or .tsv)
    if name.endswith(".tsv") or name.endswith(".txt"):
        sep = "\t"
    else:
        sep = "," # Default to comma for .csv
        
    await file.seek(0)
    
    try:
        df = pd.read_csv(file.file, sep=sep)
    except Exception as e:
        LOGGER.error("Failed to parse CSV for upload: %s", e)
        raise HTTPException(400, f"Failed to parse CSV: {e}")

    # Use the importers module to parse the dataframe
    try:
        features = importers.parse_upload(df, fmt=format)
    except Exception as e:
        LOGGER.error("Import failed for upload: %s", e)
        raise HTTPException(400, f"Import failed: {e}")

    # Build meta dict
    meta_dict = {
        "run_id": run_id.strip() or None,
        "instrument": instrument.strip() or None,
        "method": method.strip() or None,
        "original_filename": name
    }
    run_id_final = meta_dict.get("run_id") or f"RUN_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Store in SQLite
    con = db.get_db_connection(DATA_DIR)
    db.insert_run(con, run_id_final, meta_dict, user_id=str(user.id))

    for feat in features:
        db.insert_feature(con, run_id_final, feat)
        
    con.commit()
    db.update_user_credits(con, str(user.id), 10) # Increment credits
    LOGGER.info("User %s mined 10 credits", user.email)
    con.close()
    rows_ingested = len(features)
    LOGGER.info(
        "Upload ingested %s features for run %s (user %s, file=%s)",
        rows_ingested,
        run_id_final,
        user.id,
        name,
    )

    # --- Background Jobs via Celery ---
    try:
        job = queue_embed_run(run_id_final, str(user.id))
        LOGGER.info(
            "Dispatched Celery embed_run_task for run %s (user %s, task_id=%s, broker=%s)",
            run_id_final,
            user.id,
            job.id,
            config.CELERY_BROKER_URL,
        )
    except Exception as e:
        LOGGER.exception(
            "Failed to dispatch Celery embed_run_task for run %s (user %s): %s",
            run_id_final,
            user.id,
            e,
        )
        raise HTTPException(status_code=500, detail="Failed to queue embedding job. Please try again later.")

    # --- Success Response ---
    first_feature_id = features[0].feature_id if features else None
    
    return {
        "status": "success",
        "run_id": run_id_final,
        "rows_ingested": rows_ingested,
        "format": format,
        "first_feature_id": first_feature_id,
        "message": "Upload complete and embedding started in background."
    }

@app.get("/peptide/search/{run_id}/{feature_id}")
def similar(run_id: str, feature_id: str, k: int = 5, user: User = Depends(current_active_user)):
    """Find peptides with similar biophysical properties."""
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )
        return search.search_similar_peptides(con, DATA_DIR, run_id, feature_id, k)
    except HTTPException as e:
        raise e
    finally:
        con.close()

@app.get("/peptide/explain/{run_id}/{feature_id}")
def explain_peptide(run_id: str, feature_id: str, user: User = Depends(current_active_user)):
    """Generate an LLM-powered explanation for a specific peptide."""
    con = db.get_db_connection(DATA_DIR)
    try:
        # Verify run ownership
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )
        
        # Get peptide data
        props = db.get_feature_properties(con, run_id, feature_id)
        if not props:
            raise HTTPException(404, "Peptide not found")
        
        sequence, intensity = props
        
        # Calculate biophysical properties
        peptide_props = calculate_properties(sequence)
        
        # Get similar peptides (k=10 neighbors)
        similar_peptides_response = search.search_similar_peptides(con, DATA_DIR, run_id, feature_id, k=10)
        neighbors = similar_peptides_response.neighbors
        
        # Prepare neighbor data for LLM
        neighbors_summary = []
        for n in neighbors[:5]:  # Top 5 for brevity
            neighbors_summary.append({
                "sequence": n.peptide_sequence,
                "similarity_score": round(n.similarity, 3),
                "intensity": n.intensity or 0.0
            })
        
        # Build LLM prompt
        prompt = f"""You are an expert proteomics assistant. A user is looking at a peptide from a mass spectrometry experiment.

Peptide: {sequence}
Length: {peptide_props['length']} amino acids
Intensity: {intensity:,.0f}
Hydrophobicity (GRAVY): {peptide_props['hydrophobicity']}
Charge: {peptide_props['charge']}
Similar peptides in this dataset:
{chr(10).join([f"- {n['sequence']} (similarity: {n['similarity_score']}, intensity: {n.get('intensity', 0):,.0f})" for n in neighbors_summary])}

Explain in 2 short paragraphs what's interesting about this peptide. Mention any obvious biophysical features, whether it looks like a common tryptic peptide (check for K/R cleavage sites), and what the neighbor set suggests (e.g., shared motifs, possible functional or structural similarities). Use accessible scientific language but keep it concise."""

        # Call Gemini API
        try:
            import google.generativeai as genai
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            
            if not GOOGLE_API_KEY:
                raise HTTPException(500, "GOOGLE_API_KEY not configured")
            
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            explanation = response.text
            
        except Exception as e:
            raise HTTPException(500, f"LLM generation failed: {str(e)}")
        
        # Return structured response
        return {
            "sequence": sequence,
            "length": peptide_props['length'],
            "intensity": intensity,
            "hydrophobicity": peptide_props['hydrophobicity'],
            "charge": peptide_props['charge'],
            "neighbors": neighbors_summary,
            "explanation": explanation
        }
        
    finally:
        con.close()

@app.post("/summary/run/{run_id}")
def get_run_summary(run_id: str, user: User = Depends(current_active_user)):
    """
    Generates a text summary of the run using the Summarization Engine.
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        summary_result = None
        # Prefer Celery to keep the request thread light; fall back inline if the worker is unavailable.
        try:
            job = queue_generate_summary(run_id)
            LOGGER.info(
                "Dispatched generate_summary_task for run %s from /summary/run (task_id=%s, broker=%s)",
                run_id,
                job.id,
                config.CELERY_BROKER_URL,
            )
            try:
                summary_result = job.get(timeout=60)
            except Exception as celery_err:
                LOGGER.warning(
                    "Celery summary fetch failed for run %s; falling back inline: %s",
                    run_id,
                    celery_err,
                )
        except Exception as e:
            LOGGER.warning(
                "Celery dispatch for summary failed for run %s; falling back inline: %s",
                run_id,
                e,
            )

        if summary_result is None:
            summary_result = summarizer.generate_summary(run_id, con)
        return summary_result
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Summary generation failed for run %s: %s", run_id, e)
        raise HTTPException(
            status_code=500, detail="Summary generation failed due to an internal error."
        )
    finally:
        con.close()

@app.get("/runs")
def list_runs(user: User = Depends(current_active_user)):
    """List all runs for the current user."""
    con = db.get_db_connection(DATA_DIR)
    try:
        run_summaries = db.get_run_summaries(con, user_id=str(user.id))
        user_credits = db.get_user_credits(con, str(user.id))
        
        return {
            "user_credits": user_credits if user_credits is not None else 0,
            "runs": run_summaries
        }
    finally:
        con.close()

@app.get("/runs/{run_id}")
def get_run_details(run_id: str, user: User = Depends(current_active_user)):
    """Get detailed information about a specific run."""
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        
        # If the run doesn't exist or doesn't belong to this user, hide it
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )
            
        # Get counts
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM features WHERE run_id=?", (run_id,))
        n_features = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM peptide_embeddings WHERE run_id=?", (run_id,))
        n_embeddings = cur.fetchone()[0]
        
        # Check for first feature for search link
        cur.execute("SELECT feature_id FROM features WHERE run_id=? LIMIT 1", (run_id,))
        row = cur.fetchone()
        first_feature_id = row[0] if row else None
        
        return {
            "run": run,
            "stats": {
                "n_features": n_features,
                "n_embeddings": n_embeddings
            },
            "links": {
                "summary": f"/summary/run/{run_id}",
                "search_example": f"/peptide/search/{run_id}/{first_feature_id}" if first_feature_id else None
            }
        }
    finally:
        con.close()

@app.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: str, background_tasks: BackgroundTasks, user: User = Depends(current_active_user)):
    """
    Deletes a run and all its associated data (features, embeddings).
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        # 1. Verify run ownership
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found or you do not have permission to delete it.",
            )

        # 2. Delete all associated data from tables
        cur = con.cursor()
        cur.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM features WHERE run_id=?", (run_id,))
        cur.execute("DELETE FROM peptide_embeddings WHERE run_id=?", (run_id,))
        con.commit()

        # 3. Rebuild the search index in the background
        # This is important to remove the deleted vectors from the search index.
        background_tasks.add_task(search.rebuild_faiss_index, None, DATA_DIR)

        logging.info(f"User {user.email} deleted run {run_id}.")
        # No content is returned on success, per HTTP 204.

    finally:
        con.close()

# In app.py

@app.get("/runs/{run_id}/fingerprint")
def get_run_fingerprint(run_id: str, user: User = Depends(current_active_user)):
    """
    Generates a 'Semantic Fingerprint' for a run, including peptide clustering
    and a 2D projection.
    """
    logging.warning(f"ACCESSING /runs/{run_id}/fingerprint")
    con = db.get_db_connection(DATA_DIR)
    try:
        # 1. Authentication & Authorization
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            logging.warning(f"Run {run_id} not found or user mismatch.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        # 2. Fetch peptide embeddings for the run
        peptides = db.get_peptide_embeddings(con, run_id)
        if not peptides:
            raise HTTPException(404, "No peptide embeddings found for this run.")

        total_peptides = len(peptides)
        
        # 🚨 THE FIX: Minimum sample size check to prevent scikit-learn crash
        if total_peptides < 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Fingerprint requires at least 4 peptides (found {total_peptides}). Please upload more data."
            )
        
        # 3. Sample if necessary (rest of logic remains the same)
        sample_size = 1000
        if total_peptides > 2000:
            peptides_to_cluster = random.sample(peptides, sample_size)
        else:
            peptides_to_cluster = peptides
        
        # Prepare data for scikit-learn
        embedding_matrix = np.array([p['embedding'] for p in peptides_to_cluster])
        
        # 4. K-means clustering
        n_clusters = 4 # This could be made dynamic
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embedding_matrix)

        # 5. Compute cluster statistics
        clusters_summary = []
        for i in range(n_clusters):
            # [Logic for calculating cluster means and stats...]
            cluster_indices = np.where(cluster_labels == i)[0]
            if not cluster_indices.any():
                continue

            cluster_peptides = [peptides_to_cluster[j] for j in cluster_indices]
            
            mean_length = np.mean([p['length'] for p in cluster_peptides])
            mean_hydrophobicity = np.mean([p['hydrophobicity'] for p in cluster_peptides])
            mean_intensity = np.mean([p['intensity'] for p in cluster_peptides])

            clusters_summary.append({
                "id": i,
                "size": len(cluster_peptides),
                "mean_length": round(mean_length, 2),
                "mean_hydrophobicity": round(mean_hydrophobicity, 4),
                "mean_intensity": float(mean_intensity)
            })

        # 6. Dimensionality Reduction for plotting
        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(embedding_matrix)
        
        embedding_2d_with_labels = [
            {"x": float(point[0]), "y": float(point[1]), "cluster_id": int(label)}
            for point, label in zip(embedding_2d, cluster_labels)
        ]

        # 7. Return JSON response
        return {
            "run_id": run_id,
            "total_peptides": total_peptides,
            "clusters": clusters_summary,
            "embedding_2d": embedding_2d_with_labels,
        }

    finally:
        con.close()

@app.get("/compare/{run_id_1}/{run_id_2}")
def compare_runs(
    run_id_1: str,
    run_id_2: str,
    user: User = Depends(current_active_user),
):
    """
    Compare two runs based on distinct peptide sequences.

    We use the `annotation_name` column from the `features` table,
    which is where peptide sequences are stored.
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        # Access check ― both runs must exist and belong to the current user
        run1 = db.get_run(con, run_id_1)
        run2 = db.get_run(con, run_id_2)

        if (
            run1 is None
            or run2 is None
            or str(run1.user_id) != str(user.id)
            or str(run2.user_id) != str(user.id)
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run(s) not found or not owned by current user.",
            )

        cur = con.cursor()

        # Fetch distinct peptide sequences for run 1
        cur.execute(
            "SELECT DISTINCT annotation_name FROM features WHERE run_id=?",
            (run_id_1,),
        )
        raw_1 = [row[0] for row in cur.fetchall()]

        # Fetch distinct peptide sequences for run 2
        cur.execute(
            "SELECT DISTINCT annotation_name FROM features WHERE run_id=?",
            (run_id_2,),
        )
        raw_2 = [row[0] for row in cur.fetchall()]

        # Normalize everything to JSON-safe strings and drop None/bytes
        def normalize_list(raw_list: List[Optional[str]]) -> List[str]:
            """Filters out None values and ensures all items are strings."""
            return [
                str(s) for s in raw_list
                if s is not None
            ]

        list_1 = normalize_list(raw_1)
        list_2 = normalize_list(raw_2)

        seqs_1 = set(list_1)
        seqs_2 = set(list_2)

        shared = seqs_1 & seqs_2
        unique_1 = seqs_1 - seqs_2
        unique_2 = seqs_2 - seqs_1

        union = seqs_1 | seqs_2
        jaccard = len(shared) / len(union) if len(union) > 0 else 0.0

        return {
            "run_1": run_id_1,
            "run_2": run_id_2,
            "stats": {
                "total_run_1": len(seqs_1),
                "total_run_2": len(seqs_2),
                "shared_count": len(shared),
                "unique_run_1_count": len(unique_1),
                "unique_run_2_count": len(unique_2),
                "jaccard_index": jaccard,
            },
            "shared_peptides": list(shared)[:100],
            "unique_run_1": list(unique_1)[:100],
            "unique_run_2": list(unique_2)[:100],
        }

    except HTTPException:
        # Re-raise known HTTP exceptions unchanged
        raise
    except Exception as e:
        logging.exception(
            f"Unexpected error while comparing runs {run_id_1} and {run_id_2}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Compare runs failed due to an internal error.",
        )
    finally:
        con.close()





@app.get("/export/embeddings/{run_id}", response_model=models.OmicsTokenExportV1)
def export_embeddings_v1(run_id: str, user: User = Depends(current_active_user)):
    """
    Export peptide embeddings for a specific run in the canonical OmicsToken v1 format.

    Returns:
        OmicsTokenExportV1:
            {
              "run_id": "...",
              "export_version": "omics_export_v1",
              "total_embeddings": N,
              "data": [ OmicsTokenV1, ... ]
            }
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        # 1. Verify run ownership
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        # 2. Fetch raw embeddings for this run
        raw_embeddings = db.get_peptide_embeddings(con, run_id)
        if not raw_embeddings:
            LOGGER.warning("Export attempted for run %s with no embeddings", run_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No peptide embeddings found for run '{run_id}'.",
            )

        # 3. Normalize into OmicsToken v1 records
        canonical_records = export.normalize_to_omics_token_v1(run_id, raw_embeddings)

        if not canonical_records:
            LOGGER.error(
                "Export normalization produced no records for run %s (raw count=%s)",
                run_id,
                len(raw_embeddings),
            )
            # This means every record failed validation; treat as server error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to normalize peptide embeddings for this run.",
            )

        # 4. Return standardised response
        return models.OmicsTokenExportV1(
            run_id=run_id,
            total_embeddings=len(canonical_records),
            data=canonical_records,
        )
    finally:
        con.close()



class ExportBundleRequest(BaseModel):
    run_ids: List[str]


@app.post("/export/bundle")
async def export_bundle(payload: ExportBundleRequest, user: User = Depends(current_active_user)):
    """
    Export multiple runs as a zip bundle of OmicsToken v1 JSON files.
    """
    if not payload.run_ids:
        raise HTTPException(status_code=400, detail="No run_ids provided.")

    con = db.get_db_connection(DATA_DIR)
    try:
        bundle_buffer = io.BytesIO()
        with zipfile.ZipFile(bundle_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for run_id in payload.run_ids:
                run = db.get_run(con, run_id)
                if run is None or str(run.user_id) != str(user.id):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="One or more runs cannot be exported. Check ownership and run status.",
                    )

                raw_embeddings = db.get_peptide_embeddings(con, run_id)
                if not raw_embeddings:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Run {run_id} is incomplete (no embeddings).",
                    )

                canonical_records = export.normalize_to_omics_token_v1(run_id, raw_embeddings)
                if not canonical_records:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to normalize embeddings for run {run_id}.",
                    )

                payload_json = models.OmicsTokenExportV1(
                    run_id=run_id,
                    total_embeddings=len(canonical_records),
                    data=canonical_records,
                ).model_dump()

                zf.writestr(f"{run_id}_omics_export_v1.json", json.dumps(payload_json, indent=2))

            bundle_meta = {
                "run_ids": payload.run_ids,
                "run_count": len(payload.run_ids),
                "bundle_created_at": datetime.utcnow().isoformat() + "Z",
            }
            zf.writestr("bundle.json", json.dumps(bundle_meta, indent=2))

        bundle_buffer.seek(0)
        filename = f"omics_bundle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.zip"
        return StreamingResponse(
            bundle_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    finally:
        con.close()

# --- API Tokens ---

@app.post("/auth/api-tokens")
async def create_api_token(label: Optional[str] = None, user: User = Depends(current_active_user)):
    token_payload = auth.create_api_token_for_user(str(user.id), label=label)
    if inspect.isawaitable(token_payload):
        token_payload = await token_payload
    created_at_raw = token_payload.get("created_at")
    if isinstance(created_at_raw, str):
        created_at_iso = created_at_raw
    elif created_at_raw is not None and hasattr(created_at_raw, "isoformat"):
        created_at_iso = created_at_raw.isoformat() + "Z"
    else:
        created_at_iso = None

    # Return plain token once; do not log it
    return {
        "token": token_payload["token"],
        "token_id": token_payload["token_id"],
        "label": token_payload.get("label"),
        "created_at": created_at_iso,
    }


@app.delete("/auth/api-tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_token(token_id: int, user: User = Depends(current_active_user)):
    deleted = await auth.delete_api_token_for_user(str(user.id), token_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Insight Report helpers ---

def _generate_report_from_fingerprint(data: dict) -> dict:
    total = data.get("total_peptides") or 0
    clusters = [c for c in data.get("clusters", []) if c and c.get("size") is not None]
    clusters_sorted = sorted(clusters, key=lambda c: c.get("size", 0), reverse=True)

    def pct(size: int) -> float:
        return round((size / total) * 100, 1) if total > 0 else 0.0

    def cluster_entry(c):
        if not c:
            return None
        return {
            "id": c.get("id"),
            "size": c.get("size"),
            "pct": pct(c.get("size", 0)),
            "mean_length": c.get("mean_length"),
            "mean_hydrophobicity": c.get("mean_hydrophobicity"),
            "mean_intensity": c.get("mean_intensity"),
        }

    largest = clusters_sorted[0] if clusters_sorted else None
    smallest = clusters_sorted[-1] if clusters_sorted else None

    avg_hydro = 0.0
    if clusters:
        avg_hydro = sum(c.get("mean_hydrophobicity") or 0 for c in clusters) / len(clusters)
    hydro_phrase = "skews hydrophobic" if avg_hydro > 0.5 else "appears balanced"

    largest_pct = pct(largest.get("size", 0)) if largest else 0
    smallest_pct = pct(smallest.get("size", 0)) if smallest else 0

    summary_parts = []
    if largest:
        summary_parts.append(f"Most peptides fall into Cluster {largest.get('id')} (~{largest_pct}%).")
    if smallest:
        summary_parts.append(f"Smallest cluster is {smallest.get('id')} (~{smallest_pct}%).")
    summary_parts.append(f"The run {hydro_phrase} overall.")

    return {
        "run_id": data.get("run_id"),
        "total_peptides": total,
        "total_clusters": len(clusters),
        "largest_cluster": cluster_entry(largest),
        "smallest_cluster": cluster_entry(smallest),
        "cluster_overview": [cluster_entry(c) for c in clusters if c],
        "plain_language_summary": " ".join(summary_parts),
    }


@app.get("/report/{run_id}")
def get_insight_report(run_id: str, user: User = Depends(current_active_user)):
    """
    Return a structured insight report derived from the fingerprint for a run.
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

        peptides = db.get_peptide_embeddings(con, run_id)
        if not peptides or len(peptides) < 4:
            raise HTTPException(status_code=400, detail="Run does not have enough peptides to build a fingerprint.")

        total_peptides = len(peptides)
        embedding_matrix = np.array([p['embedding'] for p in peptides])

        n_clusters = min(4, total_peptides)
        if n_clusters < 1:
            raise HTTPException(status_code=400, detail="Run does not have enough peptides to build a fingerprint.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embedding_matrix)

        clusters_summary = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if not cluster_indices.any():
                continue
            cluster_peptides = [peptides[j] for j in cluster_indices]
            mean_length = np.mean([p['length'] for p in cluster_peptides])
            mean_hydrophobicity = np.mean([p['hydrophobicity'] for p in cluster_peptides])
            mean_intensity = np.mean([p['intensity'] for p in cluster_peptides])
            clusters_summary.append({
                "id": i,
                "size": len(cluster_peptides),
                "mean_length": round(float(mean_length), 2),
                "mean_hydrophobicity": round(float(mean_hydrophobicity), 4),
                "mean_intensity": float(mean_intensity),
            })

        pca = PCA(n_components=2, random_state=42)
        embedding_2d = pca.fit_transform(embedding_matrix)
        embedding_2d_with_labels = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "cluster_id": int(label),
                "sequence": peptides[idx]["sequence"],
                "length": peptides[idx]["length"],
                "hydrophobicity": peptides[idx]["hydrophobicity"],
                "intensity": peptides[idx]["intensity"],
            }
            for idx, (point, label) in enumerate(zip(embedding_2d, cluster_labels))
        ]

        fingerprint_payload = {
            "run_id": run_id,
            "total_peptides": total_peptides,
            "clusters": clusters_summary,
            "embedding_2d": embedding_2d_with_labels,
        }

        return _generate_report_from_fingerprint(fingerprint_payload)
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.exception(f"Failed to build report for run {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report.")
    finally:
        con.close()


# --- Programmatic run upload ---

class UploadFeature(BaseModel):
    feature_id: str
    sequence: str
    intensity: Optional[float] = 0.0
    length: Optional[int] = 0
    charge: Optional[int] = 0
    hydrophobicity: Optional[float] = 0.0


class UploadRunRequest(BaseModel):
    name: Optional[str] = None
    metadata: Optional[dict] = None
    features: List[UploadFeature]
    auto_embed: Optional[bool] = False


@app.post("/api/runs")
def create_run_api(payload: UploadRunRequest, user: User = Depends(current_active_user_or_token)):
    if not payload.features:
        raise HTTPException(status_code=400, detail="Features are required.")

    run_id = f"RUN_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}"
    meta_dict = payload.metadata or {}
    if payload.name:
        meta_dict["name"] = payload.name

    con = db.get_db_connection(DATA_DIR)
    try:
        db.insert_run(con, run_id, meta_dict, user_id=str(user.id))

        for feat in payload.features:
            feature_model = models.Feature(
                feature_id=feat.feature_id,
                mz=0.0,
                rt_sec=None,
                intensity=feat.intensity or 0.0,
                adduct=None,
                polarity=None,
                peptide_sequence=feat.sequence,
                annotation_score=None,
                metadata={
                    "length": feat.length,
                    "charge": feat.charge,
                    "hydrophobicity": feat.hydrophobicity,
                },
            )
            db.insert_feature(con, run_id, feature_model)

        con.commit()

        if payload.auto_embed:
            try:
                job = queue_embed_run(run_id, str(user.id))
                LOGGER.info(
                    "Dispatched Celery embed_run_task for run %s (user %s) via API (task_id=%s, broker=%s)",
                    run_id,
                    user.id,
                    job.id,
                    config.CELERY_BROKER_URL,
                )
            except Exception as e:
                LOGGER.exception(
                    "Failed to dispatch Celery embed_run_task for run %s (user %s): %s",
                    run_id,
                    user.id,
                    e,
                )
                raise HTTPException(status_code=500, detail="Failed to queue embedding job.")

        return {
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "auto_embed": bool(payload.auto_embed),
        }
    except HTTPException:
        con.rollback()
        raise
    except Exception as e:
        con.rollback()
        LOGGER.exception("Failed to create run via API: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create run.")
    finally:
        con.close()

# --- Biophysics Logic ---
from collections import Counter

HYDROPHOBICITY_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def calculate_properties(sequence: str):
    if not sequence:
        return {"hydrophobicity": 0, "molecular_weight": 0, "charge": 0, "length": 0}
    
    # Filter for valid amino acids only for hydrophobicity calculation to avoid errors
    valid_seq = [aa for aa in sequence.upper() if aa in HYDROPHOBICITY_SCALE]
    hydro = sum(HYDROPHOBICITY_SCALE.get(aa, 0) for aa in valid_seq) / len(valid_seq) if valid_seq else 0
    
    mw = export.calculate_molecular_weight(sequence)
    pos = sum(sequence.upper().count(aa) for aa in ['K', 'R', 'H'])
    neg = sum(sequence.upper().count(aa) for aa in ['D', 'E'])
    return {"hydrophobicity": round(hydro, 2), "molecular_weight": mw, "charge": pos - neg, "length": len(sequence)}

@app.get("/dashboard-data/{run_id}")
def get_dashboard_data(run_id: str, user: User = Depends(current_active_user)):
    """
    Fetches features, calculates biophysical properties, and returns top 500 features.
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if run is None or str(run.user_id) != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found",
            )

        features = db.get_features_for_run(con, run_id)
        if not features:
            raise HTTPException(404, "Run not found or empty")
        
        # Sort by intensity descending
        # Sort by intensity descending (robust to None)
        features.sort(key=lambda x: (x.intensity is None, x.intensity or 0), reverse=True)
        top_features = features[:500]
        
        results = []
        for f in top_features:
            props = calculate_properties(f.peptide_sequence)
            
            # Extract stats from metadata if available
            meta = f.metadata or {}
            fc = meta.get("fold_change")
            qval = meta.get("q_value")
            pval = meta.get("p_value")
            
            results.append({
                "feature_id": f.feature_id,
                "sequence": f.peptide_sequence,
                "intensity": f.intensity,
                "properties": props,
                "stats": {
                    "fold_change": fc,
                    "q_value": qval,
                    "p_value": pval
                }
            })
            
        return {
            "run_id": run_id,
            "total_features": len(features),
            "data": results
        }
    finally:
        con.close()

# Mount static files (must be after all routes)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    # This block allows running the app directly with `python app.py`
    # It will start the server on http://127.0.0.1:8080, which matches
    # the desired port for development and testing.
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
