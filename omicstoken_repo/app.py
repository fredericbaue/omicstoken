from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import Request
from pydantic import BaseModel, Field
import uvicorn
import os
import shutil
import pandas as pd
from typing import List, Optional
import logging

import embeddings
from embeddings import peptide_to_vector, EMBEDDING_DIM
import db
import search
import models
import importers
import summarizer

# --- Configuration ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Immuno-Peptidomics MVP")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS (Optional, good for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs", status_code=307)

@app.post("/peptide/embed/{run_id}")
def embed(run_id: str):
    """Vectorize each peptide in a run and persist the vectors."""
    con = db.get_db_connection(DATA_DIR)
    
    # Verify run exists
    run = db.get_run(con, run_id)
    if not run:
        con.close()
        raise HTTPException(404, "run_id not found")

    features = db.get_features_for_run(con, run_id)
    
    count = 0
    for feat in features:
        # The core of our new engine: converting sequence to vector
        vec = peptide_to_vector(feat.peptide_sequence)

        # Skip if vectorization failed
        if vec is None or vec.shape[0] != EMBEDDING_DIM:
            print(f"Skipping embedding for feature {feat.feature_id} due to invalid sequence or vector.")
            continue

        db.insert_embedding(con, run_id, feat.feature_id, run.method or "", run.polarity or "", vec)
        count += 1

    con.commit()
    # After committing, rebuild the global search index
    search.rebuild_faiss_index(con, DATA_DIR)

    con.close()
    return {"run_id": run_id, "peptides_embedded": count}

@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Immuno-Engine - Upload</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 2rem; }
    form { display: block; max-width: 560px; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
    label { display:block; margin-top: 1rem; font-weight: 600; }
    input, select, textarea { width: 100%; padding: .6rem; margin-top: .4rem; }
    button { margin-top: 1rem; padding: .6rem 1rem; cursor: pointer; }
    .tip { color:#555; font-size:.9rem; margin-bottom: 1rem; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  </style>
</head>
  <h1>Upload Peptide Table</h1>
  <p class="tip">Accepted files: CSV/TSV from MaxQuant, DIA-NN, Spectronaut, or Generic.</p>

  <form action="/upload" method="post" enctype="multipart/form-data">
    <label>Peptide table file
      <input type="file" name="file" required />
    </label>

    <div class="row">
      <div>
        <label>Format
          <select name="format">
            <option value="auto">Auto-detect</option>
            <option value="maxquant">MaxQuant (peptides.txt)</option>
            <option value="diann">DIA-NN (report.tsv)</option>
            <option value="spectronaut">Spectronaut</option>
            <option value="generic">Generic CSV</option>
          </select>
        </label>
      </div>
      <div>
        <label>Run ID (optional)
          <input type="text" name="run_id" placeholder="e.g., RUN_2025_11_03_A" />
        </label>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Instrument (optional)
          <input type="text" name="instrument" placeholder="e.g., Q Exactive" />
        </label>
      </div>
      <div>
        <label>Method (optional)
          <input type="text" name="method" placeholder="e.g., HILIC or RP" />
        </label>
      </div>
    </div>

    <button type="submit">Upload & Ingest</button>
  </form>

  <p style="margin-top:2rem;">
    Or use the API docs: <a href="/docs">/docs</a>
  </p>
</body>
</html>
    """

@app.post("/upload", response_class=HTMLResponse, include_in_schema=False)
async def upload_handler(request: Request, background_tasks: BackgroundTasks,
                         file: UploadFile = File(...),
                         run_id: str = Form(""),
                         instrument: str = Form(""),
                         method: str = Form(""),
                         format: str = Form("auto")):
    """
    Handles file uploads, detects format, ingests data, and triggers embedding.
    """
    # --- Ingest Logic ---
    name = file.filename or "upload.csv"
    sep = "\t" if name.endswith(".tsv") or name.endswith(".txt") else ","
    await file.seek(0)
    
    try:
        df = pd.read_csv(file.file, sep=sep)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {e}")

    # Use the importers module to parse the dataframe
    try:
        features = importers.parse_upload(df, fmt=format)
    except Exception as e:
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
    db.insert_run(con, run_id_final, meta_dict)

    for feat in features:
        db.insert_feature(con, run_id_final, feat)
        
    con.commit()
    con.close()
    rows_ingested = len(features)

    # --- Background Tasks ---
    def _background_work(run_id_to_process: str):
        # 1. Embed
        embed_result = embed(run_id=run_id_to_process)
        print(f"Background: Embedded {embed_result['peptides_embedded']} peptides.")
        # 2. Summarize (Pre-compute? Or just let user request it later? For now, just print)
        print(f"Background: Ready for summarization.")

    background_tasks.add_task(_background_work, run_id_final)

    # --- Success Response ---
    first_feature_id = features[0].feature_id if features else None
    search_link_html = ""
    if first_feature_id:
        search_url = f"/peptide/search/{run_id_final}/{first_feature_id}"
        search_link_html = f'<li><a href="{search_url}" target="_blank">Test search for first peptide: "{first_feature_id}"</a></li>'

    return HTMLResponse(f"""
        <h2>Upload complete</h2>
        <p>Run ID: <b>{run_id_final}</b></p>
        <p>✅ Ingested <b>{rows_ingested}</b> peptides using format <b>{format}</b>.</p>
        <p>Embedding running in background...</p>
        <p>
          Next steps:
          <ul>
            {search_link_html}
            <li>
                <form action="/summary/run/{run_id_final}" method="post" target="_blank" style="display:inline;">
                    <button type="submit" style="background:none; border:none; text-decoration:underline; color:blue; cursor:pointer; padding:0;">
                        Generate Summary for this Run
                    </button>
                </form>
            </li>
            <li><a href="/docs" target="_blank">API Docs</a></li>
          </ul>
        </p>
        <p><a href="/upload">Upload another file</a></p>
    """, status_code=200)

@app.get("/peptide/search/{run_id}/{feature_id}")
def similar(run_id: str, feature_id: str, k: int = 5):
    """Find peptides with similar biophysical properties."""
    con = db.get_db_connection(DATA_DIR)
    try:
        return search.search_similar_peptides(con, DATA_DIR, run_id, feature_id, k)
    except HTTPException as e:
        raise e
    finally:
        con.close()

@app.post("/summary/run/{run_id}")
def get_run_summary(run_id: str):
    """
    Generates a text summary of the run using the Summarization Engine.
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        summary = summarizer.generate_summary(run_id, con)
        return summary
    finally:
        con.close()

@app.get("/runs")
def list_runs():
    """List all runs with summary statistics."""
    con = db.get_db_connection(DATA_DIR)
    try:
        return db.get_run_summaries(con)
    finally:
        con.close()

@app.get("/runs/{run_id}")
def get_run_details(run_id: str):
    """Get detailed information about a specific run."""
    con = db.get_db_connection(DATA_DIR)
    try:
        run = db.get_run(con, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
            
        # Get counts
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM features WHERE run_id=?", (run_id,))
        n_features = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE run_id=?", (run_id,))
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

@app.get("/compare/{run_id_1}/{run_id_2}")
def compare_runs(run_id_1: str, run_id_2: str):
    """
    Compares two runs to find shared and unique peptides.
    """
    con = db.get_db_connection(DATA_DIR)
    try:
        # Fetch features for both runs
        features_1 = db.get_features_for_run(con, run_id_1)
        features_2 = db.get_features_for_run(con, run_id_2)
        
        if not features_1:
            raise HTTPException(404, f"Run {run_id_1} not found or empty")
        if not features_2:
            raise HTTPException(404, f"Run {run_id_2} not found or empty")
            
        # Extract sequences
        seqs_1 = set(f.peptide_sequence for f in features_1)
        seqs_2 = set(f.peptide_sequence for f in features_2)
        
        # Calculate intersections and differences
        shared = seqs_1.intersection(seqs_2)
        unique_1 = seqs_1 - seqs_2
        unique_2 = seqs_2 - seqs_1
        
        return {
            "run_1": run_id_1,
            "run_2": run_id_2,
            "stats": {
                "total_run_1": len(seqs_1),
                "total_run_2": len(seqs_2),
                "shared_count": len(shared),
                "unique_run_1_count": len(unique_1),
                "unique_run_2_count": len(unique_2),
                "jaccard_index": len(shared) / len(seqs_1.union(seqs_2)) if seqs_1 or seqs_2 else 0
            },
            "shared_peptides": list(shared)[:100], # Limit for display
            "unique_run_1": list(unique_1)[:100],
            "unique_run_2": list(unique_2)[:100]
        }
    finally:
        con.close()
