from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse


import pandas as pd
import numpy as np
import json, os, sqlite3, pickle
import faiss
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(APP_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "metabo.sqlite")
os.makedirs(DATA_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "features.faiss")
FAISS_IDS_PATH = os.path.join(DATA_DIR, "features_ids.json")

SCHEMA_VERSION = "metabo-0.1.0"

# -------------------- Data Models --------------------
class Spectrum(BaseModel):
    mz_array: List[float]
    intensity_array: List[float]

class Feature(BaseModel):
    feature_id: str
    mz: float
    rt_sec: Optional[float] = None
    intensity: float
    adduct: Optional[str] = None
    polarity: Optional[str] = None
    annotation_name: Optional[str] = None
    annotation_score: Optional[float] = None
    msms_spectrum: Optional[Spectrum] = None
    tokens: Optional[List[str]] = None
    fragment_embedding: Optional[List[float]] = None

class Run(BaseModel):
    run_id: str
    instrument: Optional[str] = None
    method: Optional[str] = None
    polarity: Optional[str] = None
    schema_version: str = Field(default=SCHEMA_VERSION)
    features: List[Feature]
    meta: Dict[str, Any] = {}

app = FastAPI(title="Metabo MVP (Windows)", version="0.1.3")

# Mount static files (like favicon.ico)
# This must come AFTER `app` is created.
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- Database Helpers --------------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS runs(
        run_id TEXT PRIMARY KEY,
        instrument TEXT, method TEXT, polarity TEXT,
        schema_version TEXT, meta_json TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS features(
        run_id TEXT, feature_id TEXT,
        mz REAL, rt_sec REAL, intensity REAL,
        adduct TEXT, polarity TEXT,
        annotation_name TEXT, annotation_score REAL,
        tokens_json TEXT,
        PRIMARY KEY(run_id, feature_id)
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS embeddings(
        run_id TEXT, feature_id TEXT,
        method TEXT, polarity TEXT,
        vec_json TEXT,
        PRIMARY KEY(run_id, feature_id)
    )""")
    return con

# -------------------- Tokenization --------------------
def tokenize_feature(feat: Feature, run_meta: Dict[str, Any]):
    # m/z bin: 50–1200 with 0.1 Da bins
    mz_idx = max(0, int(((feat.mz or 0.0) - 50.0) / 0.1))
    # RT bucket: 0–60 min, 30 s buckets (2 per minute)
    rt_bucket = int(((feat.rt_sec or 0.0) / 60.0) * 2)
    # Intensity band: log10 scaled, clamped 0..8
    i_band = int(np.clip(np.log10(max(feat.intensity, 1.0)), 0, 8))
    tokens = [
        f"MZ_{mz_idx}",
        f"RT_{rt_bucket}",
        f"I_{i_band}",
        f"POL_{str(feat.polarity or run_meta.get('polarity') or 'NA').upper()}",
    ]
    if feat.adduct:
        tokens.append("ADDUCT_" + feat.adduct.replace("+", "PLUS").replace("-", "MINUS"))
    if run_meta.get("instrument"):
        tokens.append("INSTR_" + str(run_meta["instrument"]).replace(" ", "_")[:24])
    # Make tokenization robust to missing meta values
    tokens.append("METHOD_" + str(run_meta.get("method") or "NA").upper())
    return tokens

# -------------------- Feature → Vector --------------------
def feature_to_vector(feat: Feature) -> np.ndarray:
    """Turn simple properties into a 128-D vector (cosine-normalized)."""
    base = np.array(
        [(feat.mz or 0.0), (feat.rt_sec or 0.0), np.log1p(feat.intensity)], dtype=np.float32
    )
    base = base / (np.linalg.norm(base) + 1e-8)
    z = np.zeros((128,), dtype=np.float32)
    z[:3] = base
    return z

# (Optional future) load a PCA model for MS/MS -> 128D embeddings
def load_pca() -> PCA:
    path = os.path.join(DATA_DIR, "pca_1150_to_128.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return PCA(n_components=128, random_state=42)

# -------------------- Vector Indexing (FAISS) --------------------
def rebuild_faiss_index():
    """
    Queries all embeddings from the DB, builds a FAISS index, and saves it.
    This is slow and should be run as a background task in a real app.
    """
    print("Rebuilding FAISS index...")
    con = db()
    # For this MVP, we rebuild the index for ALL methods/polarities together.
    # A real app might build separate indexes per method/polarity.
    rows = con.execute("SELECT run_id, feature_id, vec_json FROM embeddings").fetchall()
    con.close()

    if not rows:
        print("No embeddings found to build index.")
        return 0

    ids = [(row[0], row[1]) for row in rows]
    vectors = np.array([json.loads(row[2]) for row in rows], dtype=np.float32)

    d = vectors.shape[1]  # vector dimension
    index = faiss.IndexFlatL2(d)  # Using L2 distance; for cosine, we need to normalize
    faiss.normalize_L2(vectors) # Normalize vectors to use L2 as cosine similarity
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "w") as f:
        json.dump(ids, f)
    print(f"✅ FAISS index rebuilt with {index.ntotal} vectors.")
    return index.ntotal

# -------------------- API Endpoints --------------------
@app.get("/health")
def health():
    return {"ok": True, "schema": SCHEMA_VERSION}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)

@app.post("/metabo/embed/{run_id}")
def embed(run_id: str):
    """Tokenize and vectorize each feature, then persist the vectors."""
    con = db()
    cur = con.cursor()
    cur.execute("SELECT instrument, method, polarity, meta_json FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(404, "run_id not found")
    instrument, method, polarity, meta_json = row
    run_meta = json.loads(meta_json or "{}")

    cur.execute(
        "SELECT feature_id,mz,rt_sec,intensity,adduct,polarity FROM features WHERE run_id=?",
        (run_id,),
    )
    feats = cur.fetchall()

    count = 0
    for (fid, mz, rt, inten, adduct, pol) in feats:
        f = Feature(
            feature_id=fid,
            mz=float(mz) if mz is not None else 0.0,
            rt_sec=float(rt) if rt is not None else 0.0,
            intensity=float(inten),
            adduct=adduct,
            polarity=pol,
        )
        tokens = tokenize_feature(
            f,
            {"instrument": instrument, "method": method, "polarity": polarity, **run_meta},
        )
        vec = feature_to_vector(f)

        cur.execute(
            "UPDATE features SET tokens_json=? WHERE run_id=? AND feature_id=?",
            (json.dumps(tokens), run_id, fid),
        )
        cur.execute(
            """INSERT OR REPLACE INTO embeddings(run_id,feature_id,method,polarity,vec_json)
               VALUES(?,?,?,?,?)""",
            (run_id, fid, method or "", polarity or "", json.dumps(vec.tolist())),
        )
        count += 1

    con.commit()
    # After committing, rebuild the global search index
    rebuild_faiss_index()

    con.close()
    return {"run_id": run_id, "features_embedded": count, "method": method, "polarity": polarity}

@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page():
    # A tiny HTML form (no frameworks) that posts file + fields.
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Metabo MVP - Upload</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 2rem; }
    form { display: block; max-width: 560px; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
    label { display:block; margin-top: 1rem; font-weight: 600; }
    input, select, textarea { width: 100%; padding: .6rem; margin-top: .4rem; }
    button { margin-top: 1rem; padding: .6rem 1rem; cursor: pointer; }
    .tip { color:#555; font-size:.9rem; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  </style>
</head>
<body>
  <h1>Upload Metabolomics Feature Table</h1>
  <p class="tip">Accepted files: CSV or TSV with columns: <code>feature_id, mz, intensity</code> (optional: <code>rt_sec, adduct, polarity, annotation_name, annotation_score</code>)</p>

  <form action="/upload" method="post" enctype="multipart/form-data">
    <label>Feature table file (CSV/TSV)
      <input type="file" name="file" required />
    </label>

    <div class="row">
      <div>
        <label>Run ID (optional)
          <input type="text" name="run_id" placeholder="e.g., RUN_2025_11_03_A" />
        </label>
      </div>
      <div>
        <label>Instrument (optional)
          <input type="text" name="instrument" placeholder="e.g., Q Exactive" />
        </label>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Method (optional)
          <input type="text" name="method" placeholder="e.g., HILIC or RP" />
        </label>
      </div>
      <div>
        <label>Polarity (optional)
          <select name="polarity">
            <option value="">(none)</option>
            <option value="POS">POS</option>
            <option value="NEG">NEG</option>
          </select>
        </label>
      </div>
    </div>

    <p class="tip">Tip: If you’re unsure, leave fields blank—defaults are okay.</p>
    <button type="submit">Upload & Ingest</button>
  </form>

  <p style="margin-top:2rem;">
    Or use the API docs: <a href="/docs">/docs</a>
  </p>
</body>
</html>
    """

@app.post("/upload", response_class=HTMLResponse, include_in_schema=False)
async def upload_handler(request: Request,
                         file: UploadFile = File(...),
                         run_id: str = Form(""),
                         instrument: str = Form(""),
                         method: str = Form(""),
                         polarity: str = Form("")):
    """
    This converts the simple form fields into the JSON 'meta' your /metabo/ingest
    endpoint expects, then calls the same ingest logic under the hood.
    """
    # --- Ingest Logic ---
    # Parse CSV/TSV
    name = file.filename or "upload.csv"
    sep = "\t" if name.endswith(".tsv") else ","
    await file.seek(0)
    df = pd.read_csv(file.file, sep=sep)

    # Build the meta dict from simple fields
    meta_dict = {
        "run_id": run_id.strip() or None,
        "instrument": instrument.strip() or None,
        "method": method.strip() or None,
        "polarity": polarity.strip() or None,
    }
    run_id_final = meta_dict.get("run_id") or f"RUN_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Store in SQLite
    con = db()
    con.execute(
        "INSERT OR REPLACE INTO runs(run_id,instrument,method,polarity,schema_version,meta_json) VALUES(?,?,?,?,?,?)",
        (run_id_final, meta_dict.get("instrument"), meta_dict.get("method"), meta_dict.get("polarity"), SCHEMA_VERSION, json.dumps(meta_dict)),
    )

    for _, r in df.iterrows():
        con.execute(
            """INSERT OR REPLACE INTO features(run_id,feature_id,mz,rt_sec,intensity,adduct,polarity,annotation_name,annotation_score,tokens_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id_final,
                str(r.get("feature_id")),
                float(r.get("mz", np.nan))
                if not pd.isna(r.get("mz", np.nan))
                else None,
                float(r.get("rt_sec", np.nan))
                if not pd.isna(r.get("rt_sec", np.nan))
                else None,
                float(r.get("intensity")),
                r.get("adduct"),
                r.get("polarity"),
                r.get("annotation_name"),
                float(r.get("annotation_score", np.nan))
                if not pd.isna(r.get("annotation_score", np.nan))
else None,
                None,
            ),
        )
    con.commit()
    con.close()
    rows_ingested = len(df)
    # --- End Ingest Logic ---

    # Rewind file pointer before reusing the UploadFile (safety)
    await file.seek(0)

    try:
        # --- NEW: Automatically trigger embedding ---
        embed_result = embed(run_id=run_id_final)
        features_embedded = embed_result["features_embedded"]

        # --- Create a helpful link for the user ---
        # Get the first feature_id from the uploaded file to create a sample search link.
        # We need to rewind the file one last time to read its contents.
        await file.seek(0)
        df_for_link = pd.read_csv(file.file, sep="\t" if (file.filename or "").endswith(".tsv") else ",")
        first_feature_id = df_for_link["feature_id"].iloc[0] if not df_for_link.empty else None

        search_link_html = ""
        if first_feature_id:
            search_url = f"/metabo/search/{run_id_final}/{first_feature_id}"
            search_link_html = f'<li><a href="{search_url}" target="_blank">Test search for first feature: "{first_feature_id}"</a></li>'

        # Friendly success page with quick links
        return HTMLResponse(f"""
            <h2>Upload complete</h2>
            <p>Run ID: <b>{run_id_final}</b></p>
            <p>✅ Rows ingested: <b>{rows_ingested}</b></p>
            <p>✅ Features embedded: <b>{features_embedded}</b></p>
            <p>
              Next steps:
              <ul>
                {search_link_html}
                <li><a href="/docs#/default/similar_metabo_search__run_id___feature_id__get" target="_blank">Go to API docs for custom search</a></li>
              </ul>
            </p>
            <p><a href="/upload">Upload another file</a></p>
        """, status_code=200)

    except Exception as e:
        return HTMLResponse(f"<h3>Upload failed</h3><pre>{e}</pre><p><a href='/upload'>Back</a></p>", status_code=400)


@app.get("/metabo/search/{run_id}/{feature_id}")
def similar(run_id: str, feature_id: str, k: int = 5):
    """Find nearest neighbors of a feature using cosine similarity."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(FAISS_IDS_PATH):
        raise HTTPException(404, "FAISS index not found. Please run the /embed endpoint first.")

    # Load the FAISS index and the ID mapping
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_IDS_PATH, "r") as f:
        ids_map = json.load(f)

    # The new IDs map is just a list of [run_id, feature_id]
    # We need to find the index of our query feature
    try:
        # Find the query vector from the database first
        con = db()
        cur = con.cursor()
        cur.execute(
            "SELECT vec_json FROM embeddings WHERE run_id=? AND feature_id=?",
            (run_id, feature_id),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            raise HTTPException(404, "embedding not found (did you run /embed?)")
        query_vector = np.array(json.loads(row[0]), dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector) # Normalize the query vector as well

    except ValueError:
        raise HTTPException(404, f"Feature '{feature_id}' not found in the search index for run '{run_id}'.")

    # Unlike before, we don't need to query the DB for the vector, it's in the index!
    con = db()
    cur = con.cursor()
    # We need to get the original properties of the query feature to display them
    cur.execute(
        "SELECT mz, rt_sec, intensity FROM features WHERE run_id=? AND feature_id=? LIMIT 1",
        (run_id, feature_id),
    )
    query_feat_props = cur.fetchone()
    
    con.close() # Close connection after fetching query vector

    if not query_feat_props:
        raise HTTPException(404, "embedding not found (did you run /embed?)")

    # Prepare the query vector
    # The vector from FAISS is already normalized.
    
    # Search the index
    # FAISS L2 distance on normalized vectors is related to cosine similarity.
    # D^2 = 2 - 2 * cos(sim), so smaller distance is higher similarity.
    distances, indices = index.search(query_vector, k)

    # Enrich results with feature details
    out = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue # Should not happen in a simple index

        neighbor_run_id, neighbor_feature_id = ids_map[idx]
        dist = distances[0][i]
        # Convert L2 distance back to cosine similarity
        similarity = 1.0 - (dist**2) / 2.0

        # Re-open connection to fetch details. In a real app, use a connection pool or Depends().
        with sqlite3.connect(DB_PATH) as con_details:
            cur_details = con_details.cursor()
            cur_details.execute(
                "SELECT mz, rt_sec, intensity, annotation_name FROM features WHERE run_id=? AND feature_id=?",
                (neighbor_run_id, neighbor_feature_id)
            )
            feat_row = cur_details.fetchone()

        # For profile search, intensity of a single point is less meaningful.
        # We can show the intensity from the first sample as an example.
        neighbor_intensity = feat_row[2] if feat_row else None

        out.append({
            "run_id": neighbor_run_id,
            "feature_id": neighbor_feature_id,
            "similarity": float(similarity),
            "mz": feat_row[0] if feat_row else "N/A",
            "rt_sec": feat_row[1] if feat_row else "N/A",
            "intensity": neighbor_intensity,
            "annotation_name": feat_row[3] if feat_row else None,
        })

    query_props_dict = {"mz": query_feat_props[0] or "N/A", "rt_sec": query_feat_props[1] or "N/A", "intensity_example": query_feat_props[2]}

    return {"query": {"run_id": run_id, "feature_id": feature_id, **query_props_dict}, "neighbors": out}
