from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import RedirectResponse
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse


import pandas as pd
import numpy as np
import json, os, sqlite3, pickle
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(APP_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "metabo.sqlite")
os.makedirs(DATA_DIR, exist_ok=True)

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
    mz_idx = max(0, int((feat.mz - 50.0) / 0.1))
    # RT bucket: 0–60 min, 30 s buckets (2 per minute)
    rt_bucket = int(((feat.rt_sec or 0.0) / 60.0) * 2)
    # Intensity band: log10 scaled, clamped 0..8
    i_band = int(np.clip(np.log10(max(feat.intensity, 1.0)), 0, 8))
    tokens = [
        f"MZ_{mz_idx}",
        f"RT_{rt_bucket}",
        f"I_{i_band}",
        f"POL_{(feat.polarity or run_meta.get('polarity','NA')).upper()}",
    ]
    if feat.adduct:
        tokens.append("ADDUCT_" + feat.adduct.replace("+", "PLUS").replace("-", "MINUS"))
    if run_meta.get("instrument"):
        tokens.append("INSTR_" + str(run_meta["instrument"]).replace(" ", "_")[:24])
    if run_meta.get("method"):
        tokens.append("METHOD_" + str(run_meta["method"]).upper())
    return tokens

# -------------------- Feature → Vector --------------------
def feature_to_vector(feat: Feature) -> np.ndarray:
    """Turn simple properties into a 128-D vector (cosine-normalized)."""
    base = np.array(
        [feat.mz, (feat.rt_sec or 0.0), np.log1p(feat.intensity)], dtype=np.float32
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

# -------------------- API Endpoints --------------------
@app.get("/health")
def health():
    return {"ok": True, "schema": SCHEMA_VERSION}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)


@app.post("/metabo/ingest")
def ingest(
    file: UploadFile = File(...),
    meta: str = Form("{}")  # meta arrives as form text; we'll parse JSON ourselves
):
    """
    Upload a feature table (CSV/TSV) + a JSON 'meta' string in the same form.
    Example meta JSON:
    {
      "run_id": "RUN_A",
      "instrument": "Q Exactive",
      "method": "HILIC",
      "polarity": "POS"
    }
    """
    # Parse CSV/TSV
    name = file.filename or "upload.csv"
    sep = "\t" if name.endswith(".tsv") else ","
    df = pd.read_csv(file.file, sep=sep)

    # Parse meta JSON from form text
    try:
        meta_dict = json.loads(meta) if meta else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "meta must be valid JSON text")

    # Validate minimal columns
    required = ["feature_id", "mz", "intensity"]
    for col in required:
        if col not in df.columns:
            raise HTTPException(400, f"Missing required column: {col}")

    # Pull run context
    run_id = meta_dict.get("run_id") or f"RUN_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    instrument = meta_dict.get("instrument")
    method     = meta_dict.get("method")
    polarity   = meta_dict.get("polarity")

    # Store in SQLite
    con = db()
    con.execute(
        "INSERT OR REPLACE INTO runs(run_id,instrument,method,polarity,schema_version,meta_json) VALUES(?,?,?,?,?,?)",
        (run_id, instrument, method, polarity, SCHEMA_VERSION, json.dumps(meta_dict)),
    )

    for _, r in df.iterrows():
        con.execute(
            """INSERT OR REPLACE INTO features(run_id,feature_id,mz,rt_sec,intensity,adduct,polarity,annotation_name,annotation_score,tokens_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                str(r.get("feature_id")),
                float(r.get("mz")),
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
    return {"run_id": run_id, "rows": int(len(df))}

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
            mz=float(mz),
            rt_sec=rt if rt is not None else 0.0,
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
    form { max-width: 560px; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
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
    # Build the meta dict from simple fields
    meta_dict = {
        "run_id": run_id.strip() or None,
        "instrument": instrument.strip() or None,
        "method": method.strip() or None,
        "polarity": polarity.strip() or None
    }

    # Rewind file pointer before reusing the UploadFile (safety)
    await file.seek(0)

    # Call the same logic as /metabo/ingest by creating a tiny wrapper that
    # reuses your existing function (we’re going to mimic a form post).
    # Since /metabo/ingest expects meta as JSON text in a form field, we’ll
    # pass meta as a JSON string and the same UploadFile object.
    try:
        # Manually parse the CSV with the same logic as ingest()
        name = file.filename or "upload.csv"
        sep = "\t" if name.endswith(".tsv") else ","
        import pandas as pd, json

        df = pd.read_csv(file.file, sep=sep)
        required = ["feature_id", "mz", "intensity"]
        for col in required:
            if col not in df.columns:
                return HTMLResponse(f"<h3>Upload failed</h3><p>Missing required column: <b>{col}</b></p><p><a href='/upload'>Back</a></p>", status_code=400)

        # Use the same storage code as /metabo/ingest
        run_id_final = meta_dict.get("run_id") or f"RUN_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
        instrument_final = meta_dict.get("instrument")
        method_final = meta_dict.get("method")
        polarity_final = meta_dict.get("polarity")

        con = db()
        con.execute(
            "INSERT OR REPLACE INTO runs(run_id,instrument,method,polarity,schema_version,meta_json) VALUES(?,?,?,?,?,?)",
            (run_id_final, instrument_final, method_final, polarity_final, SCHEMA_VERSION, json.dumps(meta_dict)),
        )
        for _, r in df.iterrows():
            con.execute(
                """INSERT OR REPLACE INTO features(run_id,feature_id,mz,rt_sec,intensity,adduct,polarity,annotation_name,annotation_score,tokens_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id_final,
                    str(r.get("feature_id")),
                    float(r.get("mz")),
                    float(r.get("rt_sec", np.nan)) if not pd.isna(r.get("rt_sec", np.nan)) else None,
                    float(r.get("intensity")),
                    r.get("adduct"),
                    r.get("polarity"),
                    r.get("annotation_name"),
                    float(r.get("annotation_score", np.nan)) if not pd.isna(r.get("annotation_score", np.nan)) else None,
                    None,
                ),
            )
        con.commit(); con.close()

        # Friendly success page with quick links
        return HTMLResponse(f"""
            <h2>Upload complete</h2>
            <p>Run ID: <b>{run_id_final}</b></p>
            <p>Rows ingested: <b>{len(df)}</b></p>
            <p>
              Next steps:
              <ul>
                <li><a href="/docs#/default/embed_metabo_embed__run_id__post">Embed this run</a> (enter <code>{run_id_final}</code>)</li>
                <li><a href="/docs#/default/similar_metabo_search__run_id___feature_id__get">Search neighbors</a></li>
              </ul>
            </p>
            <p><a href="/upload">Upload another file</a></p>
        """, status_code=200)

    except Exception as e:
        return HTMLResponse(f"<h3>Upload failed</h3><pre>{e}</pre><p><a href='/upload'>Back</a></p>", status_code=400)


@app.get("/metabo/search/{run_id}/{feature_id}")
def similar(run_id: str, feature_id: str, k: int = 5):
    """Find nearest neighbors of a feature using cosine similarity."""
    con = db()
    cur = con.cursor()
    cur.execute("SELECT method, polarity FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(404, "run_id not found")
    method, polarity = row

    cur.execute(
        "SELECT vec_json FROM embeddings WHERE run_id=? AND feature_id=?",
        (run_id, feature_id),
    )
    row = cur.fetchone()
    if not row:
        con.close()
        raise HTTPException(404, "embedding not found (did you run /embed?)")
    q = np.array(json.loads(row[0]), dtype=np.float32).reshape(1, -1)

    cur.execute(
        "SELECT run_id, feature_id, vec_json FROM embeddings WHERE method=? AND polarity=?",
        (method or "", polarity or ""),
    )
    rows = cur.fetchall()
    con.close()

    if not rows:
        raise HTTPException(400, "no embeddings to search")

    ids = []
    X = []
    for rid, fid, vjson in rows:
        ids.append((rid, fid))
        X.append(json.loads(vjson))
    X = np.array(X, dtype=np.float32)

    sims = cosine_similarity(q, X)[0]
    order = np.argsort(-sims)[:max(1, k)]
    out = [{"run_id": ids[i][0], "feature_id": ids[i][1], "similarity": float(sims[i])} for i in order]
    return {"query": {"run_id": run_id, "feature_id": feature_id}, "neighbors": out}
