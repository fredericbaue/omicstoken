# Immuno-Engine MVP: Proteomics Peptide Similarity Engine

This is a Minimal Viable Product (MVP) for a peptide similarity engine, designed for proteomics research. It allows users to upload peptide data, which is then embedded into a high-dimensional vector space using a pre-trained protein language model (ProtTrans-T5-XL-U50). These embeddings are indexed using FAISS for fast nearest-neighbor searches, enabling the discovery of peptides with similar biophysical properties.

## Features

- **Peptide Data Ingestion**: Upload peptide data via CSV.
- **Protein Language Model Embedding**: Uses `ProtTransT5XLU50Embedder` from `bio_embeddings` to convert peptide sequences into numerical vectors.
- **FAISS Indexing**: Efficiently indexes peptide embeddings for rapid similarity searches.
- **Similarity Search API**: An endpoint to query for peptides similar to a given query peptide.
- **Modular Architecture**: Cleanly separated concerns into `app.py`, `db.py`, `models.py`, `embeddings.py`, and `search.py`.
- **Asynchronous Indexing**: FAISS index rebuilding runs in the background to avoid blocking the API.

## Getting Started

### Prerequisites

Before running the application, ensure you have Python 3.8+ and the following libraries installed:

```bash
pip install fastapi uvicorn pandas numpy faiss-cpu bio-embeddings[all] pydantic jinja2
```

Note: `faiss-cpu` is used for CPU-only FAISS. If you have a compatible GPU, you might install `faiss-gpu` instead. `bio-embeddings[all]` might install many dependencies; you can selectively install `bio-embeddings[prottrans_t5_xl_u50]` if you only need that model.

### Running the App Locally

1.  **Navigate to the project directory**:
    ```bash
    cd proteomics_mvp
    ```
2.  **Run the FastAPI application**:
    ```bash
    uvicorn app:app --reload
    ```
    The `--reload` flag is useful for development as it restarts the server on code changes.

The application will be accessible at `http://127.0.0.1:8000`.

## Usage

### 1. Upload Peptide Data

You can upload a CSV file containing your peptide data through a simple web interface or directly via the API documentation.

-   **Web UI**: Open `http://127.0.0.1:8000/upload` in your browser.
-   **API Docs**: Go to `http://127.0.0.1:8000/docs` and use the `/upload` POST endpoint.

#### Input CSV Requirements

Your CSV file must contain at least the following columns:

-   `feature_id`: A unique identifier for each peptide within a run (e.g., "peptide_1", "P00123_F1").
-   `peptide_sequence`: The amino acid sequence of the peptide (e.g., "SEQUENCE"). This will be used for embedding.
-   `intensity`: A numerical value representing the peptide's intensity.

Optional columns: `mz`, `rt_sec`, `adduct`, `polarity`, `annotation_score`.

**Example `peptides.csv`:**
```csv
feature_id,peptide_sequence,mz,rt_sec,intensity,adduct,polarity
peptide_A,SEQUENCE,1234.56,123.4,100000,M+H,+
peptide_B,ANOTHERSEQ,987.65,234.5,50000,M+H,+
peptide_C,SIMILARSEQ,1235.00,124.0,95000,M+H,+
```

After uploading, the peptides will be ingested into the SQLite database, embedded using ProtTrans-T5, and the FAISS index will be rebuilt in the background.

### 2. Query Similar Peptides

Once the data is uploaded and indexed, you can query for peptides similar to a specific peptide.

-   **API Endpoint**: `GET /peptide/search/{run_id}/{feature_id}?k={number_of_neighbors}`
-   **Example**: If you uploaded a run with `run_id="my_experiment"` and want to find peptides similar to `feature_id="peptide_A"`, you would query:
    `http://127.0.0.1:8000/peptide/search/my_experiment/peptide_A?k=5`

This will return a JSON response containing the query peptide's details and a list of its `k` most similar neighbors, including their similarity scores.

## API Documentation

Access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## Project Structure

```
proteomics_mvp/
├── app.py          # FastAPI application, routes, and orchestration
├── db.py           # SQLite database connection, schema, and CRUD operations
├── embeddings.py   # ProtTrans model loading and peptide embedding logic
├── models.py       # Pydantic data models for API requests/responses and internal data
├── search.py       # FAISS index management and similarity search logic
├── static/         # Static files (e.g., CSS, JS, favicon)
├── templates/      # HTML templates for web UI (e.g., upload form)
└── README.md       # This file
```