# Demo Readiness Notes

## What was fixed or verified
- Upload now logs ingestion per run and dispatches `embed_run_task` with task IDs; embedding is idempotent (existing embeddings for a run are cleared before re-embedding) and logs per-run progress.
- Celery pipeline logs include run context; `embed_run_task` fans out to `rebuild_index_task` and `generate_summary_task`, and FAISS rebuild logs the triggering run.
- `/summary/run/{run_id}` prefers Celery execution (with a bounded wait) and falls back inline if the worker is unavailable; auth failures are logged with method/path and correlation IDs.
- Export of OmicsToken v1 uses accurate molecular weights, logs/returns clear errors when a run has no embeddings or normalization fails.
- Dashboard analysis buttons are disabled and labeled “coming soon” to avoid dead/placeholder controls during the demo.

## Backend robustness
- Celery dispatch hardened: `/upload`, `/api/runs`, and `/summary/run/{run_id}` now send jobs via the shared `celery_app.send_task(...)` pattern (`embed_run_task`, `rebuild_index_task`, `generate_summary_task`) so worker logs clearly show each stage and silent enqueue failures are avoided.

## Known limitations
- Gemini API key is required for summaries and peptide explanations; without it, endpoints return an error payload.
- Celery/Redis must be running for background embedding/index/summary jobs; if unavailable, summaries fall back inline but background queueing will not.
- Postgres backend is scaffolded but not active; current runtime still uses SQLite paths.
- FAISS index rebuild is global; if the broker is down and embeddings are added manually, run a rebuild from the app or worker when restored.

## Running the Demo Smoke Test Locally
- Required env: `DEMO_EMAIL`, `DEMO_PASSWORD`, `SMOKE_BASE_URL` (e.g., http://localhost:8080), `CELERY_BROKER_URL` (e.g., redis://127.0.0.1:6379/0), `CELERY_RESULT_BACKEND` (same as broker).
- Terminal 1 (API): `.\.venv\Scripts\activate && uvicorn app:app --host 0.0.0.0 --port 8080 --reload`
- Terminal 2 (Celery worker): `set PYTHONPATH=c:\Users\test\metabo-mvp && .\.venv\Scripts\python.exe -m celery -A worker.celery_app worker --loglevel=info --pool=solo --logfile=celery.log`
- Terminal 3 (smoke): `set DEMO_EMAIL=demo@example.com && set DEMO_PASSWORD=supersecret && set SMOKE_BASE_URL=http://localhost:8080 && set CELERY_BROKER_URL=redis://127.0.0.1:6379/0 && set CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/0 && python scripts/smoke_demo.py`
- Output interpretation:
  - `Embeddings: OK` → core pipeline healthy.
  - `Summary: WARN` → expected if `GEMINI_API_KEY` is missing.
  - `Fingerprint: WARN` → expected for tiny demo runs (<4 peptides); not a blocker.
  - Exit code 0 = demo pipeline healthy (Auth + Upload + Embeddings succeeded).

## Recommended demo flow
1) Login to obtain a bearer token (UI uses localStorage); keep the token available for fetches.
2) Upload a dataset via `/static/upload.html`; note the returned `run_id` and rows ingested. The embed job is queued in Celery.
3) Open `/static/runs.html`, select the new run, and follow links to fingerprint (`/runs/{run_id}/fingerprint`) and search/explain views once embeddings are ready.
4) Visit `/static/summary.html?run_id=...` to show the AI summary; regenerate if desired.
5) Export embeddings via `/export/embeddings/{run_id}` or bundle multiple runs via `/export/bundle` for take-home artifacts.
