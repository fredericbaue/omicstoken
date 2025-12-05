"""End-to-end smoke test for the omicstoken demo pipeline."""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

try:
    import config
except ImportError:  # pragma: no cover - fallback for non-package execution
    config = None

BASE_URL = os.getenv("SMOKE_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT = 10
POLL_TIMEOUT_SECONDS = 90
POLL_INTERVAL_SECONDS = 3


def _get_broker_info() -> Tuple[str, str]:
    broker = os.getenv("CELERY_BROKER_URL") or getattr(config, "CELERY_BROKER_URL", "unknown")
    backend = os.getenv("CELERY_RESULT_BACKEND") or getattr(
        config, "CELERY_RESULT_BACKEND", "unknown"
    )
    return broker, backend


def _print_environment_context():
    broker, backend = _get_broker_info()
    print(f"[SMOKE DEMO] API base URL: {BASE_URL}")
    print(f"[SMOKE DEMO] Celery broker: {broker}")
    print(f"[SMOKE DEMO] Result backend: {backend}")


def _credential_or_exit(value: Optional[str], name: str) -> str:
    if value:
        return value
    flag_name = name.lower().replace("_", "-")
    print(
        f"ERROR: Missing {name}. Set {name} in the environment or pass --{flag_name}.",
        file=sys.stderr,
    )
    print(
        "Ensure the demo user exists in the auth DB before rerunning this smoke test.",
        file=sys.stderr,
    )
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the OmicsToken API + Celery pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--email", help="Demo user email (overrides DEMO_EMAIL)")
    parser.add_argument("--password", help="Demo user password (overrides DEMO_PASSWORD)")
    parser.add_argument("--base-url", help="Override API base URL (overrides SMOKE_BASE_URL)")
    return parser.parse_args()


def authenticate(session: requests.Session, email: str, password: str) -> Tuple[bool, str]:
    url = f"{BASE_URL}/auth/jwt/login"
    payload = {"username": email, "password": password}
    try:
        response = session.post(url, data=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            return False, "Auth response missing access_token"
        session.headers.update({"Authorization": f"Bearer {token}"})
        return True, "OK"
    except requests.RequestException as exc:
        return False, f"HTTP error: {exc}"
    except ValueError as exc:
        return False, f"Invalid JSON response: {exc}"


def _build_csv() -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["feature_id", "peptide_sequence", "intensity"])
    writer.writeheader()
    rows = [
        {"feature_id": "SMOKE_PEPTIDE_1", "peptide_sequence": "ACDEFGHIK", "intensity": 12500},
        {"feature_id": "SMOKE_PEPTIDE_2", "peptide_sequence": "LMNPQRSTV", "intensity": 9800},
        {"feature_id": "SMOKE_PEPTIDE_3", "peptide_sequence": "WYACDEFGH", "intensity": 15200},
    ]
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def upload_csv(session: requests.Session, run_id: str, broker_url: str) -> Tuple[bool, str, Optional[str], int]:
    url = f"{BASE_URL}/upload"
    csv_payload = _build_csv()
    files = {
        "file": ("smoke_demo.csv", csv_payload, "text/csv"),
    }
    data = {
        "run_id": run_id,
        "instrument": "SMOKE",
        "method": "DEMO",
        "format": "generic",
    }
    try:
        response = session.post(url, data=data, files=files, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        body = response.json()
        server_run_id = body.get("run_id", run_id)
        print(
            f"[SMOKE DEMO] Upload accepted (run_id={server_run_id}); Celery broker={broker_url}. "
            "Embed job queued via /upload - monitor worker logs for embed_run_task."
        )
        return True, server_run_id, body.get("first_feature_id"), int(body.get("rows_ingested", 0))
    except requests.RequestException as exc:
        return False, f"HTTP error: {exc}", None, 0
    except ValueError as exc:
        return False, f"Invalid JSON response: {exc}", None, 0


def poll_for_embeddings(session: requests.Session, run_id: str) -> Tuple[bool, str, int]:
    url = f"{BASE_URL}/runs/{run_id}"
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    last_error = ""
    while time.time() < deadline:
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 404:
                last_error = "Run not found yet."
            else:
                response.raise_for_status()
                payload = response.json()
                stats = payload.get("stats", {})
                n_embeddings = stats.get("n_embeddings", 0)
                print(f"[SMOKE DEMO] Waiting for embeddings... current count={n_embeddings}")
                if n_embeddings and n_embeddings > 0:
                    print(
                        f"[SMOKE DEMO] Embeddings: OK for run {run_id} (n_embeddings={n_embeddings})"
                    )
                    return True, "Embeddings ready", int(n_embeddings)
                last_error = f"Embeddings not ready (count={n_embeddings})"
        except requests.RequestException as exc:
            last_error = f"HTTP error: {exc}"
        except ValueError as exc:
            last_error = f"Invalid JSON response: {exc}"

        time.sleep(POLL_INTERVAL_SECONDS)

    return False, f"Timed out waiting for embeddings: {last_error}", 0


def call_summary(session: requests.Session, run_id: str) -> Tuple[str, str]:
    if not os.getenv("GEMINI_API_KEY"):
        return "WARN", "GEMINI_API_KEY not set; skipping summary check"

    url = f"{BASE_URL}/summary/run/{run_id}"
    try:
        response = session.post(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            payload = response.json()
            if payload.get("error"):
                error_text = payload["error"]
                if "gemini" in error_text.lower():
                    return "WARN", error_text
                return "FAILED", f"LLM error: {error_text}"
            return "OK", "OK"
        else:
            detail = response.text
            return "FAILED", f"HTTP {response.status_code}: {detail}"
    except requests.RequestException as exc:
        return "FAILED", f"HTTP error: {exc}"
    except ValueError as exc:
        return "FAILED", f"Invalid JSON response: {exc}"


def call_search(session: requests.Session, run_id: str, feature_id: str) -> Tuple[bool, str]:
    url = f"{BASE_URL}/peptide/search/{run_id}/{feature_id}?k=5"
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        neighbors = payload.get("neighbors", [])
        return True, f"{len(neighbors)} neighbors"
    except requests.RequestException as exc:
        return False, f"HTTP error: {exc}"
    except ValueError as exc:
        return False, f"Invalid JSON response: {exc}"


def call_fingerprint(session: requests.Session, run_id: str) -> Tuple[bool, str]:
    url = f"{BASE_URL}/runs/{run_id}/fingerprint"
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 400:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail") or payload.get("error") or ""
                else:
                    detail = str(payload)
            except ValueError:
                detail = response.text or ""
            body_text = (detail or response.text or "").lower()
            if "enough peptides" in body_text or "not enough peptides" in body_text:
                return False, "WARN_TOO_FEW"
        response.raise_for_status()
        payload = response.json()
        clusters = payload.get("clusters", [])
        total = payload.get("total_peptides", 0)
        return True, f"clusters={len(clusters)}, total_peptides={total}"
    except requests.RequestException as exc:
        return False, f"HTTP error: {exc}"
    except ValueError as exc:
        return False, f"Invalid JSON response: {exc}"


def call_export(session: requests.Session, run_id: str) -> Tuple[bool, str]:
    url = f"{BASE_URL}/export/embeddings/{run_id}"
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        total = payload.get("total_embeddings", 0)
        if payload.get("data"):
            return True, f"total_embeddings={total}"
        return False, "Export returned zero records"
    except requests.RequestException as exc:
        return False, f"HTTP error: {exc}"
    except ValueError as exc:
        return False, f"Invalid JSON response: {exc}"


def main() -> int:
    args = parse_args()
    global BASE_URL
    if args.base_url:
        BASE_URL = args.base_url
    else:
        BASE_URL = os.getenv("SMOKE_BASE_URL", BASE_URL)

    email = _credential_or_exit(args.email or os.getenv("DEMO_EMAIL"), "DEMO_EMAIL")
    password = _credential_or_exit(args.password or os.getenv("DEMO_PASSWORD"), "DEMO_PASSWORD")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    _print_environment_context()
    print(f"[SMOKE DEMO] DEMO_EMAIL: {email}")

    broker_url, _ = _get_broker_info()
    results: Dict[str, Dict[str, str]] = {}

    ok, message = authenticate(session, email, password)
    results["Auth"] = {"ok": "OK" if ok else "FAILED", "detail": message}
    if not ok:
        print_report(results)
        return 1

    run_id = f"SMOKE_DEMO_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    ok, upload_info, first_feature_id, rows_ingested = upload_csv(session, run_id, broker_url)
    if ok:
        results["Upload"] = {"ok": "OK", "detail": f"run_id={upload_info}, rows={rows_ingested}"}
        run_id = upload_info
    else:
        results["Upload"] = {"ok": "FAILED", "detail": upload_info}
        print_report(results)
        return 1

    ok, message, n_embeddings = poll_for_embeddings(session, run_id)
    results["Embeddings"] = {
        "ok": "OK" if ok else "FAILED",
        "detail": f"{message} (n_embeddings={n_embeddings})",
    }
    if not ok:
        print(f"ERROR: Embeddings did not complete for run {run_id} within {POLL_TIMEOUT_SECONDS}s.")
        print("Check Celery worker, Redis, and API availability before rerunning.")
        print_report(results)
        return 1

    summary_status, summary_detail = call_summary(session, run_id)
    results["Summary"] = {"ok": summary_status, "detail": summary_detail}

    if first_feature_id:
        search_ok, search_detail = call_search(session, run_id, first_feature_id)
        results["Search"] = {"ok": "OK" if search_ok else "FAILED", "detail": search_detail}
    else:
        results["Search"] = {"ok": "SKIPPED", "detail": "No feature_id returned from upload"}

    fingerprint_ok, fingerprint_detail = call_fingerprint(session, run_id)
    if fingerprint_ok:
        results["Fingerprint"] = {"ok": "OK", "detail": fingerprint_detail}
    else:
        too_small = (n_embeddings is not None and n_embeddings < 4) or fingerprint_detail == "WARN_TOO_FEW"
        if too_small:
            results["Fingerprint"] = {
                "ok": "WARN",
                "detail": "Run too small for fingerprint; core pipeline still OK",
            }
        else:
            results["Fingerprint"] = {"ok": "FAILED", "detail": fingerprint_detail}

    export_ok, export_detail = call_export(session, run_id)
    results["Export"] = {"ok": "OK" if export_ok else "FAILED", "detail": export_detail}

    print_report(results)
    critical = all(results[name]["ok"] == "OK" for name in ("Auth", "Upload", "Embeddings"))

    print("\nDEMO SMOKE SUMMARY")
    print(f"- Login: {results['Auth']['ok']} ({results['Auth']['detail']})")
    print(f"- Upload: {results['Upload']['ok']} ({results['Upload']['detail']})")
    print(f"- Embeddings: {results['Embeddings']['ok']} ({results['Embeddings']['detail']})")
    print(f"- Summary: {results['Summary']['ok']} ({results['Summary']['detail']})")
    print(f"- Fingerprint: {results['Fingerprint']['ok']} ({results['Fingerprint']['detail']})")
    print(f"- Export: {results['Export']['ok']} ({results['Export']['detail']})")
    print(f"- Search: {results['Search']['ok']} ({results['Search']['detail']})")

    if critical:
        return 0
    return 2


def print_report(results: Dict[str, Dict[str, str]]) -> None:
    print("\n[SMOKE DEMO RESULT]")
    for name, outcome in results.items():
        status = outcome.get("ok", "UNKNOWN")
        detail = outcome.get("detail", "")
        print(f"- {name}: {status} ({detail})")


if __name__ == "__main__":
    sys.exit(main())

# Manual smoke test checklist:
# 1. Start Redis/Memurai locally (`redis-server` or equivalent) so the broker at localhost:6379/0 is reachable.
# 2. Launch the Celery worker: `celery -A worker.celery_app worker --loglevel=info`.
# 3. Start the FastAPI server: `uvicorn app:app --host 0.0.0.0 --port 8080`.
# 4. Run this script with DEMO_EMAIL/DEMO_PASSWORD exported: `python scripts/smoke_demo.py`.
# 5. Watch the worker logs for `Task embed_run_task[...] received` plus start/completed lines, then confirm the smoke result reports `Embeddings: OK (n_embeddings > 0)`.
