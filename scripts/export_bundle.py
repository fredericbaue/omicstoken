import argparse
import os
import sys
import requests
from datetime import datetime
import json


def main():
    parser = argparse.ArgumentParser(description="Export multiple runs as an OmicsToken bundle (zip).")
    parser.add_argument("run_ids", nargs="+", help="Run IDs to export")
    parser.add_argument("--output-dir", type=str, default="exports", help="Directory to save the ZIP (default: exports)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--token", type=str, help="Bearer token; defaults to API_TOKEN env if omitted.")

    args = parser.parse_args()

    token = args.token or os.getenv("API_TOKEN")
    if not token:
        print("Missing API token. Set API_TOKEN env or pass --token <TOKEN>.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    payload = {"run_ids": args.run_ids}
    endpoint = f"{args.base_url.rstrip('/')}/export/bundle"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        print(f"Request error: {e}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        print(f"Error: status {resp.status_code}. {detail}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(args.output_dir, f"omics_bundle_{timestamp}.zip")
    with open(outfile, "wb") as f:
        f.write(resp.content)

    print(f"Saved bundle to {os.path.abspath(outfile)}")


if __name__ == "__main__":
    main()
