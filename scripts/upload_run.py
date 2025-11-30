import argparse
import json
import os
import sys
import requests


def main():
    parser = argparse.ArgumentParser(description="Programmatic run upload to OmicsToken.")
    parser.add_argument("--name", type=str, help="Optional run name/label.")
    parser.add_argument("--features-json", required=True, help="Path to features JSON file.")
    parser.add_argument("--auto-embed", action="store_true", help="Trigger embedding after upload.")
    parser.add_argument("--output", action="store_true", help="Print full response JSON.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL.")
    parser.add_argument("--token", type=str, help="API token; defaults to API_TOKEN env.")
    args = parser.parse_args()

    token = args.token or os.getenv("API_TOKEN")
    if not token:
        print("Missing API token. Set API_TOKEN env or pass --token <TOKEN>.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.features_json, "r", encoding="utf-8") as f:
            features = json.load(f)
    except Exception as e:
        print(f"Failed to read features JSON: {e}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "name": args.name,
        "metadata": {},
        "features": features,
        "auto_embed": args.auto_embed,
    }

    endpoint = f"{args.base_url.rstrip('/')}/api/runs"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        print(f"Request error: {e}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code not in (200, 201):
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        print(f"Error: status {resp.status_code}. {detail}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    print(f"Run created: {data.get('run_id')} (auto_embed={data.get('auto_embed')})")
    if args.output:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
