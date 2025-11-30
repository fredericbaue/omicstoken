import argparse
import os
import sys
import json
import requests


def main():
    """
    Main function to handle CLI arguments and export logic.
    """
    parser = argparse.ArgumentParser(
        description="Export OmicsToken v1 JSON for a given run_id from the local API."
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="The ID of the run to export."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="The directory to save the export file to. Default: 'exports'"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="The base URL of the running FastAPI application. Default: 'http://localhost:8000'"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Bearer token; defaults to API_TOKEN env if omitted."
    )

    args = parser.parse_args()

    # 1. Construct the API endpoint URL
    endpoint = f"{args.base_url.rstrip('/')}/export/embeddings/{args.run_id}"
    print(f"Fetching data from: {endpoint}")

    # 2. Resolve authentication token
    token = args.token or os.getenv("API_TOKEN")
    if not token:
        print("Missing API token. Set API_TOKEN env or pass --token <TOKEN>.", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {token}"}
    token_source = "--token" if args.token else "API_TOKEN env"
    print(f"Using Bearer token from {token_source}.")

    try:
        # 3. Call the FastAPI endpoint with a timeout
        response = requests.get(endpoint, headers=headers, timeout=30)

        # 4. Check the response
        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("\nError: Failed to parse JSON response from server.", file=sys.stderr)
                sys.exit(1)

            # 5. Ensure the output directory exists
            os.makedirs(args.output_dir, exist_ok=True)

            # 6. Define the output file path
            output_filename = f"{args.run_id}_omics_export_v1.json"
            output_path = os.path.join(args.output_dir, output_filename)

            # 7. Write the JSON response to the file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            absolute_path = os.path.abspath(output_path)
            print(f"\nSaved OmicsToken v1 export for run '{args.run_id}' to:")
            print(f"   {absolute_path}")

        else:
            print(f"\nError: Received status code {response.status_code}", file=sys.stderr)
            if response.status_code == 401:
                print("401 Unauthorized: Token was missing or rejected. Ensure the token is valid and the user owns the run.", file=sys.stderr)
            elif response.status_code == 404:
                print("404 Not Found: Run may not exist or is not owned by this token.", file=sys.stderr)
            else:
                print("Request failed. See response details below.", file=sys.stderr)
            print(f"Response body: {response.text}", file=sys.stderr)
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"\nRequest Error: Could not connect to {args.base_url}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("Please ensure the FastAPI server is running and accessible.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
