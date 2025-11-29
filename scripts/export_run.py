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
        default=os.getenv("API_TOKEN"),
        help="Authentication token. Can also be set via API_TOKEN environment variable."
    )

    args = parser.parse_args()

    # 1. Construct the API endpoint URL
    endpoint = f"{args.base_url.rstrip('/')}/export/embeddings/{args.run_id}"
    print(f"Fetching data from: {endpoint}")

    # 2. Prepare authentication headers
    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
        print("Authenticating with provided token...")
    else:
        print("⚠️  Warning: No authentication token provided. Request may fail if endpoint is protected.")

    try:
        # 3. Call the FastAPI endpoint with a timeout
        response = requests.get(endpoint, headers=headers, timeout=30)

        # 4. Check the response
        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError:
                print("\n❌ Error: Failed to parse JSON response from server.", file=sys.stderr)
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
            print(f"\n✅ Saved OmicsToken v1 export for run '{args.run_id}' to:")
            print(f"   {absolute_path}")

        else:
            # Handle non-200 responses
            print(f"\n❌ Error: Received status code {response.status_code}", file=sys.stderr)
            if response.status_code == 401:
                print("   Authentication failed. Please provide a valid token via --token or API_TOKEN.", file=sys.stderr)
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request Error: Could not connect to {args.base_url}", file=sys.stderr)
        print(f"   Details: {e}", file=sys.stderr)
        print("   Please ensure the FastAPI server is running and accessible.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
