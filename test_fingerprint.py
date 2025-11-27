import httpx
import os
import getpass

# --- Configuration ---
BASE_URL = "http://127.0.0.1:8080"
# ---

def login(client, email, password):
    """Logs in to the application and returns the auth token."""
    print("Attempting to log in...")
    login_data = {
        "username": email,
        "password": password,
    }
    try:
        r = client.post(f"{BASE_URL}/auth/jwt/login", data=login_data)
        r.raise_for_status()
        token = r.json()["access_token"]
        print("Login successful.")
        return token
    except httpx.HTTPStatusError as e:
        print(f"Login failed: {e.response.status_code} {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during login: {e}")
        return None

def get_first_run_id(client, token):
    """Fetches the list of runs and returns the ID of the first one."""
    print("Fetching user's runs...")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = client.get(f"{BASE_URL}/runs", headers=headers)
        r.raise_for_status()
        runs = r.json()
        if not runs:
            print("No runs found for this user.")
            return None
        first_run_id = runs[0]["run_id"]
        print(f"Found run_id: {first_run_id}")
        return first_run_id
    except httpx.HTTPStatusError as e:
        print(f"Failed to get runs: {e.response.status_code} {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching runs: {e}")
        return None

def get_fingerprint(client, token, run_id):
    """Fetches the semantic fingerprint for a given run_id."""
    print(f"\nFetching fingerprint for run_id: {run_id}...")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with client.stream("GET", f"{BASE_URL}/runs/{run_id}/fingerprint", headers=headers, timeout=30) as r:
            r.raise_for_status()
            print("--- Fingerprint Response ---")
            for chunk in r.iter_text():
                print(chunk, end="")
            print("\n----------------------------")
    except httpx.HTTPStatusError as e:
        print(f"Failed to get fingerprint: {e.response.status_code}")
        print("Response body:", e.response.text)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """Main function to run the test."""
    email = input("Enter your login email: ")
    password = getpass.getpass("Enter your password: ")
    
    with httpx.Client() as client:
        token = login(client, email, password)
        if not token:
            return

        run_id = get_first_run_id(client, token)
        if not run_id:
            return
            
        get_fingerprint(client, token, run_id)

if __name__ == "__main__":
    main()
