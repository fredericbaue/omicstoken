import requests
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_access_control():
    print("🔒 Starting Access Control Verification...")

    # --- Setup Users ---
    def get_token(email, password):
        # Register
        requests.post(f"{BASE_URL}/auth/register", json={"email": email, "password": password})
        # Login
        res = requests.post(f"{BASE_URL}/auth/jwt/login", data={"username": email, "password": password})
        if res.status_code != 200:
            print(f"❌ Login failed for {email}")
            sys.exit(1)
        return res.json()["access_token"]

    print("\n1. Setting up users...")
    token_a = get_token("user_secure_a@example.com", "password123")
    token_b = get_token("user_secure_b@example.com", "password123")
    print("   ✅ Users A and B logged in")

    # --- User A Uploads Data ---
    print("\n2. User A uploading data...")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    files = {"file": ("test_sec.csv", "feature_id,peptide_sequence,intensity\nPEP_SEC_1,PEPTIDE,1000", "text/csv")}
    run_id = "RUN_SECURE_A"
    res = requests.post(f"{BASE_URL}/upload", headers=headers_a, files=files, data={"run_id": run_id})
    if res.status_code != 200:
        print(f"❌ Upload failed: {res.text}")
        sys.exit(1)
    print(f"   ✅ Uploaded {run_id}")

    # --- User B Attacks ---
    headers_b = {"Authorization": f"Bearer {token_b}"}
    
    print("\n3. User B attempting to access User A's run details...")
    res = requests.get(f"{BASE_URL}/runs/{run_id}", headers=headers_b)
    if res.status_code in [403, 404]:
        print(f"   ✅ Blocked: {res.status_code} (Expected)")
    else:
        print(f"   ❌ FAIL: User B accessed run details! Status: {res.status_code}")

    print("\n4. User B attempting to generate summary for User A's run...")
    res = requests.post(f"{BASE_URL}/summary/run/{run_id}", headers=headers_b)
    if res.status_code in [403, 404]:
        print(f"   ✅ Blocked: {res.status_code} (Expected)")
    else:
        print(f"   ❌ FAIL: User B triggered summary! Status: {res.status_code}")

    print("\n5. User B attempting to compare User A's run...")
    # User B compares A's run with A's run (or any run)
    res = requests.get(f"{BASE_URL}/compare/{run_id}/{run_id}", headers=headers_b)
    if res.status_code in [403, 404]:
        print(f"   ✅ Blocked: {res.status_code} (Expected)")
    else:
        print(f"   ❌ FAIL: User B accessed comparison! Status: {res.status_code}")

    # --- User A Access (Should work) ---
    print("\n6. User A accessing their own run...")
    res = requests.get(f"{BASE_URL}/runs/{run_id}", headers=headers_a)
    if res.status_code == 200:
        print("   ✅ User A accessed run details")
    else:
        print(f"   ❌ FAIL: User A blocked from own run! Status: {res.status_code}")

if __name__ == "__main__":
    test_access_control()
