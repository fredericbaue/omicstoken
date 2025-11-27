import requests
import json
import os

BASE_URL = "http://127.0.0.1:8080"

def test_auth_flow():
    print("🚀 Starting Auth Flow Verification...")

    # 1. Register User A
    print("\n1. Registering User A (user_a@example.com)...")
    user_a = {"email": "user_a@example.com", "password": "password123", "is_active": True, "is_superuser": False, "is_verified": False}
    res = requests.post(f"{BASE_URL}/auth/register", json=user_a)
    if res.status_code == 400 and "REGISTER_USER_ALREADY_EXISTS" in res.text:
        print("   User A already exists (skipping registration)")
    elif res.status_code != 201:
        print(f"   ❌ Registration failed: {res.text}")
        return
    else:
        print("   ✅ User A registered")

    # 2. Login User A
    print("\n2. Logging in User A...")
    res = requests.post(f"{BASE_URL}/auth/jwt/login", data={"username": user_a["email"], "password": user_a["password"]})
    if res.status_code != 200:
        print(f"   ❌ Login failed: {res.text}")
        return
    token_a = res.json()["access_token"]
    print(f"   ✅ Login successful. Token: {token_a[:10]}...")

    # 3. Upload File as User A
    print("\n3. Uploading file as User A...")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    files = {"file": ("test_a.csv", "feature_id,peptide_sequence,intensity\nPEPTIDE_A,PEPTIDE,1000", "text/csv")}
    res = requests.post(f"{BASE_URL}/upload", headers=headers_a, files=files, data={"run_id": "RUN_USER_A"})
    if res.status_code != 200:
        print(f"   ❌ Upload failed: {res.text}")
        return
    print("   ✅ Upload successful")

    # 4. List Runs for User A
    print("\n4. Listing runs for User A...")
    res = requests.get(f"{BASE_URL}/runs", headers=headers_a)
    runs = res.json()
    run_ids = [r["run_id"] for r in runs]
    print(f"   Runs found: {run_ids}")
    if "RUN_USER_A" in run_ids:
        print("   ✅ User A sees their run")
    else:
        print("   ❌ User A DOES NOT see their run")

    # 5. Register User B
    print("\n5. Registering User B (user_b@example.com)...")
    user_b = {"email": "user_b@example.com", "password": "password123", "is_active": True, "is_superuser": False, "is_verified": False}
    res = requests.post(f"{BASE_URL}/auth/register", json=user_b)
    if res.status_code == 400 and "REGISTER_USER_ALREADY_EXISTS" in res.text:
        print("   User B already exists")
    elif res.status_code != 201:
        print(f"   ❌ Registration failed: {res.text}")
        return
    else:
        print("   ✅ User B registered")

    # 6. Login User B
    print("\n6. Logging in User B...")
    res = requests.post(f"{BASE_URL}/auth/jwt/login", data={"username": user_b["email"], "password": user_b["password"]})
    token_b = res.json()["access_token"]
    print(f"   ✅ Login successful")

    # 7. List Runs for User B (Should NOT see User A's run)
    print("\n7. Listing runs for User B...")
    headers_b = {"Authorization": f"Bearer {token_b}"}
    res = requests.get(f"{BASE_URL}/runs", headers=headers_b)
    runs_b = res.json()
    run_ids_b = [r["run_id"] for r in runs_b]
    print(f"   Runs found: {run_ids_b}")
    if "RUN_USER_A" in run_ids_b:
        print("   ❌ SECURITY FAIL: User B can see User A's run!")
    else:
        print("   ✅ SECURITY PASS: User B cannot see User A's run")

if __name__ == "__main__":
    test_auth_flow()
