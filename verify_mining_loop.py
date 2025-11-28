import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8080"

def test_mining_loop():
    print("⛏️  Testing Mining Loop (Proof of Contribution)...")

    # 1. Register a unique user
    # Using a random suffix to ensure it's a new user every time we run the test
    import time
    unique_id = int(time.time())
    email = f"miner_{unique_id}@example.com"
    password = "password123"
    
    print(f"\n1. Registering new user: {email}...")
    user_data = {
        "email": email, 
        "password": password, 
        "is_active": True, 
        "is_superuser": False, 
        "is_verified": False
    }
    res = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    if res.status_code != 201:
        print(f"   ❌ Registration failed: {res.text}")
        return
    print("   ✅ Registered.")

    # 2. Login
    print("\n2. Logging in...")
    login_data = {"username": email, "password": password}
    res = requests.post(f"{BASE_URL}/auth/jwt/login", data=login_data)
    if res.status_code != 200:
        print(f"   ❌ Login failed: {res.text}")
        return
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   ✅ Logged in.")

    # 3. Check Initial Credits (Should be 0)
    print("\n3. Checking initial wallet balance...")
    res = requests.get(f"{BASE_URL}/runs", headers=headers)
    if res.status_code != 200:
        print(f"   ❌ Failed to fetch runs: {res.text}")
        return
    
    data = res.json()
    initial_credits = data.get("user_credits", 0)
    print(f"   💰 Balance: {initial_credits} Credits")
    
    if initial_credits != 0:
        print(f"   ⚠️  Expected 0 credits, found {initial_credits}.")
    else:
        print("   ✅ Initial balance correct.")

    # 4. Perform Mining Action (Upload Data)
    print("\n4. Uploading data (Mining)...")
    # Create a dummy CSV
    csv_content = "feature_id,peptide_sequence,intensity\nPEP_001,ACDEFG,1000\nPEP_002,HIKLMN,2000"
    files = {"file": ("mining_test.csv", csv_content, "text/csv")}
    upload_data = {"run_id": f"MINE_RUN_{unique_id}"}
    
    res = requests.post(f"{BASE_URL}/upload", headers=headers, files=files, data=upload_data)
    if res.status_code != 200:
        print(f"   ❌ Upload failed: {res.text}")
        return
    print("   ✅ Upload successful.")

    # 5. Verify Reward (Should be +10)
    print("\n5. Verifying reward...")
    res = requests.get(f"{BASE_URL}/runs", headers=headers)
    data = res.json()
    final_credits = data.get("user_credits", 0)
    
    print(f"   💰 New Balance: {final_credits} Credits")
    
    if final_credits == initial_credits + 10:
        print("   ✅ SUCCESS: User mined 10 credits!")
    else:
        print(f"   ❌ FAIL: Credits did not update correctly. (Expected {initial_credits + 10}, got {final_credits})")

if __name__ == "__main__":
    try:
        test_mining_loop()
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server. Is 'uvicorn app:app' running?")
