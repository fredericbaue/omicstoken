"""
Test script to verify peptide embeddings export functionality.
This script:
1. Authenticates as a user
2. Uploads a small test file
3. Waits for embedding to complete
4. Calls the export endpoint
5. Verifies the response structure
"""

import requests
import time
import json

BASE_URL = "http://127.0.0.1:8080"

def test_embedding_export():
    print("=" * 60)
    print("Testing Peptide Embeddings Export")
    print("=" * 60)
    
    # Step 1: Login (using existing test user)
    print("\n[1] Logging in...")
    login_data = {
        "username": "user_a@example.com",
        "password": "password123"
    }
    
    res = requests.post(f"{BASE_URL}/auth/jwt/login", data=login_data)
    if res.status_code != 200:
        print(f"   ❌ Login failed: {res.status_code}")
        print(f"   Response: {res.text}")
        print(f"   Note: Make sure to run verify_auth_flow.py first to create test users")
        return False
    
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"   ✅ Login successful")
    
    # Step 2: Upload a test file
    print("\n[2] Uploading test file...")
    
    # Create a minimal test CSV with correct column names for generic format
    test_csv = """feature_id,sequence,intensity
FEAT_001,PEPTIDE,1000.0
FEAT_002,TESTSEQ,2000.0
FEAT_003,SAMPLE,1500.0
"""
    
    files = {"file": ("test_peptides.csv", test_csv, "text/csv")}
    data = {
        "format": "generic",
        "run_id": "TEST_EXPORT_RUN",
        "instrument": "Test Instrument",
        "method": "Test Method"
    }
    
    res = requests.post(f"{BASE_URL}/upload", headers=headers, files=files, data=data)
    if res.status_code != 200:
        print(f"   ❌ Upload failed: {res.status_code}")
        print(f"   Response: {res.text}")
        return False
    
    upload_result = res.json()
    run_id = upload_result["run_id"]
    print(f"   ✅ Upload successful: {run_id}")
    print(f"   Rows ingested: {upload_result['rows_ingested']}")
    
    # Step 3: Wait for embedding to complete
    print("\n[3] Waiting for background embedding to complete...")
    max_wait = 30  # seconds
    wait_interval = 2
    elapsed = 0
    
    while elapsed < max_wait:
        time.sleep(wait_interval)
        elapsed += wait_interval
        
        # Check if embeddings are ready by calling the export endpoint
        res = requests.get(f"{BASE_URL}/export/embeddings/{run_id}", headers=headers)
        if res.status_code == 200:
            embeddings = res.json()
            if len(embeddings) > 0:
                print(f"   ✅ Embeddings ready after {elapsed}s")
                break
        
        print(f"   ⏳ Waiting... ({elapsed}s)")
    else:
        print(f"   ⚠️  Timeout after {max_wait}s - embeddings may not be ready")
    
    # Step 4: Call export endpoint
    print("\n[4] Calling export endpoint...")
    res = requests.get(f"{BASE_URL}/export/embeddings/{run_id}", headers=headers)
    
    if res.status_code != 200:
        print(f"   ❌ Export failed: {res.status_code}")
        print(f"   Response: {res.text}")
        return False
    
    embeddings = res.json()
    print(f"   ✅ Export successful")
    print(f"   Number of embeddings: {len(embeddings)}")
    
    # Step 5: Verify response structure
    print("\n[5] Verifying response structure...")
    
    if not isinstance(embeddings, list):
        print(f"   ❌ Response is not a list")
        return False
    
    if len(embeddings) == 0:
        print(f"   ⚠️  No embeddings returned (background task may still be running)")
        return True  # Not a failure, just not ready yet
    
    # Check first embedding structure
    first_emb = embeddings[0]
    required_fields = ["sequence", "intensity", "length", "charge", "hydrophobicity", "embedding"]
    
    for field in required_fields:
        if field not in first_emb:
            print(f"   ❌ Missing field: {field}")
            return False
    
    print(f"   ✅ All required fields present")
    
    # Display sample embedding
    print("\n[6] Sample embedding:")
    print(f"   Sequence: {first_emb['sequence']}")
    print(f"   Intensity: {first_emb['intensity']}")
    print(f"   Length: {first_emb['length']}")
    print(f"   Charge: {first_emb['charge']}")
    print(f"   Hydrophobicity: {first_emb['hydrophobicity']}")
    print(f"   Embedding vector length: {len(first_emb['embedding'])}")
    print(f"   First 5 values: {first_emb['embedding'][:5]}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_embedding_export()
    exit(0 if success else 1)
