"""
Quick demo script to test the export embeddings endpoint.
This simulates what you would do in the browser console.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8080"

# Step 1: Login to get a token
print("🔐 Logging in...")
login_data = {
    "username": "user_a@example.com",
    "password": "password123"
}

res = requests.post(f"{BASE_URL}/auth/jwt/login", data=login_data)
if res.status_code != 200:
    print(f"❌ Login failed: {res.status_code}")
    exit(1)

token = res.json()["access_token"]
print(f"✅ Login successful")

# Step 2: Get list of runs
print("\n📋 Fetching your runs...")
headers = {"Authorization": f"Bearer {token}"}
res = requests.get(f"{BASE_URL}/runs", headers=headers)

if res.status_code != 200:
    print(f"❌ Failed to get runs: {res.status_code}")
    exit(1)

runs = res.json()
if not runs:
    print("⚠️  No runs found. Upload some data first!")
    exit(0)

print(f"Found {len(runs)} run(s):")
for run in runs[:5]:  # Show first 5
    print(f"  - {run['run_id']} ({run.get('n_embeddings', 0)} embeddings)")

# Step 3: Export embeddings from the first run
run_id = runs[0]['run_id']
print(f"\n📤 Exporting embeddings from run: {run_id}")

res = requests.get(f"{BASE_URL}/export/embeddings/{run_id}", headers=headers)

if res.status_code != 200:
    print(f"❌ Export failed: {res.status_code}")
    print(f"Response: {res.text}")
    exit(1)

embeddings = res.json()
print(f"✅ Export successful!")
print(f"\n📊 Results:")
print(f"  Total embeddings: {len(embeddings)}")

if embeddings:
    print(f"\n🔬 First embedding sample:")
    first = embeddings[0]
    print(f"  Sequence: {first['sequence']}")
    print(f"  Intensity: {first['intensity']}")
    print(f"  Length: {first['length']}")
    print(f"  Charge: {first['charge']}")
    print(f"  Hydrophobicity: {first['hydrophobicity']}")
    print(f"  Embedding dimensions: {len(first['embedding'])}")
    print(f"  First 5 values: {first['embedding'][:5]}")
    
    # Show a few more sequences
    if len(embeddings) > 1:
        print(f"\n📝 All sequences in this run:")
        for i, emb in enumerate(embeddings[:10], 1):
            print(f"  {i}. {emb['sequence']} (intensity: {emb['intensity']})")
        if len(embeddings) > 10:
            print(f"  ... and {len(embeddings) - 10} more")
else:
    print("  ⚠️  No embeddings found (background task may still be processing)")

print("\n✨ Done!")
