"""
Quick demo to show the export endpoint working with TEST_EXPORT_RUN
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8080"

# Login
login_data = {"username": "user_a@example.com", "password": "password123"}
res = requests.post(f"{BASE_URL}/auth/jwt/login", data=login_data)
token = res.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Export embeddings from TEST_EXPORT_RUN
run_id = "TEST_EXPORT_RUN"
print(f"📤 Exporting embeddings from: {run_id}\n")

res = requests.get(f"{BASE_URL}/export/embeddings/{run_id}", headers=headers)
embeddings = res.json()

print(f"✅ Got {len(embeddings)} embeddings\n")
print("=" * 60)

for i, emb in enumerate(embeddings, 1):
    print(f"\nPeptide #{i}:")
    print(f"  Sequence:       {emb['sequence']}")
    print(f"  Intensity:      {emb['intensity']}")
    print(f"  Length:         {emb['length']} amino acids")
    print(f"  Charge:         {emb['charge']}")
    print(f"  Hydrophobicity: {emb['hydrophobicity']}")
    print(f"  Embedding:      {len(emb['embedding'])}-dimensional vector")
    print(f"                  {emb['embedding'][:5]}...")

print("\n" + "=" * 60)
print("\n💡 This is what you'd get in the browser console!")
print("\nTo try it yourself, open the browser console at")
print("http://127.0.0.1:8080/static/runs.html and run:")
print(f"""
fetch("http://127.0.0.1:8080/export/embeddings/{run_id}", {{
  headers: {{
    "Authorization": "Bearer " + localStorage.getItem("token")
  }}
}})
.then(r => r.json())
.then(console.log)
""")
