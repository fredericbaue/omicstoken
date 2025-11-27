import os
import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock google.generativeai
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai

from app import app

client = TestClient(app)

def test_tumor_workflow():
    print("1. Uploading Tumor CSV...")
    # Create dummy tumor CSV
    csv_content = """Accession,Description,Melanoma-1,Melanoma-2,Fold Change,q-value
P12345,Protein A,100,110,2.5,0.01
P67890,Protein B,50,60,-1.5,0.04
P54321,Protein C,200,210,0.1,0.5
"""
    files = {"file": ("tumor_test.csv", csv_content, "text/csv")}
    response = client.post("/upload", files=files, data={"format": "auto", "run_id": "TUMOR_RUN_001"})
    
    if response.status_code != 200:
        print(f"[FAIL] Upload failed: {response.status_code}")
        print(response.text)
        return
    print("[PASS] Upload successful.")
    
    print("2. Checking /runs endpoint...")
    response = client.get("/runs")
    if response.status_code == 200:
        runs = response.json()
        found = False
        for r in runs:
            if r["run_id"] == "TUMOR_RUN_001":
                found = True
                print(f"[PASS] Found run in list. Features: {r['n_features']}")
                if r['n_features'] == 3:
                    print("[PASS] Feature count correct.")
                else:
                    print(f"[WARN] Expected 3 features, got {r['n_features']}")
                break
        if not found:
            print("[FAIL] Run not found in /runs list.")
    else:
        print(f"[FAIL] /runs failed: {response.status_code}")

    print("3. Checking Run Details...")
    response = client.get("/runs/TUMOR_RUN_001")
    if response.status_code == 200:
        details = response.json()
        print("[PASS] Run details fetched.")
        if details["stats"]["n_features"] == 3:
            print("[PASS] Stats correct.")
    else:
        print(f"[FAIL] /runs/ID failed: {response.status_code}")

    print("4. Testing Tumor Summary...")
    # Mock Gemini response
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Tumor summary generated."
    mock_model.generate_content.return_value = mock_response
    
    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        # Patch API key check
        import summarizer
        summarizer.GEMINI_API_KEY = "dummy"
        
        response = client.post("/summary/run/TUMOR_RUN_001")
        if response.status_code == 200:
            data = response.json()
            print("[PASS] Summary generated.")
            if data.get("type") == "tumor_comparison":
                print("[PASS] Correctly identified as tumor comparison.")
            else:
                print(f"[WARN] Expected type 'tumor_comparison', got {data.get('type')}")
                
            if data.get("num_significant") == 2:
                print("[PASS] Correct significant count (2).")
            else:
                print(f"[WARN] Expected 2 significant, got {data.get('num_significant')}")
        else:
            print(f"[FAIL] Summary failed: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_tumor_workflow()
