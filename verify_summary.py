import os
import sys
import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock google.generativeai before importing app
# This allows us to test the integration without a real API key or network call
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai

# Now import app
from app import app

client = TestClient(app)

def test_summarization_flow():
    print("1. Uploading test data...")
    # Create a dummy CSV content
    csv_content = "Sequence,Intensity,Modified sequence\nPEPTIDE,100000,PEPTIDE\nANOTHER,50000,ANOTHER"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    
    response = client.post("/upload", files=files, data={"format": "maxquant", "run_id": "TEST_RUN_001"})
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return
    
    print("✅ Upload successful.")
    
    print("2. Testing Summarization Endpoint...")
    
    # Mock the generate_content response
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a mock scientific summary of the peptide data."
    mock_model.generate_content.return_value = mock_response
    
    # Patch the GenerativeModel constructor
    with patch("google.generativeai.GenerativeModel", return_value=mock_model):
        # We need to set a dummy API key in env if not present, 
        # because summarizer.py checks for it.
        if "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = "dummy_key"
            
        # Force reload of summarizer module to pick up the env var if needed? 
        # Actually, summarizer.py reads env at module level. 
        # If it was already imported by app, it might have seen None.
        # Let's patch the module-level variable if needed, but patching os.environ before import is best.
        # Since app is already imported, let's check summarizer.GEMINI_API_KEY
        import summarizer
        summarizer.GEMINI_API_KEY = "dummy_key" # Force it for the test
        
        response = client.post("/summary/run/TEST_RUN_001")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Summarization request successful.")
            print(f"   Summary: {data.get('summary_text')}")
            
            if data.get("summary_text") == "This is a mock scientific summary of the peptide data.":
                print("✅ Mock response verified.")
            else:
                print("⚠️ Unexpected response content.")
                
            if data.get("num_peptides") == 2:
                print("✅ Correct number of peptides analyzed.")
            else:
                print(f"⚠️ Expected 2 peptides, got {data.get('num_peptides')}")
                
        else:
            print(f"❌ Summarization failed: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_summarization_flow()
