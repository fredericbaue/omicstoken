import sys
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_dashboard_endpoint():
    print("Testing dashboard endpoint...")
    
    # 1. List runs to get a valid run_id
    response = client.get("/runs")
    if response.status_code != 200:
        print(f"Failed to list runs: {response.status_code}")
        return
        
    runs = response.json()
    if not runs:
        print("No runs found. Please upload data first (or run verify_tumor_flow.py to seed data).")
        return
        
    run_id = runs[0]['run_id']
    print(f"Using Run ID: {run_id}")
    
    # 2. Fetch dashboard data
    response = client.get(f"/dashboard-data/{run_id}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success! Retrieved {len(data['data'])} data points.")
        if data['data']:
            print("Sample data point:")
            print(data['data'][0])
            
            # Verify properties exist
            props = data['data'][0]['properties']
            if 'hydrophobicity' in props and 'molecular_weight' in props:
                print("[PASS] Biophysical properties present.")
            else:
                print("[FAIL] Biophysical properties missing.")
    else:
        print(f"Failed to get dashboard data: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_dashboard_endpoint()
