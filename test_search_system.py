"""
Test the upgraded embedding system end-to-end
"""
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

import db
import search
import embeddings

DATA_DIR = "data"

def test_search():
    print("=" * 60)
    print("Testing upgraded embedding system")
    print("=" * 60)
    
    con = db.get_db_connection(DATA_DIR)
    
    # Get a sample peptide
    cur = con.cursor()
    cur.execute("""
        SELECT run_id, feature_id, annotation_name 
        FROM features 
        WHERE annotation_name IS NOT NULL 
        LIMIT 1
    """)
    row = cur.fetchone()
    
    if not row:
        print("❌ No peptides found in database")
        return
    
    run_id, feature_id, sequence = row
    print(f"\nTest peptide:")
    print(f"  Run ID: {run_id}")
    print(f"  Feature ID: {feature_id}")
    print(f"  Sequence: {sequence}")
    
    # Check if embedding exists
    embedding = db.get_embedding(con, run_id, feature_id)
    if not embedding:
        print(f"\n❌ No embedding found for this peptide")
        return
    
    print(f"\n✓ Embedding found (dimension: {len(embedding)})")
    
    # Test search
    try:
        print(f"\nSearching for similar peptides...")
        results = search.search_similar_peptides(con, DATA_DIR, run_id, feature_id, k=5)
        
        print(f"\n✓ Found {len(results.neighbors)} similar peptides:")
        for i, neighbor in enumerate(results.neighbors[:3], 1):
            print(f"  {i}. {neighbor.peptide_sequence} (similarity: {neighbor.similarity:.3f})")
        
        print(f"\n✅ Search system working correctly!")
        
    except Exception as e:
        print(f"\n❌ Search failed: {e}")
    
    con.close()

if __name__ == "__main__":
    test_search()
