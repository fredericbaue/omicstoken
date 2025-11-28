import sys
import os
import numpy as np
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure we can import from current directory
sys.path.append(os.getcwd())

def test_embedding():
    print("Testing Gemini embedding...")
    
    try:
        from embeddings import peptide_to_vector, EMBEDDING_DIM
        print(f"Expected embedding dimension: {EMBEDDING_DIM}")
        
        seq1 = "ACDEFGH"
        seq2 = "ACDEFGH" # Same
        seq3 = "WVYIQRS" # Different
        
        print(f"Embedding sequence 1: {seq1}")
        start = time.time()
        vec1 = peptide_to_vector(seq1)
        end = time.time()
        print(f"Time taken: {end - start:.4f}s")
        
        print(f"Embedding sequence 2: {seq2}")
        vec2 = peptide_to_vector(seq2)
        
        print(f"Embedding sequence 3: {seq3}")
        vec3 = peptide_to_vector(seq3)
        
        # Check dimensions
        if vec1.shape != (EMBEDDING_DIM,):
            print(f"ERROR: Incorrect dimension. Got {vec1.shape}, expected ({EMBEDDING_DIM},)")
            return
            
        # Check consistency
        if not np.allclose(vec1, vec2):
            print("ERROR: Same sequence produced different embeddings!")
            return
            
        # Check difference
        dist = np.linalg.norm(vec1 - vec3)
        print(f"Distance between seq1 and seq3: {dist:.4f}")
        
        if dist < 0.001:
             print("ERROR: Different sequences produced same embeddings!")
             return

        print("SUCCESS: Embedding test passed!")
        
    except ImportError as e:
        print(f"ImportError: {e}")
        print("Make sure dependencies are installed and you are in the project root.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_embedding()
