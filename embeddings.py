import numpy as np
from modules.embedder import get_embedder

# --- Configuration ---
# Dimension of the peptide embedding vector.
EMBEDDING_DIM = 1024

def peptide_to_vector(sequence: str) -> np.ndarray:
    """
    Embeds a peptide sequence using the Gemini API (text-embedding-004).
    Returns a 768-dimensional vector.
    """
    if not sequence or not isinstance(sequence, str):
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    embedder = get_embedder()
    try:
        return embedder.embed(sequence)
    except Exception as e:
        print(f"Error embedding sequence '{sequence}': {e}")
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)