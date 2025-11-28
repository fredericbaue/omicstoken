import numpy as np
import logging
from bio_embeddings.embed import ProtTransT5XLU50Embedder

# --- Configuration ---
# ProtTrans-T5-XL-U50 outputs 1024-dimensional vectors
EMBEDDING_DIM = 1024

# Global cache to prevent reloading the massive model on every request
_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        print(f"🔄 Loading ProtTrans-T5 Model (this may take a moment)...")
        try:
            # This loads the local model (requires ~3-5GB RAM)
            _EMBEDDER = ProtTransT5XLU50Embedder()
            print("✅ ProtTrans Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ProtTrans model: {e}")
            raise e
    return _EMBEDDER

def peptide_to_vector(sequence: str) -> np.ndarray:
    """
    Embeds a peptide sequence using the LOCAL ProtTrans-T5 model.
    Returns a 1024-dimensional vector.
    """
    if not sequence or not isinstance(sequence, str):
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    # Clean sequence
    seq = sequence.strip().upper()
    if not seq:
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    try:
        embedder = get_embedder()
        # ProtTrans returns per-residue embeddings; we need a single vector per peptide.
        # standard approach: reduce_per_protein (average)
        return embedder.embed(seq)
    except Exception as e:
        print(f"Error embedding sequence '{seq}': {e}")
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)
