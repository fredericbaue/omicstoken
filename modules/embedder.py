import torch
from bio_embeddings.embed import ProtTransT5XLU50Embedder
import numpy as np

class ProtTransEmbedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProtTransEmbedder, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            print("Loading ProtTransT5XLU50Embedder model...")
            # Check for GPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            # Instantiate the embedder
            self._model = ProtTransT5XLU50Embedder(device=device)
            print("Model loaded.")

    def embed(self, sequence: str) -> np.ndarray:
        """
        Embeds a single peptide sequence using the ProtTransT5XLU50 model.
        Returns a 1024-dimensional vector.
        """
        if not sequence:
            return np.zeros(1024, dtype=np.float32)

        try:
            # The embedder expects a list of sequences
            embedding = self._model.embed(sequence)
            # Reduce dimensions (e.g., by averaging) to get a fixed-size vector
            reduced_embedding = self._model.reduce_per_protein(embedding)
            return reduced_embedding.flatten().astype(np.float32)
        except Exception as e:
            print(f"Error embedding sequence with ProtTransT5: {e}")
            return np.zeros(1024, dtype=np.float32)

# Global instance for easy access
_embedder_instance = None

def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ProtTransEmbedder()
    return _embedder_instance

