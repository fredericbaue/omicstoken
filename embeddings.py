import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import config

# --- Configuration ---
# ESM-2 (6 layers, 8M parameters) output dimension is 320.
EMBEDDING_DIM = 320

# Global cache to prevent reloading model on every request
_TOKENIZER = None
_MODEL = None
_LOGGER = logging.getLogger(__name__)


def _zero_vector() -> np.ndarray:
    """Return a deterministic zero vector of the correct dimension."""
    return np.zeros((EMBEDDING_DIM,), dtype=np.float32)


def get_model():
    """Lazy loads the ESM-2 model from Hugging Face cache with global caching."""
    global _TOKENIZER, _MODEL
    if _MODEL is None or _TOKENIZER is None:
        _LOGGER.info("Loading ESM-2 Model (%s)...", config.EMBEDDING_MODEL_NAME)
        try:
            model_name = config.EMBEDDING_MODEL_NAME
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _MODEL = AutoModel.from_pretrained(model_name)
            _MODEL.eval()  # Set to evaluation mode
            _LOGGER.info("ESM-2 Model loaded successfully.")
        except Exception as e:
            _LOGGER.error(f"Failed to load ESM-2 model: {e}")
            return None, None
    return _TOKENIZER, _MODEL


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""

    @abstractmethod
    def embed(self, sequence: str) -> np.ndarray:
        """Embed a sequence into a vector of length EMBEDDING_DIM."""
        raise NotImplementedError


class Esm2Embedder(BaseEmbedder):
    """ESM-2 embedder with global model/tokenizer cache and deterministic fallbacks."""

    def embed(self, sequence: str) -> np.ndarray:
        # Validation and normalization
        if not isinstance(sequence, str):
            _LOGGER.warning("Received non-string sequence; returning zero vector.")
            return _zero_vector()

        seq = sequence.strip().upper()
        if not seq:
            _LOGGER.warning("Received empty/blank sequence; returning zero vector.")
            return _zero_vector()

        try:
            tokenizer, model = get_model()
            if model is None or tokenizer is None:
                _LOGGER.error("ESM-2 model unavailable; returning zero vector.")
                return _zero_vector()

            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)

            with torch.no_grad():
                outputs = model(**inputs)

            embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            # Mean Pooling over the residue tokens (excluding start/end tokens)
            if embeddings.shape[0] > 2:
                vector = np.mean(embeddings[1:-1], axis=0)
            else:
                vector = np.mean(embeddings, axis=0)

            if vector.shape[0] != EMBEDDING_DIM:
                _LOGGER.error(
                    "Embedding dimension mismatch: expected %s, got %s. Returning zero vector.",
                    EMBEDDING_DIM,
                    vector.shape[0],
                )
                return _zero_vector()

            return vector.astype(np.float32)

        except Exception as e:
            _LOGGER.error(f"Critical embedding error for {seq}: {e}")
            return _zero_vector()


# Default embedder instance for current flows (factory-driven)
def create_embedder(name: str = None) -> BaseEmbedder:
    """Factory for embedders. Default is Esm2Embedder."""
    embedder_name = (name or config.EMBEDDER_NAME or "esm2").lower()
    if embedder_name == "esm2":
        return Esm2Embedder()
    raise ValueError(f"Unknown embedder '{embedder_name}'. Supported: esm2")


_DEFAULT_EMBEDDER: BaseEmbedder = create_embedder()


def peptide_to_vector(sequence: str) -> np.ndarray:
    """
    Backward-compatible wrapper that delegates to the default embedder.
    """
    return _DEFAULT_EMBEDDER.embed(sequence)
