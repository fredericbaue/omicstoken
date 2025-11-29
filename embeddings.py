import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModel
import os

# --- Configuration ---
# ESM-2 (6 layers, 8M parameters) output dimension is 320.
EMBEDDING_DIM = 320

# Global cache to prevent reloading model on every request
_TOKENIZER = None
_MODEL = None

def get_model():
    """Lazy loads the ESM-2 model from Hugging Face cache."""
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        print(f"🔄 Loading ESM-2 Model (facebook/esm2_t6_8M_UR50D)...")
        try:
            model_name = "facebook/esm2_t6_8M_UR50D"
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _MODEL = AutoModel.from_pretrained(model_name)
            _MODEL.eval() # Set to evaluation mode
            print("✅ ESM-2 Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load ESM-2 model: {e}")
            return None, None
    return _TOKENIZER, _MODEL

def peptide_to_vector(sequence: str) -> np.ndarray:
    """
    Embeds a peptide sequence using the LOCAL ESM-2 model.
    Returns a 320-dimensional vector.
    """
    # Validation
    if not sequence or not isinstance(sequence, str):
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    # Clean sequence
    seq = sequence.strip().upper()
    if not seq:
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    # Embedding Logic
    try:
        tokenizer, model = get_model()
        if model is None:
            # Graceful fallback: return zero vector if model is unavailable
            return np.zeros((EMBEDDING_DIM,), dtype=np.float32)
        
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        embeddings = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Mean Pooling over the residue tokens (excluding start/end tokens)
        if embeddings.shape[0] > 2:
             vector = np.mean(embeddings[1:-1], axis=0)
        else:
             vector = np.mean(embeddings, axis=0)
        
        return vector.astype(np.float32)

    except Exception as e:
        # If any part of the PyTorch/Transformers pipeline fails, return zero vector
        logging.error(f"Critical embedding error for {seq}: {e}")
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)