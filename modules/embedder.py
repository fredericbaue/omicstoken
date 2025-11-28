import torch
import re
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel

class ProtTransEmbedder:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProtTransEmbedder, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            print("Loading ProtTrans T5 Model (via transformers)...")
            
            # Detect device
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self._device}")

            # Load model and tokenizer directly from Hugging Face
            # This uses the same model bio-embeddings uses under the hood
            model_name = "Rostlab/prot_t5_xl_uniref50"
            
            self._tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            self._model = T5EncoderModel.from_pretrained(model_name).to(self._device)
            
            # Set to eval mode to save memory
            self._model.eval()
            
            # Garbage collect to free memory
            if self._device.type == 'cuda':
                torch.cuda.empty_cache()
                
            print("Model loaded successfully.")

    def embed(self, sequence: str) -> np.ndarray:
        """
        Embeds a single peptide sequence.
        Returns a 1024-dimensional vector (averaged over the sequence).
        """
        if not sequence:
            return np.zeros(1024, dtype=np.float32)

        # Clean sequence (replace rare AAs with X, add spaces for T5 tokenizer)
        clean_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

        try:
            # Tokenize
            ids = self._tokenizer.batch_encode_plus(
                [clean_seq], 
                add_special_tokens=True, 
                padding="longest", 
                return_tensors='pt'
            ).to(self._device)

            # Generate embeddings (no gradient needed)
            with torch.no_grad():
                embedding_repr = self._model(ids.input_ids, attention_mask=ids.attention_mask)

            # Extract the last hidden state
            # shape: (1, seq_len, 1024)
            emb = embedding_repr.last_hidden_state.detach().cpu().numpy()

            # Remove special tokens (start/end) before averaging? 
            # Bio-embeddings usually averages the whole thing. 
            # For simplicity and robustness, we average the valid tokens.
            
            # Simply take the mean across the sequence length dimension (dim 1)
            # This creates a single vector per protein
            reduced_embedding = np.mean(emb[0], axis=0)

            return reduced_embedding.astype(np.float32)

        except Exception as e:
            print(f"Error embedding sequence with ProtTrans: {e}")
            return np.zeros(1024, dtype=np.float32)

# Global instance for easy access
_embedder_instance = None

def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ProtTransEmbedder()
    return _embedder_instance
