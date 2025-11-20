import numpy as np

# --- Configuration ---
# Dimension of the peptide embedding vector.
EMBEDDING_DIM = 16

_AMINO_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AMINO_INDEX = {aa: i for i, aa in enumerate(_AMINO_ALPHABET)}


def _sequence_to_basic_features(sequence: str) -> np.ndarray:
    """
    Very simple, fast, *local* embedder that does NOT require external libraries.

    It turns a peptide into a small numeric vector:
      - length
      - mean position in alphabet
      - fraction of charged residues (D, E, K, R, H)
      - fraction of hydrophobic residues (A, V, I, L, M, F, Y, W)
      - plus a small fixed-length bag-of-amino-acids representation
    """
    if not sequence or not isinstance(sequence, str):
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    seq = sequence.strip().upper()
    if not seq:
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    length = len(seq)

    charged = set("DEKRH")
    hydrophobic = set("AVILMFWY")

    charged_count = sum(1 for aa in seq if aa in charged)
    hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic)

    charged_frac = charged_count / length
    hydrophobic_frac = hydrophobic_count / length

    indices = [_AMINO_INDEX.get(aa, 0) for aa in seq]
    mean_idx = float(sum(indices)) / length

    bag_dim = EMBEDDING_DIM - 4  # we already used 4 slots above
    bag = np.zeros((bag_dim,), dtype=np.float32)

    for aa in seq:
        idx = _AMINO_INDEX.get(aa)
        if idx is not None and idx < bag_dim:
            bag[idx] += 1.0

    if bag.sum() > 0:
        bag = bag / bag.sum()

    vec = np.concatenate(
        [
            np.array(
                [length, mean_idx, charged_frac, hydrophobic_frac],
                dtype=np.float32,
            ),
            bag.astype(np.float32),
        ]
    )

    if vec.shape[0] != EMBEDDING_DIM:
        vec = np.resize(vec, (EMBEDDING_DIM,)).astype(np.float32)

    return vec


def peptide_to_vector(sequence: str) -> np.ndarray:
    """
    Public function used by the rest of the app.

    Later, I can replace this implementation with a real ProtTrans-based embedder,
    but for now it's a lightweight, dependency-free embedding that always returns
    an EMBEDDING_DIM-dimensional vector.
    """
    return _sequence_to_basic_features(sequence)