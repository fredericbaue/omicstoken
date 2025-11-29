"""
This module handles the logic for exporting data in canonical, versioned formats.
"""
from typing import List, Dict, Any
import logging

# --- Contract Definition ---
# I/O Contract for calculate_molecular_weight
# - Input: peptide_sequence (str)
# - Output: molecular_weight (float)
# - Side Effects: None
# - Error Cases: Returns 0.0 for invalid or empty sequences.

# Average masses of amino acids. Source: IUPAC. Water (H2O) is added to account for terminal groups.
_AMINO_ACID_MASSES = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
    'U': 168.05, 'O': 255.31 # Selenocysteine and Pyrrolysine
}
_WATER_MASS = 18.015

def calculate_molecular_weight(sequence: str) -> float:
    """Calculates the molecular weight of a peptide sequence."""
    if not sequence or not isinstance(sequence, str):
        return 0.0
    
    total_mass = _WATER_MASS
    for aa in sequence.upper():
        total_mass += _AMINO_ACID_MASSES.get(aa, 0.0)
    
    return total_mass


import models
from embeddings import EMBEDDING_DIM

# --- Contract Definition ---
# I/O Contract for normalize_to_omics_token_v1
# - Input:
#   - run_id (str): The ID of the run being exported.
#   - raw_embeddings (List[Dict[str, Any]]): Data from db.get_peptide_embeddings.
# - Output: A List of `models.OmicsTokenV1` Pydantic objects.
# - Side Effects: Logs a warning for any record with a malformed embedding.
# - Error Cases: Returns an empty list if input is empty.
# export.py
import logging
from typing import Any, Dict, List

import models
from embeddings import EMBEDDING_DIM


def _estimate_molecular_weight(sequence: str) -> float:
    """
    Very rough MW estimate: 110 Da per residue.
    Good enough for export / dashboards, not for publications.
    """
    if not sequence:
        return 0.0
    return float(len(sequence) * 110)


def normalize_to_omics_token_v1(
    run_id: str,
    raw_embeddings: List[Dict[str, Any]],
) -> List[models.OmicsTokenV1]:
    """
    Normalize raw peptide embedding rows from the database into
    the canonical OmicsToken v1 format.

    - Skips records with missing sequence.
    - Skips records whose embedding is not a list[float] of EMBEDDING_DIM.
    """
    normalized: List[models.OmicsTokenV1] = []

    for record in raw_embeddings:
        feature_id = record.get("feature_id")
        sequence = record.get("sequence")
        embedding_vec = record.get("embedding")

        # 1. Required: sequence
        if not sequence:
            logging.warning(
                f"[OmicsToken v1] Skipping feature_id={feature_id!r} in run={run_id!r}: "
                "missing sequence."
            )
            continue

        # 2. Validate embedding vector
        if not isinstance(embedding_vec, list) or len(embedding_vec) != EMBEDDING_DIM:
            logging.warning(
                f"[OmicsToken v1] Skipping feature_id={feature_id!r} in run={run_id!r}: "
                f"invalid embedding (type={type(embedding_vec)}, "
                f"len={len(embedding_vec) if isinstance(embedding_vec, list) else 'N/A'})."
            )
            continue

        # 3. Build properties payload
        length = record.get("length")
        charge = record.get("charge")
        hydrophobicity = record.get("hydrophobicity")

        props = models.OmicsTokenProperties(
            length=int(length) if length is not None else len(sequence),
            charge=int(charge) if charge is not None else None,
            hydrophobicity=float(hydrophobicity) if hydrophobicity is not None else None,
            molecular_weight=_estimate_molecular_weight(sequence),
        )

        # 4. Build canonical record
        token = models.OmicsTokenV1(
            feature_id=str(feature_id),
            sequence=sequence,
            intensity=(
                float(record.get("intensity"))
                if record.get("intensity") is not None
                else None
            ),
            properties=props,
            embedding=[float(x) for x in embedding_vec],
        )

        normalized.append(token)

    return normalized
