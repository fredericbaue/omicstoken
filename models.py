from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Tuple

# -------------------- Data Models --------------------

class HealthResponse(BaseModel):
    ok: bool
    schema: str

class Spectrum(BaseModel):
    mz_array: List[float]
    intensity_array: List[float]

class Feature(BaseModel):
    # This model is now used for Peptides. `peptide_sequence` will hold the sequence.
    # The `alias` is used to map to the 'annotation_name' column in the DB for backward compatibility.
    model_config = ConfigDict(populate_by_name=True)

    feature_id: str
    mz: float
    rt_sec: Optional[float] = None
    intensity: float
    adduct: Optional[str] = None
    polarity: Optional[str] = None
    peptide_sequence: Optional[str] = Field(None, alias="annotation_name") # Maps to annotation_name in DB
    annotation_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Run(BaseModel):
    run_id: str
    user_id: Optional[str] = None
    instrument: Optional[str] = None
    method: Optional[str] = None
    polarity: Optional[str] = None
    schema_version: str
    meta: Dict[str, Any] = {}
    features: Optional[List[Feature]] = None # Optional for when fetching run metadata without all features

class QueryPeptide(BaseModel):
    run_id: str
    feature_id: str
    peptide_sequence: Optional[str] = Field(None, alias="annotation_name")
    intensity: Optional[float] = None

class SimilarPeptide(BaseModel):
    run_id: str
    feature_id: str
    peptide_sequence: Optional[str] = Field(None, alias="annotation_name")
    similarity: float
    intensity: Optional[float] = None

class SimilarPeptidesResponse(BaseModel):
    query: QueryPeptide
    neighbors: List[SimilarPeptide]

# -------------------- OmicsToken Export v1 Schema --------------------
from pydantic import BaseModel
from typing import List, Optional


class OmicsTokenProperties(BaseModel):
    """Biophysical properties for a single peptide."""
    length: int
    charge: Optional[int] = None
    hydrophobicity: Optional[float] = None
    molecular_weight: Optional[float] = None


class OmicsTokenV1(BaseModel):
    """
    Canonical representation of a single peptide embedding record
    in the OmicsToken v1 export format.
    """
    feature_id: str
    sequence: str                      # required
    intensity: Optional[float] = None
    properties: OmicsTokenProperties
    embedding: List[float]             # JSON-serializable list of floats


class OmicsTokenExportV1(BaseModel):
    """
    Top-level response for the OmicsToken v1 export endpoint.
    """
    run_id: str
    export_version: str = "omics_export_v1"
    total_embeddings: int
    data: List[OmicsTokenV1]
