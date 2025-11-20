import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from models import Feature

def parse_tumor_csv(df: pd.DataFrame) -> List[Feature]:
    """
    Parses tumor proteomics CSVs with stats columns.
    Expected columns: Accession, Description, samples..., Fold Change, q-value, etc.
    """
    features = []
    
    # Identify sample columns (columns that are not stats or metadata)
    # This is a heuristic: exclude known non-sample columns
    exclude_cols = {
        "Accession", "Description", "Gene Symbol", "Entrez Gene ID", 
        "Fold Change", "Log2 Fold Change", "p-value", "q-value", 
        "Wilcoxon p-value", "t-test p-value", "Resampled p-value",
        "Significant", "Pi Score"
    }
    
    # Normalize columns for easier matching
    df_cols_map = {c.lower(): c for c in df.columns}
    
    # Find key columns
    acc_col = df_cols_map.get("accession")
    desc_col = df_cols_map.get("description")
    
    if not acc_col:
        # Fallback: try to find a column that looks like an ID
        for c in df.columns:
            if "accession" in c.lower() or "protein" in c.lower() or "id" in c.lower():
                acc_col = c
                break
    
    if not acc_col:
        raise ValueError("Could not identify Accession/ID column in tumor CSV.")

    # Identify stats columns
    stats_cols = []
    for c in df.columns:
        lower_c = c.lower()
        if "fold change" in lower_c or "p-value" in lower_c or "q-value" in lower_c or "score" in lower_c:
            stats_cols.append(c)
            
    # Identify sample columns (everything else)
    sample_cols = [c for c in df.columns if c not in exclude_cols and c not in stats_cols and c != acc_col and c != desc_col]
    
    for i, row in df.iterrows():
        accession = str(row[acc_col])
        description = str(row[desc_col]) if desc_col else ""
        
        # Calculate a representative intensity (e.g., max or mean of samples)
        # For the MVP, we'll just take the max to have something non-zero
        intensities = [float(row[c]) for c in sample_cols if pd.to_numeric(row[c], errors='coerce') >= 0]
        rep_intensity = max(intensities) if intensities else 0.0
        
        # Collect stats into metadata
        meta = {
            "description": description,
            "stats": {c: row[c] for c in stats_cols},
            "samples": {c: row[c] for c in sample_cols}
        }
        
        features.append(Feature(
            feature_id=f"TUMOR_{i}_{accession}",
            peptide_sequence=accession, # Using Accession as sequence for now
            intensity=rep_intensity,
            mz=0.0, # Not applicable
            rt_sec=None,
            adduct=None,
            polarity=None,
            metadata=meta
        ))
        
    return features

def detect_format(df: pd.DataFrame) -> str:
    """
    Simple heuristic to detect the format of the uploaded CSV.
    """
    cols = set(df.columns)
    # Check for tumor format specific columns
    lower_cols = {c.lower() for c in cols}
    if "accession" in lower_cols and ("fold change" in lower_cols or "q-value" in lower_cols):
        return "tumor_csv"
        
    if "Sequence" in cols and "Intensity" in cols and "Modified sequence" in cols:
        return "maxquant"
    if "Precursor.Id" in cols and "Stripped.Sequence" in cols:
        return "diann"
    if "PEP.StrippedSequence" in cols:
        return "spectronaut"
    return "generic"

def parse_upload(df: pd.DataFrame, fmt: str = "auto") -> List[Feature]:
    if fmt == "auto":
        fmt = detect_format(df)
    
    if fmt == "tumor_csv":
        return parse_tumor_csv(df)
    elif fmt == "maxquant":
        return parse_maxquant(df)
    elif fmt == "diann":
        return parse_diann(df)
    elif fmt == "spectronaut":
        return parse_spectronaut(df)
    else:
        return parse_generic(df)

def parse_generic(df: pd.DataFrame) -> List[Feature]:
    """
    Parses a generic CSV with 'feature_id', 'peptide_sequence', 'intensity' columns.
    """
    features = []
    # Normalize column names to lowercase for easier matching
    df.columns = [c.lower() for c in df.columns]
    
    # Map common variations
    col_map = {
        "feature_id": ["feature_id", "id", "peptide_id"],
        "peptide_sequence": ["peptide_sequence", "sequence", "annotation_name", "seq"],
        "intensity": ["intensity", "area", "abundance"],
        "mz": ["mz", "m/z", "mass_to_charge"],
        "rt_sec": ["rt_sec", "rt", "retention_time"],
        "adduct": ["adduct", "charge_state"], # approximate
        "polarity": ["polarity"]
    }

    def get_col(variations):
        for v in variations:
            if v in df.columns:
                return v
        return None

    fid_col = get_col(col_map["feature_id"])
    seq_col = get_col(col_map["peptide_sequence"])
    int_col = get_col(col_map["intensity"])
    mz_col = get_col(col_map["mz"])
    rt_col = get_col(col_map["rt_sec"])

    if not (fid_col and seq_col and int_col):
        raise ValueError(f"Generic CSV missing required columns. Found: {list(df.columns)}")

    for i, row in df.iterrows():
        features.append(Feature(
            feature_id=str(row[fid_col]),
            peptide_sequence=str(row[seq_col]),
            intensity=float(row[int_col]),
            mz=float(row[mz_col]) if mz_col and pd.notna(row[mz_col]) else 0.0,
            rt_sec=float(row[rt_col]) if rt_col and pd.notna(row[rt_col]) else None,
            adduct=None,
            polarity=None,
            annotation_score=None
        ))
    return features

def parse_maxquant(df: pd.DataFrame) -> List[Feature]:
    """
    Parses MaxQuant peptides.txt format.
    """
    features = []
    # MaxQuant usually has 'Sequence', 'Intensity', 'id'
    for i, row in df.iterrows():
        features.append(Feature(
            feature_id=str(row.get("id", f"MQ_{i}")),
            peptide_sequence=str(row.get("Sequence", "")),
            intensity=float(row.get("Intensity", 0.0)),
            mz=float(row.get("Mass", 0.0)), # Mass is not m/z but close enough for MVP placeholder
            rt_sec=float(row.get("Retention time", 0.0)) * 60 if "Retention time" in df.columns else None,
            adduct=str(row.get("Charge", "")),
            polarity=None
        ))
    return features

def parse_diann(df: pd.DataFrame) -> List[Feature]:
    """
    Parses DIA-NN report.tsv format.
    """
    features = []
    for i, row in df.iterrows():
        features.append(Feature(
            feature_id=str(row.get("Precursor.Id", f"DIA_{i}")),
            peptide_sequence=str(row.get("Stripped.Sequence", "")),
            intensity=float(row.get("Precursor.Quantity", 0.0)),
            mz=float(row.get("Precursor.Mz", 0.0)),
            rt_sec=float(row.get("RT", 0.0)) * 60,
            adduct=str(row.get("Precursor.Charge", "")),
            polarity=None
        ))
    return features

def parse_spectronaut(df: pd.DataFrame) -> List[Feature]:
    """
    Parses Spectronaut export format.
    """
    features = []
    for i, row in df.iterrows():
        features.append(Feature(
            feature_id=f"SPEC_{i}", # Spectronaut exports might vary, using index for now
            peptide_sequence=str(row.get("PEP.StrippedSequence", "")),
            intensity=float(row.get("PEP.Quantity", 0.0)),
            mz=0.0,
            rt_sec=float(row.get("PEP.RT", 0.0)) * 60,
            adduct=None,
            polarity=None
        ))
    return features


