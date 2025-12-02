import os
from typing import Dict, Any, List, Optional

import google.generativeai as genai

import db
import config
import models

# --- Gemini API configuration ---

# Prefer GEMINI_API_KEY but fall back to GOOGLE_API_KEY for compatibility.
_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if _API_KEY:
    genai.configure(api_key=_API_KEY)
else:
    # Keep the existing behavior of warning at import time.
    print("Warning: GEMINI_API_KEY/GOOGLE_API_KEY not found in environment variables!")


def _get_gemini_model() -> Optional[genai.GenerativeModel]:
    """
    Return a configured Gemini model instance, or None if no API key is set.

    This helper centralizes model configuration so all call sites behave
    consistently. It does not raise; higher-level functions decide what
    to do when the model is unavailable.
    """
    if not _API_KEY:
        return None
    # You can change the model name in one place if needed.
    return genai.GenerativeModel("gemini-2.5-flash")


def generate_summary(run_id: str, con=None) -> Dict[str, Any]:
    """
    Generate a natural-language summary for a given run.

    Behavior:
    - If no API key is configured, returns {"error": "...", "run_id": run_id}.
    - If there are no features for the run, returns an "error" dict.
    - Otherwise, delegates to either _generate_tumor_summary or
      _generate_peptide_summary depending on the presence of "stats" metadata.

    Parameters
    ----------
    run_id : str
        The identifier of the run to summarize.
    con : sqlite3.Connection or None
        Optional open DB connection. If None, a new connection is created
        using db.get_db_connection("data").

    Returns
    -------
    Dict[str, Any]
        A summary dictionary. On error, this includes an "error" key.
    """
    model = _get_gemini_model()
    if model is None:
        return {
            "error": "Gemini API key not configured. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.",
            "run_id": run_id,
        }

    if con is None:
        con = db.get_db_connection(config.DATA_DIR)

    # Fetch basic stats / features for the run
    features = db.get_features_for_run(con, run_id)
    if not features:
        return {"error": "No features found for this run.", "run_id": run_id}

    # Check if this is a tumor run with stats in metadata
    is_tumor_run = bool(
        features
        and features[0].metadata
        and "stats" in features[0].metadata
    )

    if is_tumor_run:
        return _generate_tumor_summary(run_id, features, model)
    else:
        return _generate_peptide_summary(run_id, features, model)


def _generate_tumor_summary(
    run_id: str,
    features: List[models.Feature],
    model: genai.GenerativeModel,
) -> Dict[str, Any]:
    """
    Generate a tumor-style differential expression summary using Gemini.

    Returns the same dictionary structure as the previous implementation:
    {
        "run_id": str,
        "type": "tumor_comparison",
        "num_proteins": int,
        "num_significant": int,
        "top_up": [...],
        "top_down": [...],
        "summary_text": str,
    }
    """
    num_proteins = len(features)

    # Extract stats
    sig_proteins: List[models.Feature] = []
    up_regulated: List[Any] = []
    down_regulated: List[Any] = []

    for f in features:
        stats = f.metadata.get("stats", {}) if f.metadata else {}

        # Try to find q-value
        q_val = None
        for k, v in stats.items():
            if "q-value" in k.lower():
                try:
                    q_val = float(v)
                    break
                except (ValueError, TypeError):
                    continue

        if q_val is None:
            # Fallback: look for generic p-value (excluding Wilcoxon / t-test labels)
            for k, v in stats.items():
                if (
                    "p-value" in k.lower()
                    and "wilcoxon" not in k.lower()
                    and "t-test" not in k.lower()
                ):
                    try:
                        q_val = float(v)
                        break
                    except (ValueError, TypeError):
                        continue

        # Try to find fold change (non-log2)
        fc = None
        for k, v in stats.items():
            if "fold change" in k.lower() and "log2" not in k.lower():
                try:
                    fc = float(v)
                    break
                except (ValueError, TypeError):
                    continue

        if q_val is not None and q_val < 0.05:
            sig_proteins.append(f)

        if fc is not None:
            if fc > 0:
                up_regulated.append((f, fc))
            elif fc < 0:
                down_regulated.append((f, fc))

    # Sort up/down-regulated lists
    up_regulated.sort(key=lambda x: x[1], reverse=True)
    down_regulated.sort(key=lambda x: x[1])  # Most negative first

    top_up = up_regulated[:5]
    top_down = down_regulated[:5]

    up_list = "\n".join(
        f"- {f.peptide_sequence}: FC={fc:.2f} ({(f.metadata or {}).get('description', '')})"
        for f, fc in top_up
    )
    down_list = "\n".join(
        f"- {f.peptide_sequence}: FC={fc:.2f} ({(f.metadata or {}).get('description', '')})"
        for f, fc in top_down
    )

    prompt = f"""You are an expert proteomics data analyst. Analyze this differential expression dataset.

Comparison: {run_id} (Inferred from filename/ID)
Total Proteins: {num_proteins}
Significant Proteins (q<0.05): {len(sig_proteins)}

Top Upregulated Proteins:
{up_list}

Top Downregulated Proteins:
{down_list}

Provide a concise scientific summary of these findings. Mention the specific proteins and their potential biological relevance if known.
"""

    try:
        response = model.generate_content(prompt)
        summary_text = response.text

        return {
            "run_id": run_id,
            "type": "tumor_comparison",
            "num_proteins": num_proteins,
            "num_significant": len(sig_proteins),
            "top_up": [
                {
                    "accession": f.peptide_sequence,
                    "fc": fc,
                    "desc": (f.metadata or {}).get("description", ""),
                }
                for f, fc in top_up
            ],
            "top_down": [
                {
                    "accession": f.peptide_sequence,
                    "fc": fc,
                    "desc": (f.metadata or {}).get("description", ""),
                }
                for f, fc in top_down
            ],
            "summary_text": summary_text,
        }
    except Exception as e:
        # Preserve existing behavior: return an error dict, not raise.
        return {
            "error": f"Gemini API Error in tumor summary: {str(e)}",
            "run_id": run_id,
        }


def _generate_peptide_summary(
    run_id: str,
    features: List[models.Feature],
    model: genai.GenerativeModel,
) -> Dict[str, Any]:
    """
    Generate a peptide-focused run summary using Gemini.

    Returns the same dictionary structure as the previous implementation:
    {
        "run_id": ...,
        "num_peptides": ...,
        "avg_intensity": ...,
        "min_intensity": ...,
        "max_intensity": ...,
        "avg_sequence_length": ...,
        "physico_chem": {...},
        "top_peptides": [...],
        "summary_text": ...,
        "llm_model": "gemini-2.5-flash",
    }
    """
    num_peptides = len(features)
    avg_intensity = (
        sum(f.intensity for f in features) / num_peptides if num_peptides > 0 else 0
    )

    # Sort by intensity to find "top" peptides
    sorted_features = sorted(features, key=lambda x: x.intensity, reverse=True)
    top_10 = sorted_features[:10]
    top_sequences = [f.peptide_sequence for f in top_10]
    top_intensities = [f.intensity for f in top_10]

    # Additional statistics
    min_intensity = min(f.intensity for f in features)
    max_intensity = max(f.intensity for f in features)

    # Sequence length distribution
    seq_lengths = [
        len(f.peptide_sequence) if f.peptide_sequence else 0 for f in features
    ]
    avg_seq_length = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0.0

    # --- Enhanced Metrics ---

    # 1. Charge Distribution (Adducts)
    adduct_counts: Dict[str, int] = {}
    for f in features:
        a = getattr(f, "adduct", None) or "Unknown"
        adduct_counts[a] = adduct_counts.get(a, 0) + 1

    # 2. Hydrophobicity (GRAVY score approximation)
    hydropathy = {
        "I": 4.5,
        "V": 4.2,
        "L": 3.8,
        "F": 2.8,
        "C": 2.5,
        "M": 1.9,
        "A": 1.8,
        "G": -0.4,
        "T": -0.7,
        "S": -0.8,
        "W": -0.9,
        "Y": -1.3,
        "P": -1.6,
        "H": -3.2,
        "E": -3.5,
        "Q": -3.5,
        "D": -3.5,
        "N": -3.5,
        "K": -3.9,
        "R": -4.5,
    }

    def calculate_gravy(seq: str) -> float:
        if not seq:
            return 0.0
        score = sum(hydropathy.get(aa.upper(), 0.0) for aa in seq)
        return score / len(seq)

    gravy_scores = [calculate_gravy(f.peptide_sequence) for f in features]
    avg_gravy = sum(gravy_scores) / len(gravy_scores) if gravy_scores else 0.0

    hydrophobic_count = sum(1 for s in gravy_scores if s > 0)
    hydrophilic_count = sum(1 for s in gravy_scores if s <= 0)

    # Build a scientific prompt for Gemini
    peptide_list = "\n".join(
        f"{i+1}. {seq} (Intensity: {int_val:,.0f})"
        for i, (seq, int_val) in enumerate(
            zip(top_sequences[:5], top_intensities[:5])
        )
    )

    adduct_str = ", ".join(f"{k}: {v}" for k, v in adduct_counts.items())

    prompt = f"""You are an expert proteomics data analyst. Analyze this peptide dataset and provide a concise scientific summary.

Dataset: Run ID "{run_id}"

Key Statistics:
- Total Peptides: {num_peptides}
- Intensity: Avg {avg_intensity:,.0f} (Range: {min_intensity:,.0f} - {max_intensity:,.0f})
- Sequence Length: Avg {avg_seq_length:.1f} AA

Physicochemical Properties:
- Hydrophobicity (GRAVY): Avg {avg_gravy:.2f}
- Hydrophobic Peptides: {hydrophobic_count}
- Hydrophilic Peptides: {hydrophilic_count}
- Charge States (Adducts): {adduct_str}

Top 5 Peptides by Abundance:
{peptide_list}

Please provide a 2-3 paragraph summary covering:
1. Overall dataset characteristics and quality.
2. Physicochemical profile (hydrophobicity, charge). What does this suggest about the sample preparation or column type?
3. Biological significance of the top peptides.
4. Recommendations for downstream analysis.

Keep the summary scientific but accessible.
"""

    try:
        response = model.generate_content(prompt)
        summary_text = response.text

        return {
            "run_id": run_id,
            "num_peptides": num_peptides,
            "avg_intensity": avg_intensity,
            "min_intensity": min_intensity,
            "max_intensity": max_intensity,
            "avg_sequence_length": avg_seq_length,
            "physico_chem": {
                "avg_gravy": avg_gravy,
                "hydrophobic_count": hydrophobic_count,
                "hydrophilic_count": hydrophilic_count,
                "adduct_counts": adduct_counts,
            },
            "top_peptides": [
                {
                    "sequence": seq,
                    "intensity": int_val,
                    "feature_id": top_10[i].feature_id,
                }
                for i, (seq, int_val) in enumerate(
                    zip(top_sequences[:10], top_intensities[:10])
                )
            ],
            "summary_text": summary_text,
            "llm_model": "gemini-2.5-flash",
        }
    except Exception as e:
        # Preserve existing behavior: return an error dict, not raise.
        return {
            "error": f"Failed to generate summary: {str(e)}",
            "run_id": run_id,
            "num_peptides": num_peptides,
            "avg_intensity": avg_intensity,
            "top_peptides": top_sequences[:5],
        }
