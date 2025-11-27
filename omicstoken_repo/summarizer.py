import os
import google.generativeai as genai
from typing import Dict, Any, List
import db
import models

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables!")

def generate_summary(run_id: str, con=None) -> Dict[str, Any]:
    """
    Generates a summary for a given run using Gemini API.
    Provides scientific insights about peptide composition and patterns.
    """
    # Check if API key is configured
    if not GEMINI_API_KEY:
        return {
            "error": "GEMINI_API_KEY not configured. Please set it in your .env file.",
            "run_id": run_id
        }
    
    if con is None:
        con = db.get_db_connection(db.get_db_path("data"))
    
    # Fetch basic stats
    features = db.get_features_for_run(con, run_id)
    if not features:
        return {"error": "No features found for this run.", "run_id": run_id}
    
    # Check if this is a tumor run with stats
    is_tumor_run = False
    if features and features[0].metadata and "stats" in features[0].metadata:
        is_tumor_run = True
        
    if is_tumor_run:
        return _generate_tumor_summary(run_id, features)
    else:
        return _generate_peptide_summary(run_id, features)

def _generate_tumor_summary(run_id: str, features: List[models.Feature]) -> Dict[str, Any]:
    num_proteins = len(features)
    
    # Extract stats
    sig_proteins = []
    up_regulated = []
    down_regulated = []
    
    for f in features:
        stats = f.metadata.get("stats", {})
        # Try to find q-value or p-value
        q_val = None
        for k, v in stats.items():
            if "q-value" in k.lower():
                q_val = float(v)
                break
        if q_val is None:
             for k, v in stats.items():
                if "p-value" in k.lower() and "wilcoxon" not in k.lower() and "t-test" not in k.lower():
                    q_val = float(v) # Fallback to p-value
                    break
        
        # Try to find fold change
        fc = None
        for k, v in stats.items():
            if "fold change" in k.lower() and "log2" not in k.lower():
                fc = float(v)
                break
        
        if q_val is not None and q_val < 0.05:
            sig_proteins.append(f)
            
        if fc is not None:
            if fc > 0:
                up_regulated.append((f, fc))
            elif fc < 0:
                down_regulated.append((f, fc))
                
    # Sort up/down
    up_regulated.sort(key=lambda x: x[1], reverse=True)
    down_regulated.sort(key=lambda x: x[1]) # Most negative first
    
    top_up = up_regulated[:5]
    top_down = down_regulated[:5]
    
    # Build prompt
    up_list = "\n".join([f"- {f.peptide_sequence}: FC={fc:.2f} ({f.metadata.get('description', '')})" for f, fc in top_up])
    down_list = "\n".join([f"- {f.peptide_sequence}: FC={fc:.2f} ({f.metadata.get('description', '')})" for f, fc in top_down])
    
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
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        summary_text = response.text
        
        return {
            "run_id": run_id,
            "type": "tumor_comparison",
            "num_proteins": num_proteins,
            "num_significant": len(sig_proteins),
            "top_up": [{"accession": f.peptide_sequence, "fc": fc, "desc": f.metadata.get("description")} for f, fc in top_up],
            "top_down": [{"accession": f.peptide_sequence, "fc": fc, "desc": f.metadata.get("description")} for f, fc in top_down],
            "summary_text": summary_text
        }
    except Exception as e:
        return {"error": f"Gemini API Error: {str(e)}", "run_id": run_id}

def _generate_peptide_summary(run_id: str, features: List[models.Feature]) -> Dict[str, Any]:
    num_peptides = len(features)
    avg_intensity = sum(f.intensity for f in features) / num_peptides if num_peptides > 0 else 0
    
    # Sort by intensity to find "top" peptides
    sorted_features = sorted(features, key=lambda x: x.intensity, reverse=True)
    top_10 = sorted_features[:10]
    top_sequences = [f.peptide_sequence for f in top_10]
    top_intensities = [f.intensity for f in top_10]
    
    # Calculate additional statistics
    min_intensity = min(f.intensity for f in features)
    max_intensity = max(f.intensity for f in features)
    
    # Get sequence length distribution
    seq_lengths = [len(f.peptide_sequence) if f.peptide_sequence else 0 for f in features]
    avg_seq_length = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
    
    # --- Enhanced Metrics ---
    # 1. Charge Distribution (Adducts)
    adduct_counts = {}
    for f in features:
        a = f.adduct if f.adduct else "Unknown"
        adduct_counts[a] = adduct_counts.get(a, 0) + 1
    
    # 2. Hydrophobicity (GRAVY score approximation)
    # Kyte-Doolittle scale
    hydropathy = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
        'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
        'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }
    
    def calculate_gravy(seq):
        if not seq: return 0
        score = sum(hydropathy.get(aa.upper(), 0) for aa in seq)
        return score / len(seq)

    gravy_scores = [calculate_gravy(f.peptide_sequence) for f in features]
    avg_gravy = sum(gravy_scores) / len(gravy_scores) if gravy_scores else 0
    
    # Binning GRAVY scores
    hydrophobic_count = sum(1 for s in gravy_scores if s > 0)
    hydrophilic_count = sum(1 for s in gravy_scores if s <= 0)

    # 3. Clustering (FAISS K-Means)
    # We need embeddings for this. 
    # Since we don't have easy access to embeddings here without querying DB again, 
    # we will skip actual K-Means for this MVP step to avoid circular imports/complexity,
    # OR we can generate them on the fly since we have the code.
    # Let's do a simple sequence-based clustering (e.g. by length or start AA) for the prompt
    # to keep it fast and dependency-free in this module.
    # Actually, let's just report the hydrophobicity clusters.
    
    # Build a scientific prompt for Gemini
    peptide_list = "\n".join([
        f"{i+1}. {seq} (Intensity: {int_val:,.0f})" 
        for i, (seq, int_val) in enumerate(zip(top_sequences[:5], top_intensities[:5]))
    ])
    
    adduct_str = ", ".join([f"{k}: {v}" for k, v in adduct_counts.items()])
    
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
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate content
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
                "adduct_counts": adduct_counts
            },
            "top_peptides": [
                {
                    "sequence": seq,
                    "intensity": int_val,
                    "feature_id": top_10[i].feature_id
                }
                for i, (seq, int_val) in enumerate(zip(top_sequences[:10], top_intensities[:10]))
            ],
            "summary_text": summary_text,
            "llm_model": "gemini-2.5-flash"
        }
    
    except Exception as e:
        return {
            "error": f"Failed to generate summary: {str(e)}",
            "run_id": run_id,
            "num_peptides": num_peptides,
            "avg_intensity": avg_intensity,
            "top_peptides": top_sequences[:5]
        }
