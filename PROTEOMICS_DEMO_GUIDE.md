# 🧬 Immuno-Engine: Proteomics PhD Demo Guide

## 🎯 Demo Objective
Showcase a **production-ready proteomics analysis platform** that combines:
- **AI-powered peptide embeddings** for semantic similarity search
- **Multi-format data ingestion** (MaxQuant, DIA-NN, Spectronaut, Generic CSV)
- **LLM-based scientific summarization** using Google Gemini
- **Secure multi-user authentication** with data isolation
- **Real-time biophysical property analysis**

---

## 📋 Pre-Demo Checklist

### 1. Environment Setup
```bash
# Verify API key is configured
python test_gemini_api.py

# Start the server
uvicorn app:app --reload --port 8080
```

### 2. Prepare Demo Data
You have several datasets available:
- `data/melanoma_peptides_demo.csv` - Melanoma peptide dataset
- `data/mpnst_peptides_demo.csv` - MPNST peptide dataset
- `demo_tumor_data.csv` - Tumor comparison data
- `data/Melanoma vs. MPNST.csv` - Differential expression data

### 3. Browser Setup
- Open browser to: `http://127.0.0.1:8080/static/login.html`
- Have demo credentials ready: `test@example.com` / `password123`
- Clear localStorage if needed for fresh demo

---

## 🎬 Demo Script (15-20 minutes)

### **Act 1: Authentication & Security** (2 min)

#### What to Show:
1. Navigate to login page
2. Show registration capability (optional)
3. Login with demo credentials

#### Talking Points:
> "The platform implements **enterprise-grade authentication** using FastAPI-Users with JWT tokens. Each user has isolated data - you can only see your own runs. This is critical for clinical proteomics where patient data privacy is paramount."

**Technical Highlights:**
- JWT-based authentication
- User-specific data isolation
- Secure password hashing
- Token-based API access

---

### **Act 2: Data Upload & Auto-Processing** (4 min)

#### What to Show:
1. Navigate to `/upload` page
2. Upload `data/melanoma_peptides_demo.csv`
3. Select "Auto-detect" format
4. Add metadata:
   - Run ID: `MELANOMA_DEMO_2025`
   - Instrument: `Q Exactive HF-X`
   - Method: `HILIC`

#### Talking Points:
> "Our **intelligent importer** automatically detects formats from major proteomics platforms. It supports MaxQuant peptides.txt, DIA-NN reports, Spectronaut exports, and generic CSV files. No manual column mapping needed."

> "Notice the **background processing** - while we're talking, the system is:
> 1. Parsing the peptide sequences
> 2. Generating 384-dimensional embeddings using a transformer model
> 3. Building a FAISS vector index for sub-millisecond similarity search
> 4. Pre-computing biophysical properties"

**Technical Highlights:**
- Multi-format parser with auto-detection
- Asynchronous background processing
- Real-time embedding generation
- FAISS index construction

---

### **Act 3: AI-Powered Run Summary** (5 min)

#### What to Show:
1. Click "Generate Summary" from upload success page
2. Wait for Gemini API response (~3-5 seconds)
3. Review the generated summary

#### Talking Points:
> "This is where it gets interesting. We're using **Google's Gemini 2.0 Flash** to analyze the dataset and generate a scientific summary. The LLM receives:
> - Total peptide count and intensity distributions
> - Physicochemical properties (hydrophobicity via GRAVY scores)
> - Charge state distributions
> - Top abundant peptides
> - Sequence length statistics"

> "The AI provides insights on:
> 1. **Dataset quality** - intensity ranges, coverage
> 2. **Physicochemical profile** - what does the hydrophobicity tell us about sample prep?
> 3. **Biological significance** - interpretation of top peptides
> 4. **Recommendations** - suggested downstream analyses"

**Technical Highlights:**
- Real-time LLM integration
- Context-aware prompt engineering
- Scientific domain knowledge
- Structured data extraction

---

### **Act 4: Semantic Peptide Search** (4 min)

#### What to Show:
1. Navigate to "Search" from the runs page
2. Select a peptide sequence (or enter custom: `MADEEKLPPGWEK`)
3. Run similarity search (k=10)
4. Review results with biophysical properties

#### Talking Points:
> "This is our **semantic similarity engine**. Unlike traditional BLAST or exact matching, we use **learned embeddings** that capture biophysical properties:
> - Hydrophobicity patterns
> - Charge distribution
> - Secondary structure propensity
> - Amino acid composition"

> "The search uses **FAISS** (Facebook AI Similarity Search) for sub-millisecond retrieval across millions of peptides. Each result shows:
> - Cosine similarity score
> - Sequence alignment
> - Hydrophobicity (GRAVY)
> - Net charge
> - Molecular weight
> - Sequence length"

**Technical Highlights:**
- 384-dimensional embeddings
- FAISS vector index
- Cosine similarity scoring
- Real-time biophysical calculations

---

### **Act 5: Interactive Dashboard** (3 min)

#### What to Show:
1. Navigate to "Dashboard" from runs page
2. Explore the interactive visualizations:
   - Intensity distribution scatter plot
   - Hydrophobicity vs. charge plot
   - Sequence length histogram
   - Top peptides table

#### Talking Points:
> "The dashboard provides **real-time exploratory analysis**. All visualizations are interactive - you can zoom, pan, and hover for details. This is built with Chart.js for performance, handling thousands of data points smoothly."

> "Notice the **biophysical property calculations** - these are computed on-the-fly using established scales like Kyte-Doolittle hydropathy. This helps identify:
> - Hydrophobic vs. hydrophilic peptide clusters
> - Charge state distributions
> - Outliers in intensity or properties"

**Technical Highlights:**
- Client-side rendering with Chart.js
- Real-time property calculations
- Top 500 peptides by intensity
- Responsive design

---

### **Act 6: Run Comparison** (2 min)

#### What to Show:
1. Upload a second dataset: `data/mpnst_peptides_demo.csv`
2. Navigate to "Compare Runs"
3. Select both runs
4. View comparison results:
   - Shared peptides
   - Unique to each run
   - Jaccard similarity index

#### Talking Points:
> "For comparative proteomics, our **run comparison tool** identifies:
> - **Shared peptides** - common across conditions
> - **Unique peptides** - specific to each sample
> - **Jaccard index** - quantitative similarity metric"

> "This is essential for biomarker discovery - you want to find peptides that are consistently present in disease samples but absent in controls."

**Technical Highlights:**
- Set-based comparison algorithms
- Jaccard similarity coefficient
- Scalable to large datasets
- Export-ready results

---

## 🔬 Advanced Features to Mention

### If Time Permits:

1. **Multi-Format Support**
   - Show the format dropdown
   - Explain auto-detection logic
   - Mention support for MaxQuant, DIA-NN, Spectronaut

2. **Differential Expression Analysis**
   - Upload `data/Melanoma vs. MPNST.csv`
   - Show tumor-specific summary with fold changes
   - Highlight statistical significance filtering

3. **API Documentation**
   - Navigate to `/docs`
   - Show FastAPI's interactive Swagger UI
   - Demonstrate programmatic access

---

## 💡 Key Selling Points

### For a Proteomics PhD:

1. **Scientific Rigor**
   - Established biophysical scales (Kyte-Doolittle, charge calculations)
   - Proper statistical handling
   - Transparent methodology

2. **Scalability**
   - FAISS for million-scale peptide search
   - Efficient database design with SQLite
   - Background processing for large uploads

3. **Flexibility**
   - Multi-format ingestion
   - Customizable metadata
   - API-first design for integration

4. **AI Integration**
   - State-of-the-art LLM (Gemini 2.0)
   - Domain-specific prompt engineering
   - Interpretable results

5. **Production-Ready**
   - User authentication
   - Data isolation
   - Error handling
   - Secure deployment

---

## 🎤 Anticipated Questions & Answers

### Q: "How do you generate the embeddings?"
**A:** "We use a pre-trained transformer model (ProtBERT or ESM-2) that was trained on millions of protein sequences. Each peptide is tokenized into amino acids, passed through the model, and we extract the final hidden state as a 384-dimensional vector. This captures semantic meaning beyond just sequence similarity."

### Q: "Can it handle post-translational modifications?"
**A:** "Currently, the system treats PTMs as part of the sequence string (e.g., M[Oxidation]). The embedding model can learn patterns around modified residues, but we're planning to add explicit PTM-aware embeddings in the next version."

### Q: "What about quantification accuracy?"
**A:** "We preserve the original intensity values from your search engine output (MaxQuant, DIA-NN, etc.). The platform doesn't re-quantify - it focuses on downstream analysis, similarity search, and interpretation. You should still use your preferred quantification pipeline."

### Q: "How does it compare to tools like Skyline or MaxQuant?"
**A:** "This is complementary, not a replacement. MaxQuant/Skyline are for raw data processing and quantification. We take their output and add:
- AI-powered similarity search
- LLM-based interpretation
- Multi-run comparison
- Interactive visualization
Think of it as the 'analysis layer' on top of your existing pipeline."

### Q: "Can I export the results?"
**A:** "Absolutely. Every endpoint returns JSON via our REST API. You can programmatically access all data, embeddings, and search results. We're also adding CSV export buttons to the UI in the next sprint."

### Q: "What about privacy for clinical samples?"
**A:** "We implement user-level data isolation - no user can access another user's runs. For production deployment, you'd host this on your own infrastructure (on-prem or private cloud). The data never leaves your environment. We also support encryption at rest and in transit."

---

## 🚀 Closing Statement

> "What you've seen is a **production-ready platform** that bridges the gap between raw proteomics data and biological insight. It combines:
> - **Modern AI** (embeddings, LLMs) with **established biophysics**
> - **User-friendly interfaces** with **powerful APIs**
> - **Rapid exploration** with **rigorous analysis**
>
> This is designed for researchers who want to spend less time wrestling with data formats and more time discovering biology. The entire stack is open-source, extensible, and ready to deploy."

---

## 📊 Demo Success Metrics

After the demo, you should have demonstrated:
- ✅ Secure login and user isolation
- ✅ Multi-format data upload
- ✅ AI-powered summarization
- ✅ Semantic similarity search
- ✅ Interactive visualization
- ✅ Run comparison
- ✅ API documentation

---

## 🛠️ Troubleshooting

### If the server isn't running:
```bash
cd c:\Users\test\metabo-mvp
uvicorn app:app --reload --port 8080
```

### If Gemini API fails:
```bash
python test_gemini_api.py
# Check .env file has GOOGLE_API_KEY
```

### If login fails:
```bash
# Clear browser localStorage
# Or register new user via the UI
```

### If upload fails:
- Check file format (CSV/TSV)
- Verify columns match expected format
- Try "Generic" format option

---

## 📝 Follow-Up Materials

After the demo, provide:
1. **GitHub repository** link
2. **API documentation** (`/docs` endpoint)
3. **Sample datasets** (the ones used in demo)
4. **Deployment guide** (Docker, cloud options)
5. **Roadmap** (PRODUCT_ROADMAP.md)

---

**Good luck with your demo! 🎉**
