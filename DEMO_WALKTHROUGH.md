# 🧪 Immuno-Engine MVP: Demo Walkthrough

This guide will help you demonstrate the **Immuno-Engine** to your professor. It highlights the end-to-end workflow: **Ingestion -> AI Analysis -> Scientific Insight**.

## 1. Preparation

1.  **Start the Server**:
    Open your terminal and run:
    ```bash
    uvicorn app:app --reload
    ```
2.  **Open the Browser**:
    Navigate to `http://127.0.0.1:8000/upload`.

## 2. The Pitch (What to say)

> "Professor, this is the Immuno-Engine. It's designed to move beyond simple peptide matching. It uses a **Protein Language Model (ProtTrans)** to understand the biophysical properties of peptides and an **AI Agent** to summarize findings automatically."

## 3. The Demo Steps

### Step 1: Intelligent Ingestion
1.  On the **Upload Page** (`/upload`), click **Choose File**.
2.  Select the `demo_tumor_data.csv` file (I just created this for you in the project root).
3.  **Key Point**: Point out the "Format" dropdown but leave it on **Auto-detect**.
    > "The system automatically detects formats like MaxQuant, DIA-NN, or custom tumor datasets with statistical columns."
4.  Enter a Run ID: `TUMOR_DEMO_01`.
5.  Click **Upload & Ingest**.

### Step 2: The "Black Box" (Embedding)
1.  You will see the **Upload Complete** screen.
2.  **Key Point**: Point to the text "Embedding running in background...".
    > "Right now, the system is converting these peptide sequences into high-dimensional vectors using the T5 model. This allows us to search for peptides with similar properties, not just similar sequences."

### Step 3: The Dashboard (Visualization)
1.  Open a new tab to: `http://127.0.0.1:8000/static/dashboard.html`
2.  In the sidebar, enter the Run ID: `TUMOR_DEMO_01`.
3.  Click **LOAD RUN**.
4.  **The Wow Moment**:
    *   The **Volcano Plot** (Figure 1) will appear, showing Up/Down regulated proteins (Red/Blue).
    *   Click on a **Red dot** (Upregulated).
    *   The **Detail View** (bottom pane) will populate.
    *   Show the **Hydrophobicity Bar**:
        > "We instantly calculate biophysical properties like Hydrophobicity and Molecular Weight for every hit."

### Step 4: AI Summarization (Optional / If API Key is set)
1.  Go back to the **Upload Complete** page.
2.  Click the **Generate Summary for this Run** button.
3.  If you have the Gemini API key set, it will generate a paragraph describing the upregulated proteins (like Alpha-1-antitrypsin and CRP) and their biological significance.

## 4. Wrap Up
> "This MVP demonstrates how we can combine modern LLMs with traditional proteomics to accelerate insight generation."
