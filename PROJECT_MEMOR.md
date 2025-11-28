# 🧠 OmicsToken Project Memory Core
**Last Updated:** Phase I Completion (The "Ferrari" Build)

## 🚦 Current Status
* **Phase:** Transitioning from Phase I (MVP) to Phase II (Tokenization).
* **Stability:** Stable. Windows dependency hell (Argon2/Bio-Embeddings) has been resolved.
* **Next Immediate Goal:** Prototype Phase II.1 (ERC-20 Token Smart Contracts).

## 🏛️ The "Trillion Dollar" Strategy
1.  **Phase I: The Tool (✅ DONE)**
    * Build a "Google for Biology" using AI embeddings (ESM-2) to find semantic peptide matches.
    * *Value:* High-utility SaaS for researchers.
2.  **Phase II: The Bank (🚧 STARTING NOW)**
    * Tokenize data uploads. Pay researchers in "OmicsToken" (OMT) for high-quality data.
    * *Value:* Accumulate a "Data Monopoly" of biological assets.
3.  **Phase III: The Utility (FUTURE)**
    * Sell compute/query access to Pharma. Create a decentralized market for drug discovery.
    * *Value:* Global infrastructure for biotech.

## ⚙️ Critical Technical Decisions (DO NOT REVERT)
*If you change these, the app will crash on Windows.*

### 1. Embedding Engine
* **Model:** ESM-2 (`facebook/esm2_t6_8M_UR50D`).
* **Library:** `transformers` + `torch`.
* **Vector Dimension:** **320**. (Do not use 1024; that was for ProtTrans).
* **FORBIDDEN LIBRARY:** `bio-embeddings`. (Reason: Fails to compile `gensim` on Windows. Do not add to `requirements.txt`).

### 2. Authentication & Security
* **Hasher:** `bcrypt` (via `passlib`).
* **Config:** Explicitly set `CryptContext(schemes=["bcrypt"])` in `auth.py`.
* **Dependencies:** `argon2-cffi` is installed ONLY to satisfy `fastapi-users` import checks, but it is **not** used for hashing.
* **FORBIDDEN LIBRARY:** `pwdlib[argon2]`. (Reason: Causes `_cffi_backend` crash on Windows).

### 3. Database
* **Engine:** SQLite + `aiosqlite`.
* **Dependency:** Requires `greenlet` to bridge async/sync calls.
* **Schema:** `User` table includes `credits` (Integer) to track Shadow Tokenomics.

## 🛠️ How to Start the Server (The "Ferrari" Protocol)
1.  **Install:** `pip install -r requirements.txt`
2.  **Run:** `python -m uvicorn app:app --reload --port 8080`
3.  **Verify:** Login -> Upload Data -> Check if "Credits" increase by 10.

## 🔮 Next Steps (Phase II)
* [ ] Design `OmicsToken` (ERC-20) Smart Contract.
* [ ] Design `DataNFT` (ERC-721) for datasets.
* [ ] Connect Python backend to Blockchain (Web3.py).
