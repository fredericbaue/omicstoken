# 🎙️ Sunday Demo Master Plan

## 1. Pre-Flight Check (15 Mins Before)
1. **Clean Start:** Close all terminals and browser windows.
2. **Launch:** Open PowerShell and run `.\launch.ps1`.
3. **Verify:**
   - Wait for "Application startup complete".
   - Open Chrome Incognito (to avoid caching issues).
   - Go to: `http://localhost:8080/static/login.html`
   - Log in: `demo@omicstoken.com` / `password123`
   - **Crucial:** Ensure the runs page loads without a dark overlay.

## 2. The Demo Script
**Step 1: The "Hook" (Upload)**
- Navigate to **Upload**.
- Select `demo_data/demo_tumor_data.csv`.
- *Narrative:* "We take raw metabolomic data—usually just messy text..."
- Click **Upload**.
- *Action:* Watch the status change from WAITING -> READY automatically (approx 5s).

**Step 2: The "Intelligence" (Summary)**
- Click the **Summary** button on the new run.
- *Narrative:* "Instead of manually analyzing 3,500 rows, our engine generates a biological profile instantly."
- *Action:* Wait for the modal to pop (Gemini generates synchronously).

**Step 3: The "Deep Dive" (Search)**
- Click **Search**.
- Query: `SIINFEKL` (or a known peptide from your data).
- *Narrative:* "We can now find similar patients based on vector math, not just keyword matching."

## 3. Emergency Troubleshooting
- **Dark Screen?** Hard Refresh (Ctrl+F5) or use Incognito.
- **Stuck on Waiting?** Refresh the page manually (Auto-polling usually handles this).
- **Summary Failed?** Check the PowerShell logs. If API fails, show the Search feature instead.
- **Everything Frozen?** Ctrl+C the terminal, run `.\launch.ps1` again.
