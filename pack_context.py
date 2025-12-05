import os
import datetime

# --- CONFIGURATION ---
OUTPUT_FILE = "_SESSION_CONTEXT.txt"

# Files strictly required for Horizon 2 context
FILES_TO_PACK = [
    "tasks.txt",            # The Roadmap
    "memory_log.txt",       # Technical History
    "codex guide.md",       # The Rules
    "db.py",                # The Database Schema (Crucial for H2)
    "app.py",               # The API
    "worker.py",            # The Celery Worker (If exists)
    "requirements.txt"      # The Environment
]

def pack():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        # Header
        out.write(f"# OMICSTOKEN SESSION CONTEXT\n")
        out.write(f"# Generated: {timestamp}\n")
        out.write(f"# ==========================================\n\n")

        for filename in FILES_TO_PACK:
            if os.path.exists(filename):
                out.write(f"\n# --- FILE: {filename} ---\n")
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        out.write(f.read())
                    out.write(f"\n# --- END OF {filename} ---\n")
                    print(f"[OK] Packed: {filename}")
                except Exception as e:
                    print(f"[ERR] Error reading {filename}: {e}")
            else:
                print(f"[WARN] Missing: {filename} (Skipped)")
                out.write(f"\n# --- FILE: {filename} (MISSING) ---\n")

    print(f"\nContext packed into: {OUTPUT_FILE}")

if __name__ == "__main__":
    pack()
