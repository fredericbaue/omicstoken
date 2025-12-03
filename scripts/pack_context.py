import os
import subprocess
import sys

# The "Brain" of the project - add files here as we touch them
CRITICAL_FILES = [
    # Documentation & Tracking
    "tasks.txt",
    "memory_log.txt",
    "core_context.txt",
    
    # Core Backend
    "app.py",
    "worker.py",
    "db.py",
    "config.py",
    "models.py",
    "export.py",
    
    # Key Frontend (Horizon 1)
    "static/runs.html",
    "static/dashboard.html",
    "static/upload.html",
    "static/style.css",
]

def read_file(filepath):
    """Reads a file and returns formatted block or error placeholder."""
    if not os.path.exists(filepath):
        return f"[MISSING FILE: {filepath}]\n"
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            ext = filepath.split('.')[-1]
            return f"\n--- START FILE: {filepath} ---\n```{ext}\n{content}\n```\n--- END FILE: {filepath} ---\n"
    except Exception as e:
        return f"[ERROR READING {filepath}: {str(e)}]\n"

def main():
    print(f"📦 Packing {len(CRITICAL_FILES)} critical files...")
    
    # 1. Header
    output = []
    output.append("# OMICSTOKEN CONTEXT DUMP\n")
    output.append(f"# Generated via scripts/pack_context.py\n")
    
    # 2. Pack Files
    for filepath in CRITICAL_FILES:
        output.append(read_file(filepath))
        print(f"  - Packed: {filepath}")
    
    full_text = "".join(output)
    
    # 3. Copy to Clipboard (Windows specific)
    # Uses the built-in 'clip' command so no pip install required
    try:
        process = subprocess.Popen('clip', stdin=subprocess.PIPE, shell=True)
        process.communicate(input=full_text.encode('utf-16')) # clip expects utf-16 usually
        print("\n✅ SUCCESS: Context copied to clipboard!")
        print("   -> Go to Gemini and press Ctrl+V")
    except Exception as e:
        print(f"\n⚠️  Clipboard copy failed: {e}")
        print("   -> Writing to '_context_dump.txt' instead.")
        with open("_context_dump.txt", "w", encoding="utf-8") as f:
            f.write(full_text)

if __name__ == "__main__":
    main()
