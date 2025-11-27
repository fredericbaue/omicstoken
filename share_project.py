import os

# Files to ignore
IGNORE_DIRS = {'.venv', 'venv', '__pycache__', '.git', 'data', 'node_modules'}
IGNORE_FILES = {'.env', 'project_context.txt', 'share_project.py', 'uniprot_data.csv', 'metabo.db'}
EXTENSIONS = {'.py', '.html', '.css', '.js', '.sql'}

output_file = "project_context.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write(f"--- PROJECT STRUCTURE ---\n")
    for root, dirs, files in os.walk("."):
        # Filter directories in place
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES: continue
            if not any(file.endswith(ext) for ext in EXTENSIONS): continue
            
            path = os.path.join(root, file)
            outfile.write(f"{path}\n")

    outfile.write(f"\n\n--- FILE CONTENTS ---\n")
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES: continue
            if not any(file.endswith(ext) for ext in EXTENSIONS): continue
            
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(f"\n\n{'='*20}\nFILE: {path}\n{'='*20}\n")
                    outfile.write(content)
            except Exception as e:
                print(f"Skipping {path}: {e}")

print(f"Done! Please upload '{output_file}' to the chat.")