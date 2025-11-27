"""
This script will help you update your .env file with the correct API key.
"""
import os

api_key = "AIzaSyAJ6IpRR0eojz042OylU7mPSpy3SkHs7uI"
env_file_path = ".env"

print("=" * 60)
print("🔧 .env File Update Helper")
print("=" * 60)
print()

# Read existing .env file if it exists
existing_content = []
if os.path.exists(env_file_path):
    print(f"✅ Found existing .env file")
    with open(env_file_path, 'r') as f:
        existing_content = f.readlines()
else:
    print(f"📝 Creating new .env file")

# Remove old GEMINI_API_KEY and GOOGLE_API_KEY lines
new_content = []
removed_old = False
for line in existing_content:
    if line.strip().startswith('GEMINI_API_KEY=') or line.strip().startswith('GOOGLE_API_KEY='):
        removed_old = True
        continue
    new_content.append(line)

# Add the new GOOGLE_API_KEY
new_content.append(f'GOOGLE_API_KEY={api_key}\n')

# Write back to file
with open(env_file_path, 'w') as f:
    f.writelines(new_content)

print()
if removed_old:
    print("✅ Removed old API key configuration")
print(f"✅ Added: GOOGLE_API_KEY={api_key[:20]}...")
print()
print("=" * 60)
print("📋 Your .env file now contains:")
print("=" * 60)
with open(env_file_path, 'r') as f:
    print(f.read())
print("=" * 60)
print()
print("🎉 Configuration complete!")
print()
print("📝 Next steps:")
print("   1. Test the API: python test_gemini_api.py")
print("   2. Start server: uvicorn app:app --reload")
print("   3. Try summarization in the app!")
print()
print("=" * 60)
