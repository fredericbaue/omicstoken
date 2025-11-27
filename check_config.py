"""
Quick API Configuration Helper
Run this to check your current setup and get instructions
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("🔍 GEMINI API CONFIGURATION CHECK")
print("=" * 60)

# Check for old variable
old_key = os.getenv("GEMINI_API_KEY")
if old_key:
    print("⚠️  WARNING: Found GEMINI_API_KEY in your .env file")
    print("   This is the OLD variable name and won't work!")
    print()

# Check for new variable
new_key = os.getenv("GOOGLE_API_KEY")
if new_key:
    print(f"✅ GOOGLE_API_KEY found (starts with: {new_key[:10]}...)")
    print("   Configuration looks correct!")
else:
    print("❌ GOOGLE_API_KEY not found!")
    print()
    print("📝 ACTION REQUIRED:")
    print("   1. Open your .env file")
    if old_key:
        print("   2. Change: GEMINI_API_KEY=... ")
        print("      To:     GOOGLE_API_KEY=...")
    else:
        print("   2. Add this line: GOOGLE_API_KEY=your_api_key_here")
    print("   3. Save the file")
    print("   4. Run: python test_gemini_api.py")

print()
print("=" * 60)
print("📚 QUICK REFERENCE")
print("=" * 60)
print("Environment Variable: GOOGLE_API_KEY")
print("Model Name:          gemini-2.0-flash-exp")
print("Test Command:        python test_gemini_api.py")
print("Start Server:        uvicorn app:app --reload")
print("=" * 60)
