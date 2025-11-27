"""
Quick test to verify Gemini API integration works correctly.
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Check if key exists
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY not found in environment!")
    exit(1)
else:
    print(f"✅ API Key found: {api_key[:20]}...")

try:
    # Configure and test the API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Simple test query
    print("\n🧪 Testing Gemini API connection...")
    response = model.generate_content("Say 'Hello from Gemini!' in a scientific tone.")
    
    print(f"✅ Gemini API works!")
    print(f"\n📝 Response:\n{response.text}\n")
    
    print("🎉 All checks passed! Your Gemini integration is ready.")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    exit(1)
