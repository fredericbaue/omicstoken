import google.generativeai as genai

# Test the provided API key
api_key = "AIzaSyAJ6IpRR0eojz042OylU7mPSpy3SkHs7uI"

print("=" * 60)
print("🧪 Testing Provided API Key")
print("=" * 60)
print(f"API Key: {api_key[:20]}...")
print()

try:
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Test with the correct model
    print("📡 Connecting to Gemini API...")
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    # Simple test query
    print("💬 Sending test query...")
    response = model.generate_content("Say 'Hello! The API key is working perfectly!' in one sentence.")
    
    print()
    print("=" * 60)
    print("✅ SUCCESS!")
    print("=" * 60)
    print(f"Response: {response.text}")
    print()
    print("🎉 The API key is valid and working!")
    print()
    print("📝 Next steps:")
    print("   1. Add this to your .env file:")
    print(f"      GOOGLE_API_KEY={api_key}")
    print("   2. Restart the server: uvicorn app:app --reload")
    print("   3. Test summarization in the app")
    print("=" * 60)
    
except Exception as e:
    print()
    print("=" * 60)
    print("❌ ERROR")
    print("=" * 60)
    print(f"Error: {str(e)}")
    print()
    print("This could mean:")
    print("  - The API key is invalid or expired")
    print("  - The API key doesn't have access to Gemini models")
    print("  - Network connectivity issues")
    print("=" * 60)
