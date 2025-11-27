import google.generativeai as genai

# Test the provided API key with different model names
api_key = "AIzaSyAJ6IpRR0eojz042OylU7mPSpy3SkHs7uI"

print("=" * 60)
print("🧪 Testing Different Model Names")
print("=" * 60)

genai.configure(api_key=api_key)

# Try different model name formats
model_names = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-pro',
    'gemini-2.0-flash-exp',
    'gemini-flash-latest',
]

for model_name in model_names:
    try:
        print(f"\n🔍 Trying: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello!' in one word.")
        
        print(f"   ✅ SUCCESS with {model_name}")
        print(f"   Response: {response.text}")
        print()
        print("=" * 60)
        print(f"🎉 WORKING MODEL FOUND: {model_name}")
        print("=" * 60)
        print(f"\nAdd to your .env file:")
        print(f"GOOGLE_API_KEY={api_key}")
        print()
        print(f"Working model: {model_name}")
        print("=" * 60)
        break
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)[:80]}...")
        continue
else:
    print("\n❌ None of the models worked. Listing available models...")
    try:
        print("\nAvailable models:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
