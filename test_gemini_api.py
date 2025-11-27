import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables!")
        print("Please ensure your .env file contains: GOOGLE_API_KEY=your_key_here")
        return False
    
    print(f"✅ API Key found (starts with: {api_key[:10]}...)")
    
    try:
        genai.configure(api_key=api_key)
        
        # Test with the updated model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content("Say 'Hello, the API is working!' in one sentence.")
        
        print(f"✅ Model: models/gemini-1.5-flash")
        print(f"✅ Response: {response.text}")
        print("\n🎉 Gemini API is configured correctly!")
        return True
        
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_api()
