from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"Testing API Key: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    response = model.generate_content("Say hello")
    print(f"SUCCESS: API Key works! Response: {response.text[:50]}")
except Exception as e:
    print(f"FAILED: API Key error: {e}")
