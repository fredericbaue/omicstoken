"""Quick test of Gemini API key"""
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")

if api_key:
    print(f"Key prefix: {api_key[:15]}...")
    genai.configure(api_key=api_key)
    
    try:
        # Test with a simple embedding
        result = genai.embed_content(
            model="models/text-embedding-004",
            content="ACDEFGH",
            task_type="semantic_similarity"
        )
        
        if 'embedding' in result:
            print(f"✅ API key works! Embedding dimension: {len(result['embedding'])}")
        else:
            print("❌ No embedding returned")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ No API key found in .env file")
