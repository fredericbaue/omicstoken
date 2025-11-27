from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {api_key}")
print(f"Key length: {len(api_key) if api_key else 0}")
