import google.generativeai as genai
import os
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not configured; skipping Gemini API test.")

    print(f"�o. API Key found (starts with: {api_key[:10]}...)")

    try:
        genai.configure(api_key=api_key)

        # Test with the updated model
        model = genai.GenerativeModel("gemini-2.0-flash-exp")

        response = model.generate_content("Say 'Hello, the API is working!' in one sentence.")

        print(f"�o. Model: models/gemini-1.5-flash")
        print(f"�o. Response: {response.text}")
        print("\n�YZ% Gemini API is configured correctly!")
        assert True

    except Exception as e:
        pytest.fail(f"API Error: {str(e)}")


if __name__ == "__main__":
    test_api()
