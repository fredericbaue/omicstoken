import requests

# Test login endpoint
url = "http://127.0.0.1:8080/auth/jwt/login"
data = {
    "username": "test@example.com",
    "password": "password123"
}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
