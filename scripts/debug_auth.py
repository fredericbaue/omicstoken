import os
import sys
import asyncio
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import auth
from auth import User
from sqlalchemy import select

API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8080')
TEST_EMAIL = os.environ.get('DEBUG_AUTH_EMAIL', 'test@example.com')
TEST_PASSWORD = os.environ.get('DEBUG_AUTH_PASSWORD', 'password123')

def try_login(email: str, password: str) -> bool:
    response = requests.post(
        f"{API_URL}/auth/jwt/login",
        data={'username': email, 'password': password},
        timeout=10,
    )
    if response.ok:
        print(f"? Login succeeded for {email}")
        return True
    print(f"??  Login failed ({response.status_code}): {response.text}")
    return False

async def reset_password(email: str, password: str):
    async with auth.async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        if not user:
            print(f"? User {email} not found in database.")
            return False
        user.hashed_password = auth.pwd_context.hash(password)
        await session.commit()
        print(f"?? Password for {email} has been reset using current hasher.")
        return True

def main():
    print(f"Attempting login as {TEST_EMAIL}...")
    if try_login(TEST_EMAIL, TEST_PASSWORD):
        return

    print("Resetting password via direct DB update...")
    asyncio.run(reset_password(TEST_EMAIL, TEST_PASSWORD))

    print("Re-attempting login...")
    try_login(TEST_EMAIL, TEST_PASSWORD)

if __name__ == '__main__':
    main()
