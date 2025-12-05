import os
import uuid
import inspect
import secrets
import hashlib
from datetime import datetime
from typing import Optional
from fastapi import Depends, Request, HTTPException, status
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin, schemas, models
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy import Column, Integer, String, DateTime, select
from passlib.context import CryptContext

# --- Configuration ---
SECRET = "SECRET_KEY_FOR_MVP_ONLY_CHANGE_IN_PROD"
DATABASE_URL = "sqlite+aiosqlite:///./data/users.db"

# --- Database Setup ---
Base: DeclarativeMeta = declarative_base()

class User(SQLAlchemyBaseUserTableUUID, Base):
    credits = Column(Integer, default=0)


class APIToken(Base):
    __tablename__ = "api_tokens"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True, nullable=False)
    token_hash = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_async_session():
    async with async_session_maker() as session:
        yield session

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)

# --- Password Helper (The Fix) ---
# Explicitly support legacy Argon2 hashes while hashing new passwords with bcrypt.
# If argon2 is unavailable (e.g., missing optional dependency), fall back to bcrypt-only.
try:
    pwd_context = CryptContext(schemes=["bcrypt", "argon2"], deprecated="auto")
except Exception:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- User Manager ---
class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def create(
        self, user_create: schemas.UC, safe: bool = False, request: Optional[Request] = None
    ) -> models.UP:
        # This override handles the custom 'credits' field.
        # It separates standard user creation data from our custom field.
        user_dict = user_create.dict()
        credits = user_dict.pop("credits", 0)
        
        # Use the parent class to create the user with standard fields.
        created_user = await super().create(user_create, safe, request)
        
        created_user.credits = credits
        return created_user

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

async def get_user_manager(user_db=Depends(get_user_db)):
    user_manager = UserManager(user_db)
    user_manager.password_helper = pwd_context
    yield user_manager

# --- Schemas ---
class UserRead(schemas.BaseUser[uuid.UUID]):
    credits: int = 0

class UserCreate(schemas.BaseUserCreate):
    credits: int = 0

class UserUpdate(schemas.BaseUserUpdate):
    pass

# --- Authentication Backend ---
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)


async def _get_user_from_api_token(raw_token: str) -> Optional[User]:
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    async with async_session_maker() as session:
        result = await session.execute(select(APIToken).where(APIToken.token_hash == token_hash))
        token_row = result.scalars().first()
        if not token_row:
            return None
        return await session.get(User, uuid.UUID(token_row.user_id))

def get_token_from_request(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization")
    if not header:
        return None
    if header.startswith("Bearer "):
        return header.split(" ", 1)[1]
    return None


async def current_active_user_or_token(request: Request) -> User:
    """
    Resolve the current user either via API token (Authorization: Bearer <token>)
    or via the standard FastAPI-Users JWT backend.

    Order:
      1. If a Bearer token is present, try API token lookup first.
      2. If that fails or no token, fall back to FastAPI-Users current_user.
    """
    token = get_token_from_request(request)

    if token:
        user_or_awaitable = _get_user_from_api_token(token)
        # Support both sync and async implementations of _get_user_from_api_token
        if inspect.isawaitable(user_or_awaitable):
            user = await user_or_awaitable
        else:
            user = user_or_awaitable

        if user:
            return user

    # No valid API token – fall back to standard JWT user resolution
    return await fastapi_users.current_user(active=True)(request)

# --- Init DB Utility ---
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# --- API Token helpers ---

async def create_api_token_for_user(user_id: str, label: Optional[str] = None):
    """
    Creates an API token for the given user_id, stores only the hash, and returns the plain token once.
    """
    plain = secrets.token_hex(32)
    token_hash = hashlib.sha256(plain.encode("utf-8")).hexdigest()
    async with async_session_maker() as session:
        token_row = APIToken(user_id=str(user_id), token_hash=token_hash, label=label)
        session.add(token_row)
        await session.commit()
        await session.refresh(token_row)
    return {"token": plain, "token_id": token_row.id, "label": label, "created_at": token_row.created_at}


async def delete_api_token_for_user(user_id: str, token_id: int) -> bool:
    async with async_session_maker() as session:
        result = await session.execute(
            select(APIToken).where(APIToken.id == token_id, APIToken.user_id == str(user_id))
        )
        token_row = result.scalars().first()
        if not token_row:
            return False
        await session.delete(token_row)
        await session.commit()
        return True
