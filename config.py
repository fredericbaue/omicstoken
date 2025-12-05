import logging
import os

# Configure root logger early so downstream modules can rely on it.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s - %(message)s")
if not logging.getLogger().handlers:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)

# Centralized configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Embedder selection and model configuration
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "esm2")
EMBEDDING_MODEL_NAME = os.getenv("ESM_MODEL_NAME", "facebook/esm2_t6_8M_UR50D")

# --- Database backend selection (prep for Postgres) ---
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").lower()
DATABASE_URL = os.getenv("DATABASE_URL", "")

# --- Celery / job engine configuration ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

# Gemini / Google Generative AI configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
HAS_GEMINI_KEY = bool(GEMINI_API_KEY)
if not HAS_GEMINI_KEY:
    LOGGER.warning(
        "Gemini API key not configured; AI summaries/explanations will return a friendly error. "
        "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment/.env."
    )

# Demo / sandbox settings
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DEMO_USER_EMAIL = os.getenv("DEMO_USER_EMAIL", "demo@omicstoken.local")
