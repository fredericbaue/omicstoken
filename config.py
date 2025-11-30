import os
import logging

# Centralized configuration
# DATA_DIR: root directory for SQLite and artifacts; defaults to "data".
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Embedder selection and model configuration
# EMBEDDER_NAME: which embedder to use (default: "esm2").
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "esm2")
# ESM_MODEL_NAME: HF model name for the ESM-2 embedder.
EMBEDDING_MODEL_NAME = os.getenv("ESM_MODEL_NAME", "facebook/esm2_t6_8M_UR50D")

# Logging configuration
# LOG_LEVEL: default INFO; override via env LOG_LEVEL=DEBUG/ERROR.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s - %(message)s")

# Demo / sandbox settings
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
DEMO_USER_EMAIL = os.getenv("DEMO_USER_EMAIL", "demo@omicstoken.local")

if not logging.getLogger().handlers:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
