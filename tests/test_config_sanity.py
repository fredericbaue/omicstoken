import importlib
import os
import tempfile

import pytest


def test_config_defaults_and_embedder_factory(monkeypatch):
    # Use a temp directory for DATA_DIR to ensure it can be created.
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("DATA_DIR", tmpdir)
        monkeypatch.delenv("EMBEDDER_NAME", raising=False)
        monkeypatch.delenv("ESM_MODEL_NAME", raising=False)

        import config
        import embeddings

        importlib.reload(config)
        importlib.reload(embeddings)

        assert config.DATA_DIR == tmpdir
        assert config.EMBEDDER_NAME == "esm2"
        assert config.EMBEDDING_MODEL_NAME == "facebook/esm2_t6_8M_UR50D"

        # Factory should default to Esm2Embedder
        embedder = embeddings.create_embedder()
        assert isinstance(embedder, embeddings.Esm2Embedder)

        # DATA_DIR path exists
        assert os.path.exists(config.DATA_DIR)
