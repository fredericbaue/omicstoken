import importlib
import logging
import os
import tempfile

def test_logging_config_imports_and_emits(monkeypatch, caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("DATA_DIR", tmpdir)
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        monkeypatch.delenv("EMBEDDER_NAME", raising=False)
        monkeypatch.delenv("ESM_MODEL_NAME", raising=False)

        import config
        import embeddings

        importlib.reload(config)
        importlib.reload(embeddings)

        assert os.path.exists(config.DATA_DIR)

        with caplog.at_level(logging.INFO):
            embeddings.peptide_to_vector("   ")
        # Should have logged at least one warning about empty/blank sequence
        assert any("empty/blank sequence" in msg for msg in caplog.text.splitlines())
