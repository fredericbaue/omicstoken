import importlib
import os

import numpy as np
import pytest
import torch

import config
import embeddings
from embeddings import EMBEDDING_DIM, BaseEmbedder, Esm2Embedder


def _reset_cache():
    embeddings._MODEL = None
    embeddings._TOKENIZER = None
    embeddings._DEFAULT_EMBEDDER = embeddings.create_embedder("esm2")


class DummyTokenizer:
    calls = 0

    @classmethod
    def from_pretrained(cls, name):
        cls.calls += 1
        return cls()

    def __call__(self, seq, return_tensors=None, add_special_tokens=True):
        return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}


class DummyModel:
    calls = 0

    @classmethod
    def from_pretrained(cls, name):
        cls.calls += 1
        return cls()

    def eval(self):
        return self

    def __call__(self, **kwargs):
        class Output:
            def __init__(self):
                self.last_hidden_state = torch.ones((1, 5, EMBEDDING_DIM))

        return Output()


def test_valid_peptide_returns_vector(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(embeddings, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(embeddings, "AutoModel", DummyModel)

    vec = embeddings.peptide_to_vector("PEPTIDE")

    assert vec.shape == (EMBEDDING_DIM,)
    assert np.any(vec != 0)


def test_invalid_peptide_returns_zero(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(embeddings, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(embeddings, "AutoModel", DummyModel)

    vec = embeddings.peptide_to_vector("   ")

    assert vec.shape == (EMBEDDING_DIM,)
    assert np.allclose(vec, 0.0)


def test_model_is_cached(monkeypatch):
    _reset_cache()
    DummyTokenizer.calls = 0
    DummyModel.calls = 0
    monkeypatch.setattr(embeddings, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(embeddings, "AutoModel", DummyModel)

    embeddings.peptide_to_vector("PEPTIDE")
    embeddings.peptide_to_vector("PEPTIDE")

    assert DummyTokenizer.calls == 1
    assert DummyModel.calls == 1


def test_embedder_interface(monkeypatch):
    """Ensure Esm2Embedder follows BaseEmbedder contract."""
    _reset_cache()
    monkeypatch.setattr(embeddings, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(embeddings, "AutoModel", DummyModel)

    embedder: BaseEmbedder = Esm2Embedder()
    vec = embedder.embed("PEPTIDE")
    assert vec.shape == (EMBEDDING_DIM,)
    assert np.any(vec != 0)

    vec_empty = embedder.embed("")
    assert vec_empty.shape == (EMBEDDING_DIM,)
    assert np.allclose(vec_empty, 0.0)


def test_default_factory_uses_esm2(monkeypatch):
    # Simulate env requesting esm2 (default)
    monkeypatch.setenv("EMBEDDER_NAME", "esm2")
    monkeypatch.setenv("ESM_MODEL_NAME", "facebook/esm2_t6_8M_UR50D")
    importlib.reload(config)
    importlib.reload(embeddings)

    monkeypatch.setattr(embeddings, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(embeddings, "AutoModel", DummyModel)

    vec = embeddings.peptide_to_vector("PEPTIDE")
    assert isinstance(embeddings._DEFAULT_EMBEDDER, embeddings.Esm2Embedder)
    assert vec.shape == (EMBEDDING_DIM,)
    assert np.any(vec != 0)


def test_unknown_embedder_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDER_NAME", "unknown")
    with pytest.raises(ValueError):
        importlib.reload(config)
        importlib.reload(embeddings)
