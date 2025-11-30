import db


def test_get_embedding_reads_peptide_embeddings_only(monkeypatch):
    calls = {"legacy": 0, "canonical": 0}

    def legacy_fetch(*args, **kwargs):
        calls["legacy"] += 1
        return []

    def canonical_fetch(*args, **kwargs):
        calls["canonical"] += 1
        return []

    cur_mock = type(
        "Cur",
        (),
        {
            "execute": lambda self, *args, **kwargs: None,
            "fetchone": lambda self: None,
        },
    )()

    con_mock = type("Con", (), {"cursor": lambda self: cur_mock})()

    # Ensure peptide_embeddings path is used; we don't call legacy helpers
    db.get_embedding(con_mock, "run", "feat")
    assert calls["legacy"] == 0
