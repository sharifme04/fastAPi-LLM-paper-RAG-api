"""Embedder + reranker fakes integration tests."""

import numpy as np

from app.services.embedder import embed_query, embed_texts, rerank


def test_embed_texts_returns_lists():
    out = embed_texts(["hello", "world"])
    assert len(out) == 2
    assert all(isinstance(v, list) for v in out)
    assert len(out[0]) == 384


def test_embed_query_deterministic():
    a = embed_query("the quick brown fox")
    b = embed_query("the quick brown fox")
    assert a == b


def test_rerank_orders_relevant_higher():
    scores = rerank(
        "what is attention",
        ["attention is all you need", "we like cookies"],
    )
    assert scores[0] > scores[1]


def test_rerank_empty():
    assert rerank("anything", []) == []
