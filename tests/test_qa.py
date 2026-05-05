"""Tests for the end-to-end Q&A orchestrator."""

import json

import pytest

from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedder import embed_query
from app.services.qa import answer_question


async def _seed_chunks(db):
    doc = Document(
        filename="attn.pdf", title="Attn",
        file_size_bytes=100, total_chunks=2, text_length=200,
    )
    db.add(doc)
    await db.flush()
    for i, t in enumerate(
        [
            "Attention mechanisms compute weights over tokens using query and key dot products.",
            "Layer normalization is applied to stabilize training.",
        ]
    ):
        db.add(
            Chunk(
                document_id=doc.id, chunk_index=i, text=t,
                embedding=embed_query(t), token_count=10,
                page_start=1, page_end=1,
            )
        )
    await db.flush()
    return doc


@pytest.mark.asyncio
async def test_answer_question_full_flow(db_session, fake_redis, patch_anthropic):
    await _seed_chunks(db_session)
    json_payload = json.dumps(
        {
            "answer": "Attention weighs tokens using dot products [1].",
            "used_passages": [1],
        }
    )
    patch_anthropic(json_payload, input_tokens=120, output_tokens=40)

    q = await answer_question(db_session, fake_redis, "How does attention work?")
    assert q.answer.startswith("Attention weighs tokens")
    assert q.cache_hit is False
    assert q.input_tokens == 120
    assert q.output_tokens == 40
    assert q.cost > 0
    assert q.sources and len(q.sources) >= 1


@pytest.mark.asyncio
async def test_answer_question_cache_hit(db_session, fake_redis, patch_anthropic):
    await _seed_chunks(db_session)
    patch_anthropic(json.dumps({"answer": "X.", "used_passages": [1]}))
    q1 = await answer_question(db_session, fake_redis, "What is attention?")
    q2 = await answer_question(db_session, fake_redis, "What is attention?")
    assert q2.cache_hit is True
    assert q2.input_tokens == 0
    assert q2.cost == 0.0
