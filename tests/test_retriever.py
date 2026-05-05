"""End-to-end retrieval tests against the SQLite fallback path."""

import pytest

from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedder import embed_query
from app.services.retriever import retrieve


@pytest.mark.asyncio
async def test_retrieve_returns_top_k(db_session):
    # Seed two documents with three chunks each
    doc = Document(
        filename="paper.pdf",
        title="Paper",
        file_size_bytes=100,
        total_chunks=3,
        text_length=300,
    )
    db_session.add(doc)
    await db_session.flush()

    texts = [
        "Attention mechanisms allow models to weigh tokens.",
        "Layer normalization stabilizes training.",
        "Convolutional filters detect local patterns.",
    ]
    for i, t in enumerate(texts):
        emb = embed_query(t)
        db_session.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                text=t,
                embedding=emb,
                token_count=10,
                page_start=1,
                page_end=1,
            )
        )
    await db_session.flush()

    out = await retrieve(
        db_session,
        question="how does attention work",
        top_k_vector=3,
        top_k_rerank=2,
    )
    assert len(out) == 2
    assert out[0].relevance_score >= out[1].relevance_score
    assert "attention" in out[0].text.lower() or "attention" in out[1].text.lower()


@pytest.mark.asyncio
async def test_retrieve_empty_db(db_session):
    out = await retrieve(db_session, question="x", top_k_rerank=5)
    assert out == []
