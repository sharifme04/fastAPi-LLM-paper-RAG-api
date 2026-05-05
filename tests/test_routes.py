"""Route-level tests."""

import json

import pytest

from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedder import embed_query
from tests.conftest import test_async_session


@pytest.mark.asyncio
async def test_documents_list_empty(client):
    r = await client.get("/documents")
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_query_endpoint_404_for_unknown(client):
    r = await client.get("/query/9999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_query_endpoint_full_flow(client, patch_anthropic):
    # Seed
    async with test_async_session() as s:
        doc = Document(
            filename="p.pdf", title="P",
            file_size_bytes=100, total_chunks=1, text_length=50,
        )
        s.add(doc)
        await s.flush()
        s.add(
            Chunk(
                document_id=doc.id,
                chunk_index=0,
                text="Attention computes weights from query and key dot products.",
                embedding=embed_query("Attention computes weights from query and key dot products."),
                token_count=10,
                page_start=1,
                page_end=1,
            )
        )
        await s.commit()

    patch_anthropic(json.dumps({"answer": "It uses query·key dot products [1].", "used_passages": [1]}))
    r = await client.post("/query", json={"question": "How does attention work?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "query·key" in body["answer"]
    assert len(body["sources"]) >= 1


@pytest.mark.asyncio
async def test_analytics_summary_empty(client):
    r = await client.get("/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_documents"] == 0
    assert body["total_queries"] == 0


@pytest.mark.asyncio
async def test_eval_report_when_empty(client):
    r = await client.get("/evals/report")
    assert r.status_code == 200
    body = r.json()
    assert body["num_queries_evaluated"] == 0
    assert body["passes_threshold"] is False
