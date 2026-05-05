"""Chunker tests."""

from app.services.chunker import _split_sentences, chunk_pages


def test_split_sentences_basic():
    out = _split_sentences("Hello world. This is great! Is it? Yes.")
    assert len(out) == 4
    assert out[0] == "Hello world."


def test_chunk_pages_produces_chunks_with_overlap():
    long_text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_pages([(1, long_text)], chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(c.token_count > 0 for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_pages_records_page_range():
    pages = [(1, "Alpha. " * 30), (2, "Beta. " * 30)]
    chunks = chunk_pages(pages, chunk_size_tokens=200, overlap_tokens=20)
    pages_seen = {c.page_start for c in chunks} | {c.page_end for c in chunks}
    assert 1 in pages_seen


def test_chunk_pages_empty():
    assert chunk_pages([]) == []
