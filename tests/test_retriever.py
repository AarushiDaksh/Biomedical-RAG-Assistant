from unittest.mock import MagicMock
import rag.retriever as retriever
from rag.config import MIN_SIMILARITY


def _doc(chunk_id, doc_id, page, idx, text="some text"):
    md = {"chunk_id": chunk_id, "document_id": doc_id, "title": "T",
          "page_number": page, "chunk_index": idx, "source_url": ""}
    d = MagicMock()
    d.metadata = md
    d.page_content = text
    return d


def test_retrieve_filters_by_document_ids(monkeypatch):
    db = MagicMock()
    db.similarity_search_with_relevance_scores.return_value = [(_doc("c1", "d1", 1, 0), 0.9)]
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    retriever.retrieve_chunks("q", k=5, document_ids=["d1", "d2"])

    _, kwargs = db.similarity_search_with_relevance_scores.call_args
    assert kwargs["filter"] == {"document_id": {"$in": ["d1", "d2"]}}


def test_retrieve_no_filter_when_no_selection(monkeypatch):
    db = MagicMock()
    db.similarity_search_with_relevance_scores.return_value = []
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    retriever.retrieve_chunks("q", k=5, document_ids=None)

    _, kwargs = db.similarity_search_with_relevance_scores.call_args
    assert kwargs["filter"] is None


def test_get_document_chunks_orders_by_page_then_index(monkeypatch):
    db = MagicMock()
    db.get.return_value = {
        "ids": ["c2", "c1", "c3"],
        "documents": ["page2", "page1-b", "page1-a"],
        "metadatas": [
            {"chunk_id": "c2", "document_id": "d1", "title": "T", "page_number": 2, "chunk_index": 0, "source_url": ""},
            {"chunk_id": "c1", "document_id": "d1", "title": "T", "page_number": 1, "chunk_index": 1, "source_url": ""},
            {"chunk_id": "c3", "document_id": "d1", "title": "T", "page_number": 1, "chunk_index": 0, "source_url": ""},
        ],
    }
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    chunks = retriever.get_document_chunks("d1")

    assert [c.text for c in chunks] == ["page1-a", "page1-b", "page2"]
    db.get.assert_called_once_with(where={"document_id": "d1"})


def test_retrieve_returns_mapped_chunk(monkeypatch):
    """One doc above threshold: retrieve_chunks returns a RetrievedChunk with
    the expected chunk_id and score (exercises _to_chunk mapping)."""
    score = MIN_SIMILARITY + 0.5
    db = MagicMock()
    db.similarity_search_with_relevance_scores.return_value = [
        (_doc("cX", "dX", 3, 1, text="hello"), score)
    ]
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    chunks = retriever.retrieve_chunks("q", k=5)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "cX"
    assert chunks[0].score == score


def test_retrieve_excludes_below_min_similarity(monkeypatch):
    """Two docs returned by vector store — one above, one below MIN_SIMILARITY.
    Only the above-threshold chunk must appear in the result."""
    above_score = MIN_SIMILARITY + 0.5
    below_score = max(0.0, MIN_SIMILARITY - 0.1)
    db = MagicMock()
    db.similarity_search_with_relevance_scores.return_value = [
        (_doc("c-above", "d1", 1, 0), above_score),
        (_doc("c-below", "d1", 1, 1), below_score),
    ]
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    chunks = retriever.retrieve_chunks("q", k=5)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c-above"
