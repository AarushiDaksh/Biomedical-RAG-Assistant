import asyncio
from unittest.mock import MagicMock
import rag.compare as compare
from rag.schemas import RAGAnswer, RetrievedChunk


class FakeLLM:
    async def ainvoke(self, messages):
        return MagicMock(content='{"answer": "they differ in X", "confidence": "medium", "citations": []}')


def _run(coro):
    return asyncio.run(coro)


def test_requires_two_documents():
    ans = _run(compare.compare_documents(["d1"], "compare them", FakeLLM()))
    assert ans.confidence == "low"
    assert "two" in ans.answer.lower() or "2" in ans.answer


def test_compares_two_documents(monkeypatch):
    async def fake_summary(doc_id, llm):
        return RAGAnswer(answer=f"summary of {doc_id}", confidence="high",
                         citations=[])
    monkeypatch.setattr(compare, "summarize_document", fake_summary)
    monkeypatch.setattr(compare, "get_document_chunks",
                        lambda d: [RetrievedChunk(chunk_id="c", document_id=d, title=d,
                                                  page_number=1, text="t")])
    ans = _run(compare.compare_documents(["d1", "d2"], "compare", FakeLLM()))
    assert ans.answer == "they differ in X"
