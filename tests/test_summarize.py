import asyncio
from unittest.mock import MagicMock
import rag.summarize as summarize
from rag.schemas import RetrievedChunk


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return MagicMock(content='{"answer": "a summary", "confidence": "high", "citations": []}')


def _run(coro):
    return asyncio.run(coro)


def _chunks(n, chars):
    return [RetrievedChunk(chunk_id=f"c{i}", document_id="d1", title="Paper T",
                           page_number=i + 1, text="x" * chars) for i in range(n)]


def test_stuff_path_single_llm_call(monkeypatch):
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: _chunks(3, 500))
    llm = FakeLLM()
    ans = _run(summarize.summarize_document("d1", llm))
    assert llm.calls == 1
    assert ans.answer == "a summary"


def test_map_reduce_path_multiple_calls(monkeypatch):
    # 40 chunks * 2000 chars = 80k chars >> STUFF_CHAR_BUDGET -> map-reduce
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: _chunks(40, 2000))
    llm = FakeLLM()
    ans = _run(summarize.summarize_document("d1", llm))
    assert llm.calls > 1


def test_empty_document(monkeypatch):
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: [])
    llm = FakeLLM()
    ans = _run(summarize.summarize_document("d1", llm))
    assert ans.confidence == "low"
    assert llm.calls == 0


class FlakyLLM:
    def __init__(self, fail_first):
        self.calls = 0
        self.fail_first = fail_first

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("boom")
        return MagicMock(content='{"answer": "partial summary", "confidence": "high", "citations": []}')


def test_map_reduce_tolerates_partial_failures(monkeypatch):
    # 40 chunks * 2000 chars = 80k chars >> STUFF_CHAR_BUDGET -> map-reduce
    # 80k / 8k GROUP_CHAR_BUDGET = 10 groups; fail_first=3 -> 7 groups succeed + reduce call
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: _chunks(40, 2000))
    llm = FlakyLLM(fail_first=3)
    ans = _run(summarize.summarize_document("d1", llm))
    assert ans.answer  # non-empty; did not crash
