import asyncio
from unittest.mock import MagicMock
import rag.summarize as summarize
from rag.schemas import RetrievedChunk


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.last_user_text = ""

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_user_text = messages[-1][1]
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


def test_large_doc_single_capped_call(monkeypatch):
    # 40 chunks * 2000 chars = 80k chars >> SUMMARY_CHAR_BUDGET. The whole paper
    # must still be summarized in ONE call, over a character-capped slice.
    from rag.config import SUMMARY_CHAR_BUDGET
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: _chunks(40, 2000))
    llm = FakeLLM()
    ans = _run(summarize.summarize_document("d1", llm))
    assert llm.calls == 1
    assert ans.answer == "a summary"
    # Capped near budget (small margin for chunk-join separators), not the full 80k.
    assert len(llm.last_user_text) <= SUMMARY_CHAR_BUDGET + 100


def test_large_doc_samples_head_and_tail(monkeypatch):
    # Distinct head/tail text so we can prove both ends reach the summary.
    chunks = _chunks(40, 2000)
    chunks[0].text = "HEAD_MARKER " + chunks[0].text
    chunks[-1].text = "TAIL_MARKER " + chunks[-1].text
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: chunks)
    llm = FakeLLM()
    _run(summarize.summarize_document("d1", llm))
    assert "HEAD_MARKER" in llm.last_user_text
    assert "TAIL_MARKER" in llm.last_user_text


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


def test_summarize_handles_llm_failure(monkeypatch):
    # A failing LLM call must degrade gracefully, not crash the request.
    monkeypatch.setattr(summarize, "get_document_chunks", lambda d: _chunks(3, 500))
    llm = FlakyLLM(fail_first=1)
    ans = _run(summarize.summarize_document("d1", llm))
    assert ans.answer  # non-empty fallback message
    assert ans.confidence == "low"
