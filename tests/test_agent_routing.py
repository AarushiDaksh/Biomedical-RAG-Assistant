import asyncio
from unittest.mock import MagicMock
import rag.agent as agent
from rag.schemas import RAGAnswer


def _run(coro):
    return asyncio.run(coro)


async def _collect(gen):
    return [e async for e in gen]


def test_summary_intent_streams_summary(monkeypatch):
    async def fake_classify(q, h, llm):
        return "summary"
    monkeypatch.setattr(agent, "classify_intent", fake_classify)

    called = {}

    def fake_build(doc_id):
        called["doc"] = doc_id
        return "paper text", []
    monkeypatch.setattr(agent, "build_summary_request", fake_build)

    class FakeStreamLLM:
        async def astream(self, messages):
            for tok in ["Sum", "mary"]:
                yield MagicMock(content=tok)
    monkeypatch.setattr(agent, "get_stream_llm", lambda: FakeStreamLLM())

    events = _run(_collect(agent.answer_question_stream(
        "summarize it", history=[], document_ids=["dX"])))
    tokens = [e for e in events if e["type"] == "token"]
    done = [e for e in events if e["type"] == "done"][-1]
    assert called["doc"] == "dX"
    assert done["answer"] == "Summary"
    assert len(tokens) == 2


def test_qa_intent_with_no_chunks(monkeypatch):
    async def fake_classify(q, h, llm):
        return "qa"
    monkeypatch.setattr(agent, "classify_intent", fake_classify)
    monkeypatch.setattr(agent, "retrieve_chunks", lambda q, k=5, document_ids=None: [])

    events = _run(_collect(agent.answer_question_stream(
        "what is X", history=[], document_ids=["dX"])))
    done = [e for e in events if e["type"] == "done"][-1]
    assert done["confidence"] == "low"
    assert "could not find" in done["answer"].lower()


def test_compare_with_one_doc(monkeypatch):
    async def fake_classify(q, h, llm):
        return "compare"
    monkeypatch.setattr(agent, "classify_intent", fake_classify)
    # compare_documents itself guards <2 docs; ensure routing reaches it
    events = _run(_collect(agent.answer_question_stream(
        "compare", history=[], document_ids=["only-one"])))
    done = [e for e in events if e["type"] == "done"][-1]
    assert done["confidence"] == "low"
