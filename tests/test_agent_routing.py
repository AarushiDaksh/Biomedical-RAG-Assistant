import asyncio
import rag.agent as agent
from rag.schemas import RAGAnswer


def _run(coro):
    return asyncio.run(coro)


async def _collect(gen):
    return [e async for e in gen]


def test_summary_intent_routes_to_summarize(monkeypatch):
    async def fake_classify(q, h, llm):
        return "summary"
    monkeypatch.setattr(agent, "classify_intent", fake_classify)

    called = {}

    async def fake_summarize(doc_id, llm):
        called["doc"] = doc_id
        return RAGAnswer(answer="S", confidence="high", citations=[])
    monkeypatch.setattr(agent, "summarize_document", fake_summarize)

    events = _run(_collect(agent.answer_question_stream(
        "summarize it", history=[], document_ids=["dX"])))
    done = [e for e in events if e["type"] == "done"][-1]
    assert done["answer"] == "S"
    assert called["doc"] == "dX"


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
