import json
import time
import uuid
from typing import AsyncIterator

from langchain_ollama import ChatOllama

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K
from .retriever import retrieve_chunks
from .guardrails import check_input
from .schemas import RAGAnswer, Citation, RetrievedChunk
from .intent import classify_intent
from .summarize import summarize_document
from .compare import compare_documents

# ── Singleton LLM ─────────────────────────────────────────────────────────────
_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            num_ctx=8192,
        )
    return _llm


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Biomedical Literature RAG Assistant.

Rules:
1. Answer ONLY from the retrieved context provided below. Never use outside knowledge.
2. Do not give personal clinical advice or diagnoses.
3. If the context is insufficient, set answer to "I could not find this in the uploaded papers."
4. Do not invent citations — only cite chunk_ids that appear in the context.
5. Always return valid JSON only — no markdown, no prose outside the JSON object.

Return exactly this JSON:
{
  "answer": "your answer here",
  "confidence": "low|medium|high",
  "citations": [
    {
      "title": "paper title",
      "document_id": "document id from context",
      "page_number": 1,
      "chunk_id": "chunk id from context"
    }
  ]
}"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"SOURCE {i}\n"
            f"Title: {c.title}\n"
            f"Document ID: {c.document_id}\n"
            f"Page: {c.page_number}\n"
            f"Chunk ID: {c.chunk_id}\n"
            f"Similarity: {c.score:.3f}\n"
            f"Text:\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _safe_json_parse(text: str) -> dict:
    """Extract the first complete JSON object from LLM output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return {"answer": text.strip(), "confidence": "low", "citations": []}


def _build_answer(parsed: dict) -> RAGAnswer:
    citations = [
        Citation(
            title=c.get("title", "Unknown"),
            document_id=c.get("document_id", "unknown"),
            page_number=int(c.get("page_number", 0)),
            chunk_id=c.get("chunk_id", "unknown"),
        )
        for c in parsed.get("citations", [])
    ]
    return RAGAnswer(
        answer=parsed.get("answer", "I could not generate an answer."),
        confidence=parsed.get("confidence", "low"),
        citations=citations,
    )


def enforce_citations(answer: RAGAnswer) -> RAGAnswer:
    """Defense-in-depth: a confident, non-empty answer must cite sources."""
    if not answer.citations and answer.answer and answer.confidence != "low":
        answer.refusal_reason = "missing_citations"
    return answer


def _history_block(history: list | None, limit: int = 6) -> str:
    turns = (history or [])[-limit:]
    if not turns:
        return ""
    lines = "\n".join(f"{t.get('role','?')}: {t.get('content','')}" for t in turns)
    return f"Conversation so far:\n{lines}\n\n"


def _cites(ans: RAGAnswer) -> list:
    return [{"title": c.title, "document_id": c.document_id,
             "page_number": c.page_number, "chunk_id": c.chunk_id} for c in ans.citations]


def _chunk_payload(chunks) -> list:
    return [{"title": c.title, "page_number": c.page_number, "chunk_id": c.chunk_id,
             "score": round(c.score, 3), "text": c.text[:800]} for c in chunks]


def _done(answer, confidence, citations, chunks, trace_id, start) -> dict:
    return {"type": "done", "answer": answer, "confidence": confidence,
            "citations": citations, "chunks": chunks, "trace_id": trace_id,
            "latency": round(time.time() - start, 2)}


# ── Streaming answer (SSE events) ─────────────────────────────────────────────
async def answer_question_stream(question: str, history: list | None = None,
                                 document_ids: list | None = None) -> AsyncIterator[dict]:
    """
    Yields SSE-compatible dicts:
      {"type": "status",  "message": "..."}
      {"type": "done",    "answer": "...", "confidence": "...",
                          "citations": [...], "chunks": [...],
                          "trace_id": "...", "latency": 0.0}
    """
    trace_id = str(uuid.uuid4())[:8]
    start = time.time()

    yield {"type": "status", "message": "Checking input…"}
    guardrail = check_input(question)
    if not guardrail.allowed:
        yield _done(guardrail.reason, "low", [], [], trace_id, start)
        return

    llm = get_llm()
    intent = await classify_intent(question, history or [], llm)

    if intent == "smalltalk":
        yield _done("Hello! Ask me about your uploaded biomedical papers.",
                    "high", [], [], trace_id, start)
        return

    if intent == "compare":
        yield {"type": "status", "message": "Comparing selected papers…"}
        ans = await compare_documents(document_ids or [], question, llm)
        yield _done(ans.answer, ans.confidence, _cites(ans), [], trace_id, start)
        return

    if intent == "summary":
        yield {"type": "status", "message": "Summarizing the paper…"}
        target = (document_ids or [None])[0]
        if not target:
            yield _done("Upload and select a paper first, then ask me to summarize it.",
                        "low", [], [], trace_id, start)
            return
        ans = await summarize_document(target, llm)
        yield _done(ans.answer, ans.confidence, _cites(ans), [], trace_id, start)
        return

    # intent == "qa"
    yield {"type": "status", "message": "Searching your papers…"}
    chunks = retrieve_chunks(question, k=TOP_K, document_ids=document_ids or None)
    if not chunks:
        yield _done("I could not find relevant passages in the selected papers.",
                    "low", [], [], trace_id, start)
        return

    yield {"type": "status", "message": f"Analysing {len(chunks)} passage(s)…"}
    context = _build_context(chunks)
    user_prompt = (
        f"{_history_block(history)}"
        f"Question:\n{question}\n\nRetrieved Context:\n{context}\n\n"
        "Answer using only the retrieved context. Return JSON only."
    )
    raw = (await llm.ainvoke([("system", SYSTEM_PROMPT), ("user", user_prompt)])).content
    answer_obj = enforce_citations(_build_answer(_safe_json_parse(raw)))
    yield _done(answer_obj.answer, answer_obj.confidence, _cites(answer_obj),
                _chunk_payload(chunks), trace_id, start)


# ── Non-streaming helper (used by Celery tasks / tests) ───────────────────────
async def answer_question(question: str) -> tuple[RAGAnswer, list, str, float]:
    """Convenience wrapper that collects the stream and returns the final result."""
    result = None
    async for event in answer_question_stream(question):
        if event["type"] == "done":
            result = event
            break

    if result is None:
        ans = RAGAnswer(answer="No answer generated.", confidence="low", citations=[])
        return ans, [], "error", 0.0

    ans = RAGAnswer(
        answer=result["answer"],
        confidence=result["confidence"],
        citations=[
            Citation(
                title=c["title"],
                document_id=c["document_id"],
                page_number=c["page_number"],
                chunk_id=c["chunk_id"],
            )
            for c in result["citations"]
        ],
    )
    return ans, result["chunks"], result["trace_id"], result["latency"]
