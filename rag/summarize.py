import json
from typing import List, Tuple

from .config import SUMMARY_CHAR_BUDGET
from .retriever import get_document_chunks
from .schemas import RAGAnswer, Citation, RetrievedChunk

SUMMARY_SYSTEM = """You are a biomedical literature assistant.
Summarize the provided paper text faithfully. Do not use outside knowledge.
Cover: aim, methods, key findings, and limitations.
Return ONLY JSON: {"answer": "...", "confidence": "low|medium|high", "citations": []}"""

# Plain-prose variant used when the summary is streamed token-by-token to the UI.
SUMMARY_STREAM_SYSTEM = """You are a biomedical literature assistant.
Summarize the provided paper text faithfully using only that text — no outside knowledge.
Cover: aim, methods, key findings, and limitations. Write clear prose for the user."""


def _parse(raw: str) -> dict:
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(raw[s:e])
    except (json.JSONDecodeError, ValueError):
        pass
    return {"answer": raw.strip(), "confidence": "low", "citations": []}


def _citations(chunks: List[RetrievedChunk]) -> List[Citation]:
    seen, cites = set(), []
    for c in chunks:
        key = (c.document_id, c.page_number)
        if key in seen:
            continue
        seen.add(key)
        cites.append(Citation(title=c.title, document_id=c.document_id,
                              page_number=c.page_number, chunk_id=c.chunk_id))
    return cites


def _select_within_budget(chunks: List[RetrievedChunk], budget: int) -> str:
    """Text for a single-call summary, capped to `budget` characters.

    Short papers are passed whole. Long papers are sampled head + tail so the
    summary still sees the abstract/intro/methods (start) and the
    results/discussion/limitations (end), which is where summary-relevant
    content lives — rather than truncating to just the first few pages.
    """
    full = "\n\n".join(c.text for c in chunks)
    if len(full) <= budget:
        return full

    half = budget // 2
    head, hsize = [], 0
    for c in chunks:
        if hsize + len(c.text) > half:
            break
        head.append(c.text)
        hsize += len(c.text)

    tail, tsize = [], 0
    for c in reversed(chunks):
        if tsize + len(c.text) > (budget - hsize):
            break
        tail.append(c.text)
        tsize += len(c.text)
    tail.reverse()

    return "\n\n".join(head) + "\n\n[...]\n\n" + "\n\n".join(tail)


async def _summarize_text(system: str, text: str, llm) -> str:
    raw = (await llm.ainvoke([("system", system), ("user", text)])).content
    return _parse(raw).get("answer", "").strip()


def build_summary_request(document_id: str) -> Tuple[str, List[Citation]]:
    """Return (capped_text, citations) for a single-call summary, or ("", [])."""
    chunks = get_document_chunks(document_id)
    if not chunks:
        return "", []
    return _select_within_budget(chunks, SUMMARY_CHAR_BUDGET), _citations(chunks)


async def summarize_document(document_id: str, llm) -> RAGAnswer:
    """Single-call summary over a character-capped slice of the paper.

    Used directly for the summary intent (non-streamed callers) and by the
    compare flow. Streaming summaries are produced in rag.agent.
    """
    text, cites = build_summary_request(document_id)
    if not text:
        return RAGAnswer(answer="I have no indexed content for that paper.",
                         confidence="low", citations=[])
    try:
        answer = await _summarize_text(SUMMARY_SYSTEM, text, llm)
    except Exception:
        answer = ""
    return RAGAnswer(answer=answer or "Could not summarize this paper.",
                     confidence="high" if answer else "low", citations=cites)
