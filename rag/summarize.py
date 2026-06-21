import json
from typing import List

from .retriever import get_document_chunks
from .schemas import RAGAnswer, Citation, RetrievedChunk

# ~6k tokens of context at ~4 chars/token leaves headroom under num_ctx=8192.
STUFF_CHAR_BUDGET = 24_000
GROUP_CHAR_BUDGET = 8_000

SUMMARY_SYSTEM = """You are a biomedical literature assistant.
Summarize the provided paper text faithfully. Do not use outside knowledge.
Cover: aim, methods, key findings, and limitations.
Return ONLY JSON: {"answer": "...", "confidence": "low|medium|high", "citations": []}"""

REDUCE_SYSTEM = """Combine these partial summaries of ONE paper into a single coherent
summary covering aim, methods, key findings, and limitations. No outside knowledge.
Return ONLY JSON: {"answer": "...", "confidence": "low|medium|high", "citations": []}"""


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


def _groups(chunks: List[RetrievedChunk], budget: int) -> List[List[RetrievedChunk]]:
    groups, cur, size = [], [], 0
    for c in chunks:
        if size + len(c.text) > budget and cur:
            groups.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c.text)
    if cur:
        groups.append(cur)
    return groups


async def _summarize_text(system: str, text: str, llm) -> str:
    raw = (await llm.ainvoke([("system", system), ("user", text)])).content
    return _parse(raw).get("answer", "").strip()


async def summarize_document(document_id: str, llm) -> RAGAnswer:
    chunks = get_document_chunks(document_id)
    if not chunks:
        return RAGAnswer(answer="I have no indexed content for that paper.",
                         confidence="low", citations=[])

    full = "\n\n".join(c.text for c in chunks)
    cites = _citations(chunks)

    if len(full) <= STUFF_CHAR_BUDGET:
        answer = await _summarize_text(SUMMARY_SYSTEM, full, llm)
        return RAGAnswer(answer=answer or "Could not summarize.",
                         confidence="high" if answer else "low", citations=cites)

    # Map-reduce: summarize groups, then combine. Tolerate partial failures.
    groups = _groups(chunks, GROUP_CHAR_BUDGET)
    partials = []
    for group in groups:
        try:
            text = "\n\n".join(c.text for c in group)
            partials.append(await _summarize_text(SUMMARY_SYSTEM, text, llm))
        except Exception:
            continue
    if not partials:
        return RAGAnswer(answer="Summarization failed for this paper.",
                         confidence="low", citations=cites)
    combined = await _summarize_text(REDUCE_SYSTEM, "\n\n".join(partials), llm)
    confidence = "medium" if len(partials) >= (len(groups) + 1) // 2 else "low"
    return RAGAnswer(answer=combined or "\n\n".join(partials),
                     confidence=confidence, citations=cites)
