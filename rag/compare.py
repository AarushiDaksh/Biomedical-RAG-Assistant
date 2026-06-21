from typing import List

from .retriever import get_document_chunks
from .summarize import summarize_document, _parse, _citations
from .schemas import RAGAnswer

COMPARE_SYSTEM = """You compare biomedical papers using only the provided summaries.
Contrast them along: aim, methods, key findings, and limitations, focused on the
user's question. No outside knowledge.
Return ONLY JSON: {"answer": "...", "confidence": "low|medium|high", "citations": []}"""


async def compare_documents(document_ids: List[str], question: str, llm) -> RAGAnswer:
    ids = list(dict.fromkeys(document_ids or []))  # de-dupe, keep order
    if len(ids) < 2:
        return RAGAnswer(
            answer="Select at least two papers to compare them.",
            confidence="low", citations=[])

    blocks, all_chunks = [], []
    for doc_id in ids:
        summary = await summarize_document(doc_id, llm)
        chunks = get_document_chunks(doc_id)
        all_chunks.extend(chunks)
        title = chunks[0].title if chunks else doc_id
        blocks.append(f"PAPER: {title} (id={doc_id})\n{summary.answer}")

    user = f"Question: {question}\n\nSummaries:\n\n" + "\n\n---\n\n".join(blocks)
    raw = (await llm.ainvoke([("system", COMPARE_SYSTEM), ("user", user)])).content
    parsed = _parse(raw)
    return RAGAnswer(
        answer=parsed.get("answer", "Could not compare the papers."),
        confidence=parsed.get("confidence", "medium"),
        citations=_citations(all_chunks),
    )
