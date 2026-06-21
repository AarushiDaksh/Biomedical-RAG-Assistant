from typing import List, Optional
from langchain_core.tools import tool

from .config import MIN_SIMILARITY, TOP_K
from .schemas import RetrievedChunk
from .vectorstore import get_chroma


def _to_chunk(md: dict, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=md["chunk_id"],
        document_id=md["document_id"],
        title=md.get("title", "Unknown title"),
        page_number=int(md.get("page_number", 0)),
        text=text,
        score=float(score),
        source_url=md.get("source_url", ""),
    )


def retrieve_chunks(query: str, k: int = TOP_K,
                    document_ids: Optional[List[str]] = None) -> List[RetrievedChunk]:
    db = get_chroma()
    where = {"document_id": {"$in": document_ids}} if document_ids else None
    results = db.similarity_search_with_relevance_scores(query, k=k, filter=where)
    chunks: List[RetrievedChunk] = []
    for doc, score in results:
        if score < MIN_SIMILARITY:
            continue
        chunks.append(_to_chunk(doc.metadata, doc.page_content, score))
    return chunks


def get_document_chunks(document_id: str) -> List[RetrievedChunk]:
    """Return every chunk of one document, ordered by page then chunk index."""
    db = get_chroma()
    got = db.get(where={"document_id": document_id})
    items = list(zip(got.get("metadatas", []), got.get("documents", [])))
    ordered = sorted(
        items, key=lambda mt: (int(mt[0].get("page_number", 0)), int(mt[0].get("chunk_index", 0)))
    )
    return [_to_chunk(md, text) for md, text in ordered]


@tool
def biomedical_retriever(query: str) -> str:
    """Retrieve relevant biomedical literature chunks for a research question."""
    chunks = retrieve_chunks(query)
    return "\n\n".join(
        f"[chunk_id={c.chunk_id}; doc={c.document_id}; title={c.title}; page={c.page_number}; score={c.score:.3f}]\n{c.text}"
        for c in chunks
    )
