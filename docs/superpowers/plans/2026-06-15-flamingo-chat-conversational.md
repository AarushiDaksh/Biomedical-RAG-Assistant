# Flamingo Conversational Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Flamingo answer whole-document questions (summaries), resolve follow-ups/"it" via conversation history, scope answers to user-selected papers, and compare 2+ papers.

**Architecture:** An LLM intent classifier routes each message to a handler (`summary` / `qa` / `compare` / `smalltalk`) via a registry in `rag/agent.py`. Handlers receive the browser-supplied `document_ids` (selection) and trimmed `history`. Summaries pull all chunks of the selected paper(s); Q&A does similarity search filtered to the selection; compare contrasts the selected set. The JSON answer + SSE contract is unchanged so the frontend stays compatible.

**Tech Stack:** Python 3.12, FastAPI (SSE), langchain-ollama `ChatOllama` (`llama3.2:3b`), langchain-chroma, pytest. Frontend: Next.js (rag-ui) — read `node_modules/next/dist/docs/` before editing.

**Scope note:** OCR for scanned PDFs is a *separate* plan (`2026-06-15-flamingo-ocr-ingestion.md`); it is not covered here.

**Test commands:** run from repo root with the venv interpreter:
`.venv/Scripts/python -m pytest tests/<file>::<test> -v`

---

### Task 0: Get the existing test suite green (pre-existing debt)

`test_guardrails.py` imports `enforce_citations` (absent) and `agent.py` reads
`guardrail.message` (field is `.reason`). Fix both before building.

**Files:**
- Modify: `rag/agent.py`
- Test: `tests/test_guardrails.py` (already exists; do not change)

- [ ] **Step 1: Run the existing test to see the import failure**

Run: `.venv/Scripts/python -m pytest tests/test_guardrails.py -v`
Expected: ERROR collecting — `ImportError: cannot import name 'enforce_citations' from 'rag.agent'`

- [ ] **Step 2: Add `enforce_citations` to `rag/agent.py`**

Add after `_build_answer` (around `rag/agent.py:96`):

```python
def enforce_citations(answer: RAGAnswer) -> RAGAnswer:
    """Defense-in-depth: a confident, non-empty answer must cite sources."""
    if not answer.citations and answer.answer and answer.confidence != "low":
        answer.refusal_reason = "missing_citations"
    return answer
```

- [ ] **Step 3: Fix the guardrail field bug**

In `rag/agent.py`, the guardrail-block branch uses `guardrail.message`. Change it to
`guardrail.reason`:

```python
    guardrail = check_input(question)
    if not guardrail.allowed:
        yield {
            "type": "done",
            "answer": guardrail.reason,
            "confidence": "low",
            "citations": [],
            "chunks": [],
            "trace_id": trace_id,
            "latency": round(time.time() - start, 2),
        }
        return
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_guardrails.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/agent.py
git commit -m "fix: restore enforce_citations and guardrail.reason in agent"
```

---

### Task 1: Retriever — selection filter + whole-document fetch

**Files:**
- Modify: `rag/retriever.py`
- Modify: `rag/vectorstore.py` (add `chunk_index` to metadata for stable ordering)
- Test: `tests/test_retriever.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_retriever.py`:

```python
from unittest.mock import MagicMock
import rag.retriever as retriever


def _doc(chunk_id, doc_id, page, idx, text="some text"):
    md = {"chunk_id": chunk_id, "document_id": doc_id, "title": "T",
          "page_number": page, "chunk_index": idx, "source_url": ""}
    d = MagicMock()
    d.metadata = md
    d.page_content = text
    return d


def test_retrieve_filters_by_document_ids(monkeypatch):
    db = MagicMock()
    db.similarity_search_with_relevance_scores.return_value = [(_doc("c1", "d1", 1, 0), 0.9)]
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    retriever.retrieve_chunks("q", k=5, document_ids=["d1", "d2"])

    _, kwargs = db.similarity_search_with_relevance_scores.call_args
    assert kwargs["filter"] == {"document_id": {"$in": ["d1", "d2"]}}


def test_retrieve_no_filter_when_no_selection(monkeypatch):
    db = MagicMock()
    db.similarity_search_with_relevance_scores.return_value = []
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    retriever.retrieve_chunks("q", k=5, document_ids=None)

    _, kwargs = db.similarity_search_with_relevance_scores.call_args
    assert kwargs["filter"] is None


def test_get_document_chunks_orders_by_page_then_index(monkeypatch):
    db = MagicMock()
    db.get.return_value = {
        "ids": ["c2", "c1", "c3"],
        "documents": ["page2", "page1-b", "page1-a"],
        "metadatas": [
            {"chunk_id": "c2", "document_id": "d1", "title": "T", "page_number": 2, "chunk_index": 0, "source_url": ""},
            {"chunk_id": "c1", "document_id": "d1", "title": "T", "page_number": 1, "chunk_index": 1, "source_url": ""},
            {"chunk_id": "c3", "document_id": "d1", "title": "T", "page_number": 1, "chunk_index": 0, "source_url": ""},
        ],
    }
    monkeypatch.setattr(retriever, "get_chroma", lambda: db)

    chunks = retriever.get_document_chunks("d1")

    assert [c.text for c in chunks] == ["page1-a", "page1-b", "page2"]
    db.get.assert_called_once_with(where={"document_id": "d1"})
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_retriever.py -v`
Expected: FAIL — `retrieve_chunks() got an unexpected keyword argument 'document_ids'` / `get_document_chunks` missing.

- [ ] **Step 3: Implement in `rag/retriever.py`**

Replace the body of `rag/retriever.py` with:

```python
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
    chunks = [_to_chunk(md, text) for md, text in items]
    chunks.sort(key=lambda c: (c.page_number,))  # stable; refined below
    # secondary sort by chunk_index (not on RetrievedChunk; sort on raw metadata)
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
```

- [ ] **Step 4: Add `chunk_index` to vector metadata in `rag/vectorstore.py`**

In `build_chroma`, the metadata dict (`rag/vectorstore.py:33-39`) add one line so future
indexing stores ordering. Use `.get` since older chunks may lack it:

```python
            metadata={
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "title": c.get("title", "Unknown title"),
                "page_number": c["page_number"],
                "chunk_index": c.get("chunk_index", 0),
                "source_url": c.get("source_url", ""),
            },
```

- [ ] **Step 5: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_retriever.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add rag/retriever.py rag/vectorstore.py tests/test_retriever.py
git commit -m "feat: document-id filter and whole-document fetch in retriever"
```

---

### Task 2: Intent classifier

**Files:**
- Create: `rag/intent.py`
- Test: `tests/test_intent.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_intent.py`:

```python
import asyncio
from unittest.mock import MagicMock
from rag.intent import classify_intent, VALID_INTENTS


class FakeLLM:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, messages):
        return MagicMock(content=self._content)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_classifies_summary():
    llm = FakeLLM('{"intent": "summary"}')
    assert _run(classify_intent("give me a summary of it", [], llm)) == "summary"


def test_classifies_compare():
    llm = FakeLLM('{"intent": "compare"}')
    assert _run(classify_intent("how do these two papers differ?", [], llm)) == "compare"


def test_unknown_label_falls_back_to_qa():
    llm = FakeLLM('{"intent": "banana"}')
    assert _run(classify_intent("anything", [], llm)) == "qa"


def test_unparseable_falls_back_to_qa():
    llm = FakeLLM("the model rambled with no json")
    assert _run(classify_intent("anything", [], llm)) == "qa"


def test_valid_intents_set():
    assert VALID_INTENTS == {"summary", "qa", "compare", "smalltalk"}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_intent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.intent'`

- [ ] **Step 3: Implement `rag/intent.py`**

```python
import json
from typing import List

VALID_INTENTS = {"summary", "qa", "compare", "smalltalk"}

CLASSIFY_SYSTEM = """You are an intent router for a biomedical paper assistant.
Classify the user's latest message into exactly one intent:
- "summary": wants an overview/summary/key findings/limitations/methods of a paper.
- "qa": asks a specific question answerable from a passage.
- "compare": wants two or more papers compared/contrasted.
- "smalltalk": greeting/thanks/chit-chat, not about paper content.
Use the conversation to resolve pronouns like "it" or "this paper".
Return ONLY JSON: {"intent": "summary|qa|compare|smalltalk"}"""


def _format_history(history: List[dict], limit: int = 6) -> str:
    turns = history[-limit:] if history else []
    return "\n".join(f"{t.get('role', '?')}: {t.get('content', '')}" for t in turns) or "(none)"


def _extract_intent(raw: str) -> str:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            label = json.loads(raw[start:end]).get("intent", "")
            if label in VALID_INTENTS:
                return label
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    return "qa"


async def classify_intent(question: str, history: List[dict], llm) -> str:
    user = f"Conversation so far:\n{_format_history(history)}\n\nLatest message:\n{question}"
    raw = (await llm.ainvoke([("system", CLASSIFY_SYSTEM), ("user", user)])).content
    return _extract_intent(raw)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_intent.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag/intent.py tests/test_intent.py
git commit -m "feat: LLM intent classifier with qa fallback"
```

---

### Task 3: Document summarization (stuff + map-reduce)

**Files:**
- Create: `rag/summarize.py`
- Test: `tests/test_summarize.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_summarize.py`:

```python
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
    return asyncio.get_event_loop().run_until_complete(coro)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_summarize.py -v`
Expected: FAIL — `No module named 'rag.summarize'`

- [ ] **Step 3: Implement `rag/summarize.py`**

```python
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
    import json
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
    partials = []
    for group in _groups(chunks, GROUP_CHAR_BUDGET):
        try:
            text = "\n\n".join(c.text for c in group)
            partials.append(await _summarize_text(SUMMARY_SYSTEM, text, llm))
        except Exception:
            continue
    if not partials:
        return RAGAnswer(answer="Summarization failed for this paper.",
                         confidence="low", citations=cites)
    combined = await _summarize_text(REDUCE_SYSTEM, "\n\n".join(partials), llm)
    return RAGAnswer(answer=combined or "\n\n".join(partials),
                     confidence="medium", citations=cites)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_summarize.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/summarize.py tests/test_summarize.py
git commit -m "feat: whole-document summarization with map-reduce fallback"
```

---

### Task 4: Multi-paper comparison

**Files:**
- Create: `rag/compare.py`
- Test: `tests/test_compare.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_compare.py`:

```python
import asyncio
from unittest.mock import MagicMock
import rag.compare as compare
from rag.schemas import RAGAnswer, RetrievedChunk


class FakeLLM:
    async def ainvoke(self, messages):
        return MagicMock(content='{"answer": "they differ in X", "confidence": "medium", "citations": []}')


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_compare.py -v`
Expected: FAIL — `No module named 'rag.compare'`

- [ ] **Step 3: Implement `rag/compare.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_compare.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add rag/compare.py tests/test_compare.py
git commit -m "feat: multi-paper comparison handler"
```

---

### Task 5: Agent routing — history, selection, num_ctx

**Files:**
- Modify: `rag/agent.py`
- Test: `tests/test_agent_routing.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent_routing.py`:

```python
import asyncio
import rag.agent as agent
from rag.schemas import RAGAnswer


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _collect(gen):
    return [e async for e in gen]


def test_summary_intent_routes_to_summarize(monkeypatch):
    monkeypatch.setattr(agent, "classify_intent",
                        lambda q, h, llm: _ai("summary"))
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


async def _ai(value):  # tiny awaitable returning value
    return value
```

(Note: `classify_intent` is async; the monkeypatch returns an awaitable.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_agent_routing.py -v`
Expected: FAIL — `answer_question_stream() got an unexpected keyword argument 'history'`

- [ ] **Step 3: Implement routing in `rag/agent.py`**

Add imports near the top:

```python
from .intent import classify_intent
from .summarize import summarize_document
from .compare import compare_documents
```

Set `num_ctx` on the singleton LLM (`rag/agent.py:20-25`):

```python
        _llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            format="json",
            num_ctx=8192,
        )
```

Add a history helper:

```python
def _history_block(history: list | None, limit: int = 6) -> str:
    turns = (history or [])[-limit:]
    if not turns:
        return ""
    lines = "\n".join(f"{t.get('role','?')}: {t.get('content','')}" for t in turns)
    return f"Conversation so far:\n{lines}\n\n"
```

Change the signature and add routing. Replace the section from the
`yield {"type": "status", "message": "Searching your papers…"}` line through the
end of the retrieval/answer logic with:

```python
async def answer_question_stream(question: str, history: list | None = None,
                                 document_ids: list | None = None):
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
```

Add these helpers (consolidate the duplicated `done` payload):

```python
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
```

Update `answer_question` (the non-streaming wrapper) call site if its signature changed
— it calls `answer_question_stream(question)`, which still works (new args default).

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_agent_routing.py tests/test_guardrails.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add rag/agent.py tests/test_agent_routing.py
git commit -m "feat: intent routing, history, and selection in answer stream"
```

---

### Task 6: Backend — request fields + GET /documents

**Files:**
- Modify: `backend/main.py`
- Test: `tests/test_documents_endpoint.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/test_documents_endpoint.py`:

```python
import json
from fastapi.testclient import TestClient


def test_documents_lists_indexed(tmp_path, monkeypatch):
    docs = tmp_path / "documents.jsonl"
    docs.write_text(json.dumps({
        "document_id": "d1", "title": "Paper One", "file_name": "one.pdf",
        "pages": [{"page_number": 1, "text": "x"}], "ocr_used": False,
    }) + "\n", encoding="utf-8")

    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path))
    import importlib, backend.main as m
    importlib.reload(m)
    client = TestClient(m.app)

    resp = client.get("/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["document_id"] == "d1"
    assert body[0]["pages"] == 1
    assert body[0]["scanned"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_documents_endpoint.py -v`
Expected: FAIL — 404 (no `/documents` route).

- [ ] **Step 3: Implement in `backend/main.py`**

Extend `ChatRequest`:

```python
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    document_ids: list[str] = []
    history: list[ChatTurn] = []
```

Pass new fields through the SSE stream — change `_sse_stream` and `/chat`:

```python
async def _sse_stream(question: str, history: list, document_ids: list):
    async for event in answer_question_stream(question, history=history,
                                              document_ids=document_ids):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    history = [t.model_dump() for t in req.history]
    return StreamingResponse(
        _sse_stream(req.message.strip(), history, req.document_ids),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

(The `_is_greeting` fast-path is now handled by the `smalltalk` intent; remove the
greeting branch from `_sse_stream`.)

Add the `/documents` endpoint (reads `documents.jsonl`):

```python
@app.get("/documents")
async def list_documents():
    from rag.config import PROCESSED_DIR
    docs_path = Path(ROOT) / PROCESSED_DIR / "documents.jsonl"
    if not docs_path.exists():
        return []
    out = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append({
                "document_id": d["document_id"],
                "title": d.get("title", "Untitled"),
                "file_name": d.get("file_name", ""),
                "pages": len(d.get("pages", [])),
                "scanned": bool(d.get("ocr_used", False)),
            })
    return out
```

- [ ] **Step 4: Run test to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_documents_endpoint.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_documents_endpoint.py
git commit -m "feat: chat history+selection fields and GET /documents"
```

---

### Task 7: Frontend — multi-select picker, history, send selection

**Files:**
- Modify: `rag-ui/app/page.tsx`

**Pre-step:** Read `rag-ui/node_modules/next/dist/docs/` for any breaking changes
(see `rag-ui/AGENTS.md`) before editing. This is a client component (`"use client"`),
so changes are React state + fetch, not server APIs.

- [ ] **Step 1: Add document list + selection state**

After the existing `useState` declarations (`rag-ui/app/page.tsx:101-110`), add:

```tsx
  type DocMeta = { document_id: string; title: string; file_name: string; pages: number; scanned: boolean };
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`);
      const data: DocMeta[] = await res.json();
      setDocs(data);
      setSelectedDocIds(prev => (prev.length ? prev : data.slice(-1).map(d => d.document_id)));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { refreshDocs(); }, [refreshDocs]);
```

Call `refreshDocs()` when indexing completes — in `pollJob`'s `data.status === "done"`
branch (`rag-ui/app/page.tsx:132-139`), add `refreshDocs();`.

- [ ] **Step 2: Send selection + history in `sendMessage`**

In `sendMessage`, replace the `/chat` fetch body (`rag-ui/app/page.tsx:178-182`):

```tsx
      const history = messages
        .filter(m => m.id !== "welcome")
        .slice(-6)
        .map(m => ({ role: m.role, content: m.text }));
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, document_ids: selectedDocIds, history }),
      });
```

- [ ] **Step 3: Remove the brittle summary gate**

Delete the `isSummaryRequest` function (`rag-ui/app/page.tsx:71-76`) and the early
return that shows `showSummaryWarning` in `sendMessage` (`rag-ui/app/page.tsx:172`).
Routing is now server-side. (Leave the summary modal markup or delete it; if deleted,
also remove `showSummaryWarning` state and its `<Modal>`.)

- [ ] **Step 4: Render the multi-select picker in the header**

In the header right-side cluster (near `rag-ui/app/page.tsx:308`), add a checkbox
dropdown listing `docs`, toggling membership in `selectedDocIds`:

```tsx
            {docs.length > 0 && (
              <details className="relative">
                <summary className="cursor-pointer list-none rounded-full px-3 py-1.5 text-xs font-medium"
                  style={{ background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.2)", color: "#67e8f9" }}>
                  {selectedDocIds.length} selected
                </summary>
                <div className="absolute right-0 z-50 mt-2 w-72 rounded-xl border border-white/10 bg-[#0d1a2e] p-2 shadow-2xl">
                  {docs.map(d => (
                    <label key={d.document_id} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-slate-300 hover:bg-white/5">
                      <input type="checkbox"
                        checked={selectedDocIds.includes(d.document_id)}
                        onChange={e => setSelectedDocIds(prev =>
                          e.target.checked ? [...prev, d.document_id] : prev.filter(x => x !== d.document_id))} />
                      <span className="truncate">{d.title}</span>
                      {d.scanned && <span className="ml-auto rounded bg-amber-500/15 px-1 text-[10px] text-amber-400">OCR</span>}
                    </label>
                  ))}
                </div>
              </details>
            )}
```

- [ ] **Step 5: Manual verification**

Start backend (`.venv/Scripts/python -m uvicorn backend.main:app --reload`), worker,
Ollama, and `npm run dev` in `rag-ui`. Upload a paper, confirm it appears in the
picker, select it, and ask "summarize it" → expect a real summary with page
citations. Ask a follow-up ("explain the methods simpler") → expect it to stay on
topic. Select two papers and ask "compare them" → expect a contrast.

- [ ] **Step 6: Commit**

```bash
git add rag-ui/app/page.tsx
git commit -m "feat: multi-select paper picker, conversation history, server-side routing"
```

---

## Self-Review

- **Spec coverage:** summary (Task 3), qa-with-history + selection (Tasks 1,5), compare
  (Task 4), intent classifier (Task 2), `GET /documents` + request fields (Task 6),
  frontend picker + history (Task 7), `num_ctx` raised (Task 5). OCR is the separate
  plan. ✓
- **Pre-existing debt:** Task 0 restores `enforce_citations` and fixes `guardrail.reason`
  so the suite runs. ✓
- **Type consistency:** `document_ids: list[str]` end-to-end; `classify_intent(question,
  history, llm)` async everywhere; `summarize_document(document_id, llm)` and
  `compare_documents(document_ids, question, llm)` signatures match their callers in
  Task 5; `_done/_cites/_chunk_payload` helpers defined in Task 5 and used there. ✓
- **Open verification during execution:** confirm `langchain_ollama.ChatOllama` accepts
  `num_ctx` (it does in current versions); confirm `langchain_chroma` `Chroma.get(where=)`
  returns `metadatas`/`documents` lists (it does). If a version differs, adjust in the
  task where it surfaces.
