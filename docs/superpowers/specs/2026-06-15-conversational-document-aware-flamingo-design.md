# Conversational, Document-Aware Flamingo — Design

**Date:** 2026-06-15
**Status:** Approved (pending spec review)

## Problem

Flamingo today is a single-shot "search-and-answer" engine with gaps the user hit directly:

1. **Cannot summarize or answer whole-document questions.** Retrieval is pure top-k
   vector similarity with a hard `MIN_SIMILARITY = 0.25` cutoff
   (`rag/retriever.py:11-14`). A query like "give me a summary of it" embeds the
   *instruction*, which has no semantic overlap with any passage, so every chunk
   scores below the cutoff, `chunks` is empty, and the agent returns
   "I could not find relevant passages" (`rag/agent.py:129-138`). A content word
   like "covid" matches passages and works.
2. **No conversation memory.** `/chat` is fully stateless: `ChatRequest` is just
   `message: str` (`backend/main.py:48-49`), the LLM is called fresh each turn
   (`rag/agent.py:152`), and there is no conversation table
   (`backend/database.py` has only `Feedback` and `IndexJob`). So "it" in
   "summarize it" has no referent, and follow-ups lose all context.
3. **Scanned PDFs produce nothing.** Ingestion uses `pypdf` text extraction only
   (`rag/ingest.py:20-39`); a scanned/image PDF yields no text and is flagged with a
   warning, but the paper is effectively unusable.

A partial summary stub already exists in the frontend (`rag-ui/app/page.tsx:71-76`)
that sends `"summarize uploaded pdfs"` to the backend, but the backend never
implemented summarization, so it falls through to the same failing similarity path.

## Goals

- Flamingo can summarize and answer whole-document questions ("what is this about",
  "key findings", "limitations", "methodology").
- Follow-up questions and pronouns ("it", "this paper") resolve using recent history.
- The user selects one or more papers; questions/summaries/comparisons act on the
  selected set.
- **Multi-paper comparison**: compare 2+ selected papers in one answer.
- **OCR for scanned PDFs**: pages with no extractable text are OCR'd at ingest so
  scanned papers become searchable and summarizable.
- No new hard dependency on Postgres (chat must keep working without it).

## Non-Goals

- Cross-device / persisted chat history.
- OCR tuning beyond default English recognition (no handwriting, no per-language
  config UI, no layout/table reconstruction).
- Cloud OCR or any cloud LLM/embedding calls (stays local-first).

## Key Decisions (from brainstorming)

1. **Document scope = papers the user selects.** A multi-select header picker; the
   selected `document_ids` drive scope. Backend filters retrieval/summarization by
   that set.
2. **Conversation history = browser-supplied.** The frontend already holds the full
   message list in React state (`rag-ui/app/page.tsx:101`). It sends the last ~6
   turns with each request; the backend trims and injects them. No server state,
   no Postgres dependency, survives server restarts.
3. **Summarization = stuff, with map-reduce fallback.** Fit the whole paper in one
   prompt when it fits (raise Ollama `num_ctx` to ~8k); fall back to chunk-group
   map-reduce for papers that exceed the budget.
4. **Intent detection = LLM classifier (not regex).** A dedicated, cheap
   `llama3.2:3b` classification call with constrained output, driving a handler
   registry. Returns `intent` only — scope comes from the picker selection.
5. **Comparison = multi-select.** `compare` acts on exactly the selected papers;
   requires ≥2 selected.
6. **OCR = EasyOCR (pure pip).** Rasterize pages with PyMuPDF, recognize with
   EasyOCR. No system binary; both are pip-installable.
7. **Heavy downloads live on E:, in a shared cache.** Packages, the pip wheel
   cache, EasyOCR models, and the HF embedding model all default to (or risk
   landing on) C:. Route them to a shared `e:\ml-cache\` so they're reusable
   across projects and off the system drive. See "Local Environment" below.

## Architecture

A new **intent router** in the agent dispatches each message to a handler via a
registry. All handlers receive the selected `document_ids` and trimmed conversation
`history`.

```
type message
  → frontend sends { message, document_ids: [...selected], history: last 6 turns }
  → guardrail check (existing)
  → classify_intent(message, history) -> intent
  → handler registry:
        smalltalk → canned greeting (existing greeting logic, generalized)
        summary   → summarize_document(docs)        [all chunks of selected, no similarity]
        qa        → retrieve_chunks(message, docs) + history   [passage answer, filtered]
        compare   → compare_documents(docs, message) [≥2 docs; cross-doc answer]
  → LLM (num_ctx raised, JSON contract) → SSE stream → render
```

OCR is a **separate, ingestion-side workstream** with no coupling to the chat path:

```
upload → ingest_folder → per page: pypdf text extract
                          → if empty/low text: rasterize (PyMuPDF) → EasyOCR → text
       → chunk → vectorize (unchanged downstream)
```

### Components

**Backend — chat path**

1. `rag/intent.py` *(new)* — `classify_intent(question, history) -> str`.
   - Calls the singleton LLM with a tiny, constrained prompt; `temperature=0`,
     small max tokens.
   - Returns one of `"summary" | "qa" | "compare" | "smalltalk"`.
   - Uses history so pronouns ("it") and follow-ups inform routing.
   - Defensive default: unparseable / low-signal output falls back to `qa`.

2. `rag/retriever.py` — extend `retrieve_chunks(query, k, document_ids=None)` to pass
   a Chroma `filter={"document_id": {"$in": document_ids}}` when a non-empty set is
   given (empty/None → search all). Add `get_document_chunks(document_id) ->
   list[RetrievedChunk]` using `db.get(where={"document_id": id})`, ordered by
   `page_number` then chunk order.

3. `rag/summarize.py` *(new)* — `summarize_document(document_id)`:
   - Fetch all chunks for the doc (ordered).
   - Estimate token budget; fits (~8k `num_ctx`) → single "stuff" prompt; else →
     map-reduce (summarize chunk-groups, then combine).
   - Citations reference page ranges. If a sub-summary fails in map-reduce, return
     the partial with a note rather than erroring out.
   - When multiple docs are selected for a `summary` intent, summarize each and
     concatenate (clearly labeled per paper).

4. `rag/compare.py` *(new)* — `compare_documents(document_ids, question)`:
   - Requires ≥2 docs; if fewer, return a prompt to select more.
   - Pull a representative context per doc (whole-doc if it fits, else its summary
     via `summarize_document`), then a single comparison prompt that contrasts them
     along the user's question (or default axes: aims, methods, findings, limits).
   - Citations attribute claims to each source paper.

5. `rag/agent.py` — `answer_question_stream(question, history=None, document_ids=None)`:
   - guardrail → `classify_intent` → dispatch via handler registry.
   - Inject trimmed history into the prompt for the `qa`/`summary`/`compare` handlers.
   - Raise Ollama `num_ctx` on the singleton `ChatOllama` (exact param name for
     `langchain_ollama` confirmed during implementation; likely `num_ctx`).
   - Preserve the existing JSON answer contract (`answer`, `confidence`,
     `citations`) and SSE event shape so the frontend stays compatible.

6. `backend/main.py`:
   - `ChatRequest` gains `history: list[ChatTurn] = []` and
     `document_ids: list[str] = []` (`ChatTurn = {role, content}`).
   - New `GET /documents` → reads `documents.jsonl`, returns
     `[{ document_id, title, file_name, pages, scanned }]` for the picker
     (`scanned` = OCR was used / no native text).
   - Pass `history` and `document_ids` into `answer_question_stream`.

**Backend — OCR (ingestion) workstream**

7. `rag/ingest.py` — in per-page extraction, when `page.extract_text()` is empty or
   below a small character threshold, rasterize the page with PyMuPDF and run EasyOCR
   to recover text. Mixed PDFs (some text pages, some scanned) are handled per page.
   - EasyOCR `Reader` is heavy to initialize (loads models) → lazy singleton,
     created only when OCR is actually needed, with
     `model_storage_directory=e:\ml-cache\easyocr` (see "Local Environment").
   - Record on the document whether OCR was used (`ocr_used: bool`) and drop/relax
     the existing "no extractable text" warning when OCR recovered text.
8. `requirements.txt` — add `easyocr` and `pymupdf`. Note: EasyOCR pulls in `torch`
   (large download) — called out in the plan/README as a setup cost.

**Frontend** (`rag-ui` — its `AGENTS.md` warns this Next.js has breaking changes;
read `node_modules/next/dist/docs/` before writing any frontend code)

9. **Multi-select paper picker** in the header, populated from `GET /documents`,
   defaulting to the most-recently-indexed paper selected. Stores
   `selectedDocIds: string[]`.
10. `/chat` request body includes `document_ids: selectedDocIds` and
    `history: <last 6 messages mapped to {role, content}>`.
11. Replace the brittle exact-match `isSummaryRequest` (`page.tsx:71-76`) — routing
    is server-side. Keep the "summary may take 30–60s" modal but trigger it on a
    looser match, or drop the gate entirely.
12. Surface a small "scanned (OCR)" badge on papers where `scanned` is true.

## Data Model / Contracts

- **`ChatRequest`** (new fields):
  ```
  { message: str, document_ids: [str], history: [{role: "user"|"assistant", content: str}] }
  ```
- **`GET /documents`** response:
  ```
  [{ document_id: str, title: str, file_name: str, pages: int, scanned: bool }]
  ```
- **Intent** (internal): one of `"summary" | "qa" | "compare" | "smalltalk"`.
- SSE `done`/`status`/`error` event shapes are unchanged.

## Error Handling

- **Summary/qa with empty selection:** fall back to most-recent doc; if zero papers
  indexed → "Upload a paper first, then ask me about it."
- **`compare` with <2 selected:** return a clear message asking to select ≥2 papers.
- **Scanned PDF where OCR also fails** (e.g. blank/illegible scan): keep a warning on
  the document and surface it instead of an empty summary.
- **Map-reduce partial failure:** return the partial summary with a note.
- **Over-long history:** trim to the last N turns / a token budget before prompting.
- **Classifier failure:** default to `qa`, so a bad classification never makes the
  assistant worse than today.
- **Postgres down:** unaffected — history is browser-supplied, `/documents` reads a
  file, not the DB.
- **OCR engine init failure / missing deps:** ingestion logs the error and falls back
  to the existing "no extractable text" warning; native-text PDFs are unaffected.

## Local Environment (Windows, E: drive)

All heavy ML downloads go to a **shared, reusable** cache on E:, never C:. The
project `.venv` is already on E:.

| What | Target | How |
|------|--------|-----|
| `torch`, `easyocr`, `pymupdf` packages | `e:\...\.venv\Lib\site-packages` | install with the **venv** pip: `\.venv\Scripts\python -m pip install ...` (NOT the global C: Python that's currently on PATH) |
| pip wheel cache | `e:\ml-cache\pip` | env `PIP_CACHE_DIR` (or `--cache-dir`) |
| EasyOCR models (~64 MB) | `e:\ml-cache\easyocr` | `easyocr.Reader(..., model_storage_directory=...)` |
| HF embedding model (~90 MB) | `e:\ml-cache\huggingface` | env `HF_HOME` (already on C:; relocate optional) |

- Create `e:\ml-cache\{pip,easyocr,huggingface}` once.
- The env vars (`PIP_CACHE_DIR`, `HF_HOME`) belong in the shell/run environment, not
  the app's `.env` (python-dotenv doesn't influence pip). The EasyOCR path is set in
  code, so it's durable regardless of env.
- **Pitfall:** the shell's `python` resolves to global C: Python
  (`C:\Users\alaukik\AppData\Local\Programs\Python\Python312`). Always use the
  `.venv` interpreter for installs/runs or torch lands on C:.

## Testing (TDD)

- `classify_intent`: summary vs qa vs compare vs smalltalk for representative phrases,
  including the four suggested-prompt buttons and pronoun cases with history
  ("summarize it"). LLM mocked.
- `get_document_chunks`: returns only the requested doc's chunks, ordered by page.
- `retrieve_chunks` with `document_ids`: `$in` filter applied; empty → no filter.
- `summarize_document`: stuff vs map-reduce path selection by token budget;
  partial-failure handling. LLM mocked.
- `compare_documents`: <2 docs → guidance message; ≥2 → both papers represented.
  LLM mocked.
- OCR: a page with no native text triggers OCR and yields recovered text (EasyOCR
  mocked / tiny fixture image); a native-text page does **not** invoke OCR; mixed PDF
  handled per page; `ocr_used`/`scanned` recorded.
- `/chat` integration: history resolves "it"; `document_ids` scopes the answer.
- `GET /documents`: returns the indexed list (with `scanned`) from `documents.jsonl`.

## Rollout / Sequencing

The two workstreams are independent and can proceed in parallel.

**Chat workstream**
1. Retrieval + document-fetch primitives (`retriever.py`).
2. `intent.py` classifier + handler registry in `agent.py`.
3. `summarize.py`, then `compare.py`.
4. `GET /documents` + `ChatRequest` fields in `main.py`.
5. Frontend multi-select picker + request-body changes.

**OCR workstream**
6. `ingest.py` per-page OCR fallback (PyMuPDF rasterize + EasyOCR) + deps.
7. `scanned`/`ocr_used` surfaced through `/documents` and the picker.

**Finally**
8. End-to-end verification across both.
