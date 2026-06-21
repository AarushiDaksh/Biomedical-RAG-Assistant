# Flamingo OCR Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scanned/image PDFs usable by OCR-ing pages that have no extractable text at ingest time, so they become searchable, summarizable, and comparable like any other paper.

**Architecture:** During ingestion, each page is first read with `pypdf`. If a page yields little/no text, it is rasterized with PyMuPDF and recognized with EasyOCR. OCR is per-page (mixed PDFs work). The EasyOCR `Reader` is a lazy singleton whose models download to a shared E: cache. Documents record `ocr_used` so the UI can badge them.

**Tech Stack:** Python 3.12, `easyocr` (pulls `torch`), `pymupdf` (fitz) for rasterization, `pypdf` (existing). All pip-installable; no system binary.

**Independence:** This plan touches only the ingestion path (`rag/ingest.py`, new `rag/ocr.py`, config, deps). It does not depend on the chat plan. The `ocr_used` flag is surfaced by the chat plan's `GET /documents` (Task 6 there); this plan just writes the flag.

**Environment:** Install with the **venv** pip so torch lands on E:; models cache to
`e:\ml-cache\easyocr` (see the spec's "Local Environment" section).

**Test commands:** `.venv/Scripts/python -m pytest tests/<file>::<test> -v`

---

### Task 1: Install dependencies (on E:)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add deps to `requirements.txt`**

Append:

```
easyocr>=1.7.0
pymupdf>=1.24.0
```

- [ ] **Step 2: Create the shared E: cache dirs**

```bash
mkdir -p /e/ml-cache/pip /e/ml-cache/easyocr /e/ml-cache/huggingface
```

- [ ] **Step 3: Install with the venv pip, caching to E:**

```bash
PIP_CACHE_DIR=e:/ml-cache/pip .venv/Scripts/python -m pip install "easyocr>=1.7.0" "pymupdf>=1.24.0"
```

Expected: torch + easyocr + pymupdf install into `.venv` (on E:). Large download.

- [ ] **Step 4: Verify imports use the venv interpreter**

Run: `.venv/Scripts/python -c "import easyocr, fitz; print('ok')"`
Expected: `ok` (no ImportError).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "build: add easyocr and pymupdf for scanned-PDF OCR"
```

---

### Task 2: Config for OCR

**Files:**
- Modify: `rag/config.py`
- Test: `tests/test_ocr_config.py` (create)

- [ ] **Step 1: Write failing test**

Create `tests/test_ocr_config.py`:

```python
import importlib


def test_ocr_defaults(monkeypatch):
    monkeypatch.delenv("OCR_MODEL_DIR", raising=False)
    monkeypatch.delenv("MIN_PAGE_TEXT_CHARS", raising=False)
    import rag.config as config
    importlib.reload(config)
    assert config.OCR_MODEL_DIR.endswith("easyocr")
    assert config.MIN_PAGE_TEXT_CHARS == 20
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_config.py -v`
Expected: FAIL — `AttributeError: module 'rag.config' has no attribute 'OCR_MODEL_DIR'`

- [ ] **Step 3: Add config (`rag/config.py`)**

Append after the existing settings:

```python
OCR_MODEL_DIR = os.getenv("OCR_MODEL_DIR", r"e:\ml-cache\easyocr")
MIN_PAGE_TEXT_CHARS = int(os.getenv("MIN_PAGE_TEXT_CHARS", "20"))
```

- [ ] **Step 4: Run test to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr_config.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add rag/config.py tests/test_ocr_config.py
git commit -m "feat: OCR config (model dir on E:, min page-text threshold)"
```

---

### Task 3: OCR module (lazy reader + page rasterize/recognize)

**Files:**
- Create: `rag/ocr.py`
- Test: `tests/test_ocr.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_ocr.py`:

```python
from unittest.mock import MagicMock
import rag.ocr as ocr


def test_get_reader_is_singleton(monkeypatch):
    ocr._reader = None
    created = []

    class FakeReader:
        def __init__(self, langs, model_storage_directory=None, **kw):
            created.append(model_storage_directory)

    fake_easyocr = MagicMock()
    fake_easyocr.Reader = FakeReader
    monkeypatch.setattr(ocr, "easyocr", fake_easyocr, raising=False)

    r1 = ocr.get_reader()
    r2 = ocr.get_reader()
    assert r1 is r2
    assert len(created) == 1  # constructed once


def test_ocr_image_joins_text(monkeypatch):
    reader = MagicMock()
    reader.readtext.return_value = ["Hello", "World"]   # detail=0 -> list[str]
    monkeypatch.setattr(ocr, "get_reader", lambda: reader)
    assert ocr.ocr_image(b"fakepng") == "Hello World"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_ocr.py -v`
Expected: FAIL — `No module named 'rag.ocr'`

- [ ] **Step 3: Implement `rag/ocr.py`**

```python
import os

import easyocr  # heavy; importing the module is cheap, Reader() is not

from .config import OCR_MODEL_DIR

_reader = None


def get_reader():
    """Lazy EasyOCR reader; models cache under OCR_MODEL_DIR (E: drive)."""
    global _reader
    if _reader is None:
        os.makedirs(OCR_MODEL_DIR, exist_ok=True)
        _reader = easyocr.Reader(["en"], model_storage_directory=OCR_MODEL_DIR)
    return _reader


def ocr_image(png_bytes: bytes) -> str:
    """OCR raw PNG bytes -> recovered text (space-joined)."""
    reader = get_reader()
    lines = reader.readtext(png_bytes, detail=0)
    return " ".join(s.strip() for s in lines if s and s.strip())


def ocr_pdf_page(pdf_path: str, page_index: int, zoom: float = 2.0) -> str:
    """Rasterize one PDF page with PyMuPDF, then OCR it."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return ocr_image(pix.tobytes("png"))
    finally:
        doc.close()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_ocr.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add rag/ocr.py tests/test_ocr.py
git commit -m "feat: EasyOCR module with lazy reader and PDF-page OCR"
```

---

### Task 4: Wire OCR fallback into ingestion

**Files:**
- Modify: `rag/ingest.py`
- Test: `tests/test_ingest_ocr.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingest_ocr.py`:

```python
from unittest.mock import MagicMock
import rag.ingest as ingest


def _fake_reader(page_texts):
    reader = MagicMock()
    pages = []
    for t in page_texts:
        p = MagicMock()
        p.extract_text.return_value = t
        pages.append(p)
    reader.pages = pages
    reader.metadata = None
    return reader


def test_ocr_runs_only_on_empty_pages(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")  # content irrelevant; PdfReader is mocked
    monkeypatch.setattr(ingest, "PdfReader", lambda p: _fake_reader(["real text here is long enough", ""]))

    calls = []
    monkeypatch.setattr(ingest, "ocr_pdf_page", lambda path, idx: calls.append(idx) or "ocr recovered text")

    doc = ingest.extract_pdf_pypdf(str(pdf))

    assert calls == [1]                       # only the empty (2nd) page was OCR'd
    assert doc["pages"][1]["text"] == "ocr recovered text"
    assert doc["ocr_used"] is True


def test_no_ocr_when_all_pages_have_text(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(ingest, "PdfReader", lambda p: _fake_reader(["page one text long", "page two text long"]))

    calls = []
    monkeypatch.setattr(ingest, "ocr_pdf_page", lambda path, idx: calls.append(idx) or "x")

    doc = ingest.extract_pdf_pypdf(str(pdf))
    assert calls == []
    assert doc["ocr_used"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python -m pytest tests/test_ingest_ocr.py -v`
Expected: FAIL — `ocr_pdf_page` not importable in `rag.ingest` / `ocr_used` missing.

- [ ] **Step 3: Implement the fallback in `rag/ingest.py`**

Add the import near the top:

```python
from .config import RAW_PDF_DIR, DOCS_PATH, MIN_PAGE_TEXT_CHARS
from .ocr import ocr_pdf_page
```

Rewrite `extract_pdf_pypdf` to OCR empty pages and record `ocr_used`:

```python
def extract_pdf_pypdf(path: str) -> Dict:
    reader = PdfReader(path)
    doc_id = file_hash(path)
    title = reader.metadata.title if reader.metadata and reader.metadata.title else Path(path).stem
    pages: List[Dict] = []
    ocr_used = False

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_PAGE_TEXT_CHARS:
            try:
                recovered = ocr_pdf_page(path, i - 1).strip()
            except Exception:
                recovered = ""
            if recovered:
                text = recovered
                ocr_used = True
        pages.append({"page_number": i, "text": text})

    return {
        "document_id": doc_id,
        "title": title,
        "authors": "",
        "year": "",
        "source_url": "",
        "file_name": Path(path).name,
        "loader": "pypdf+ocr" if ocr_used else "pypdf",
        "ocr_used": ocr_used,
        "pages": pages,
    }
```

Update the no-text warning in `ingest_folder` (`rag/ingest.py:47-49`) so it only warns
when OCR also recovered nothing:

```python
        non_empty_pages = sum(1 for p in doc["pages"] if p["text"])
        if non_empty_pages == 0:
            doc["warning"] = "No text found even after OCR. The scan may be blank or illegible."
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_ingest_ocr.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_ingest_ocr.py
git commit -m "feat: per-page OCR fallback for scanned PDFs at ingest"
```

---

### Task 5: End-to-end verification with a real scanned PDF

- [ ] **Step 1: Run the full suite**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: all passing.

- [ ] **Step 2: Manual scanned-PDF check**

Place a known image-only/scanned PDF in `data/raw_pdfs/` and run the indexer:

Run: `.venv/Scripts/python -m rag.ingest`
Then chunk + vectorize, or upload through the running app.
Expected: the document's `documents.jsonl` entry has `"ocr_used": true` and non-empty
page text; the first EasyOCR run downloads models into `e:\ml-cache\easyocr` (not C:).

- [ ] **Step 3: Confirm the UI badge**

With the chat plan's `GET /documents` in place, the scanned paper shows the `OCR`
badge in the picker. (If the chat plan isn't merged yet, verify via
`curl http://localhost:8000/documents` returning `"scanned": true`.)

---

## Self-Review

- **Spec coverage:** EasyOCR + PyMuPDF (Tasks 1,3), per-page fallback for mixed PDFs
  (Task 4), `ocr_used`/`scanned` recorded (Task 4) and surfaced (Task 5 / chat plan
  Task 6), models on E: (Tasks 1,2,3), lazy singleton reader (Task 3). ✓
- **Placeholder scan:** all steps contain real code/commands. ✓
- **Type consistency:** `ocr_pdf_page(pdf_path, page_index)` defined in Task 3 and
  monkeypatched/called identically in Task 4; `OCR_MODEL_DIR` / `MIN_PAGE_TEXT_CHARS`
  defined in Task 2 and used in Tasks 3,4. ✓
- **Note:** EasyOCR `readtext(..., detail=0)` returns `list[str]`; `ocr_image` joins
  them. If a version returns tuples, set `detail=0` (already done) or extract `[1]`.
