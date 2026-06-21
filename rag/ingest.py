import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from .config import RAW_PDF_DIR, DOCS_PATH


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def extract_pdf_pypdf(path: str) -> Dict:
    reader = PdfReader(path)
    doc_id = file_hash(path)
    title = reader.metadata.title if reader.metadata and reader.metadata.title else Path(path).stem
    pages: List[Dict] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page_number": i, "text": text.strip()})

    return {
        "document_id": doc_id,
        "title": title,
        "authors": "",
        "year": "",
        "source_url": "",
        "file_name": Path(path).name,
        "loader": "pypdf",
        "pages": pages,
    }


def ingest_folder(pdf_dir: str = RAW_PDF_DIR, output_path: str = DOCS_PATH) -> List[Dict]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    docs = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        doc = extract_pdf_pypdf(str(pdf))
        non_empty_pages = sum(1 for p in doc["pages"] if p["text"])
        if non_empty_pages == 0:
            doc["warning"] = "No extractable text found. This may be scanned PDF; run OCR or replace paper."
        docs.append(doc)

    with open(output_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs


if __name__ == "__main__":
    docs = ingest_folder()
    print(f"Ingested {len(docs)} PDFs into {DOCS_PATH}")
