import hashlib
import json
import os
from typing import Dict, Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_PATH, CHUNKS_PATH


def stable_chunk_id(document_id: str, page_number: int, chunk_index: int, text: str) -> str:
    raw = f"{document_id}:{page_number}:{chunk_index}:{text[:80]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def recursive_chunks(docs_path: str = DOCS_PATH, output_path: str = CHUNKS_PATH) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Dict] = []

    for doc in load_jsonl(docs_path):
        for page in doc["pages"]:
            page_text = page.get("text", "").strip()
            if not page_text:
                continue
            split_texts = splitter.split_text(page_text)
            for idx, text in enumerate(split_texts):
                chunk_id = stable_chunk_id(doc["document_id"], page["page_number"], idx, text)
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc["document_id"],
                    "title": doc.get("title", "Unknown title"),
                    "page_number": page["page_number"],
                    "chunk_index": idx,
                    "source_text": text,
                    "source_url": doc.get("source_url", ""),
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks


if __name__ == "__main__":
    chunks = recursive_chunks()
    print(f"Created {len(chunks)} chunks at {CHUNKS_PATH}")
