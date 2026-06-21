import json
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from .config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
)


def get_embeddings():
    if EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'. "
        "Use 'huggingface' or 'ollama'."
    )


def load_chunks(path: str = CHUNKS_PATH) -> List[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_chroma(chunks_path: str = CHUNKS_PATH, persist_dir: str = CHROMA_DIR) -> Chroma:
    os.makedirs(persist_dir, exist_ok=True)
    chunks = load_chunks(chunks_path)
    docs = []
    ids = []
    for c in chunks:
        docs.append(Document(
            page_content=c["source_text"],
            metadata={
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "title": c.get("title", "Unknown title"),
                "page_number": c["page_number"],
                "chunk_index": c.get("chunk_index", 0),
                "source_url": c.get("source_url", ""),
            },
        ))
        ids.append(c["chunk_id"])

    db = Chroma(collection_name="biomedical_rag", embedding_function=get_embeddings(), persist_directory=persist_dir)
    existing = set(db.get().get("ids", []))
    new_docs = [d for d, i in zip(docs, ids) if i not in existing]
    new_ids = [i for i in ids if i not in existing]
    if new_docs:
        db.add_documents(new_docs, ids=new_ids)
    return db


def get_chroma(persist_dir: str = CHROMA_DIR) -> Chroma:
    return Chroma(collection_name="biomedical_rag", embedding_function=get_embeddings(), persist_directory=persist_dir)


if __name__ == "__main__":
    db = build_chroma()
    print("Vector store ready:", CHROMA_DIR)
