import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_MODEL)
CHROMA_DIR = os.getenv("CHROMA_DIR", "vectorstore/chroma")
RAW_PDF_DIR = os.getenv("RAW_PDF_DIR", "data/raw_pdfs")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.jsonl")
DOCS_PATH = os.path.join(PROCESSED_DIR, "documents.jsonl")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://rag_user:rag_password@localhost:5432/biomedical_rag")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
