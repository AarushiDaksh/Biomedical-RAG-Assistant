import os
from dotenv import load_dotenv

load_dotenv()

# Chat/generation provider: "groq" (hosted) or "ollama" (local). Embeddings are separate.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
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

# Max characters of a paper fed to a summary in a single LLM call. Sized to fit
# under hosted free-tier per-minute token limits (~16k chars ≈ 4k tokens). Long
# papers are sampled head+tail (key sections) rather than summarized page-by-page.
SUMMARY_CHAR_BUDGET = int(os.getenv("SUMMARY_CHAR_BUDGET", "16000"))

# Cross-encoder reranking: fetch more candidates from the vector store, then
# rerank with a lightweight cross-encoder and keep the best TOP_K.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANK_CACHE_DIR = os.getenv("RERANK_CACHE_DIR", "data/flashrank")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://rag_user:rag_password@localhost:5432/biomedical_rag")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")]
