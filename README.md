



# Biomedical RAG Assistant

Biomedical RAG Assistant is a citation-grounded research copilot for biomedical teams. It lets users upload internal or public PDFs, indexes them asynchronously, retrieves evidence from the indexed corpus, and answers with traceable citations instead of unsupported claims.

## Setup

This repo is now Docker-first. The default setup runs the full stack with one command:

- `frontend` - Next.js UI on `http://localhost:3000`
- `backend` - FastAPI API on `http://localhost:8000`
- `celery` - background indexing worker
- `postgres` - application database on host port `5433`
- `redis` - Celery broker/result backend on `6379`
- `ollama` - local model runtime on `11434`

Docker uses Ollama embeddings by default so the image does not have to download the very large `sentence-transformers` and `torch` stack during build.

### Prerequisites

- Docker Desktop with Compose support
- At least 10-12 GB free disk space for Python, Node, and Ollama assets
- Internet access the first time you build images and pull the Ollama model

### Standardized project structure

```text
backend/        FastAPI API, database models, Celery worker
rag/            RAG pipeline: ingest, chunking, retrieval, agent logic
rag-ui/         Next.js frontend
data/           Raw PDFs and processed JSONL artifacts
vectorstore/    Persistent Chroma vector store
tests/          Backend tests
docs/           Product and implementation notes
docker-compose.yml
requirements.txt
requirements.local-embeddings.txt
.env.example
```

### First-time run

1. Create the environment file:

```bash
copy .env.example .env
```

PowerShell alternative:

```powershell
Copy-Item .env.example .env
```

2. Start the complete platform:

```bash
docker compose up --build
```

This Docker path uses:

- `OLLAMA_MODEL=llama3.2:3b` for chat
- `OLLAMA_EMBEDDING_MODEL=nomic-embed-text` for embeddings

That keeps the image much smaller and avoids the multi-GB PyTorch/CUDA download path.

3. Open the product:

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:8000/health`

4. Upload PDFs from the UI and start asking grounded questions.

### Useful Docker commands

```bash
docker compose up --build -d
docker compose logs -f backend
docker compose logs -f celery
docker compose ps
docker compose down
```

### Why Docker was slow before

- `sentence-transformers` pulled in `torch`
- Linux `torch` resolution was pulling large NVIDIA/CUDA wheels
- the old Dockerfile also disabled pip caching with `PIP_NO_CACHE_DIR=1`

The backend image now:

- excludes `sentence-transformers` from the container requirements
- uses Ollama embeddings inside Docker
- uses a BuildKit pip cache mount so repeat builds can reuse downloaded wheels

If Docker cache is enabled in Docker Desktop, later rebuilds should be noticeably faster unless `requirements.txt` changes.

### Optional local setup for Hugging Face embeddings

If you want to run the project outside Docker and keep the original local embedding model:

1. Create and activate a virtual environment.
2. Install the base dependencies:

```bash
pip install -r requirements.txt
```

3. Install the optional local embedding package set:

```bash
pip install -r requirements.local-embeddings.txt
```

4. Set these variables in `.env`:

```bash
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

If you want to use Ollama embeddings outside Docker too, use:

```bash
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Notes

- The first startup can take a while because Docker builds the images and `ollama-init` pulls both `llama3.2:3b` and `nomic-embed-text`.
- PostgreSQL is exposed on `5433` to avoid collisions with a local PostgreSQL already using `5432`.
- The frontend is configured to call the backend at `http://localhost:8000` in Docker-hosted local usage.

## Technical Flow

1. A user uploads one or more biomedical PDFs in the Next.js frontend.
2. The frontend posts the files to the FastAPI backend.
3. The backend stores the PDFs under `data/raw_pdfs`, creates an indexing job row, and sends an async task to Celery through Redis.
4. Celery runs the ingestion pipeline:
   `rag.ingest` extracts document text and metadata.
   `rag.chunking` creates citation-preserving chunks.
   `rag.vectorstore` generates embeddings and stores them in Chroma.
5. The frontend polls job status until indexing finishes.
6. During chat, the frontend sends the user message, selected document IDs, and short conversation history to `/chat`.
7. The backend streams a response using Server-Sent Events.
8. The RAG agent retrieves the most relevant chunks, applies guardrails, composes an answer, and returns citations, confidence, timing, and a trace ID.
9. Feedback votes are stored in PostgreSQL for later quality review and analytics.

## Business Idea

The product is aimed at biomedical, pharma, clinical research, and regulatory teams that work in high-stakes document environments. Their problem is not just “finding answers faster”; it is finding answers they can actually trust, cite, and defend.

Generic chat tools are weak in this setting because they can hallucinate sources, blur proprietary data boundaries, and offer no audit trail. This product is valuable because it stays grounded in the uploaded corpus, returns page-linked evidence, and can run on private infrastructure with local models.

The strongest commercial wedge is regulatory affairs, medical writing, CRO, and research operations teams. They already spend heavily on document-heavy workflows, the cost of a wrong answer is real, and the ROI story is straightforward: faster evidence retrieval, better traceability, and less manual synthesis time.

## What Next We Can Add

- PubMed or ClinicalTrials.gov import so the corpus can be expanded without manual PDF hunting.
- Auth, user workspaces, and document-level access control for team and enterprise usage.
- Citation export to Word, PDF, or submission-ready evidence packs.
- Admin analytics for feedback quality, document coverage, query trends, and knowledge gaps.
- OCR and scanned-PDF enrichment for lower-quality source documents.
- Evaluation dashboards for retrieval quality, citation faithfulness, and answer confidence over time.
