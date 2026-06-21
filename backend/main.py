import json
import sys
import time
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from rag.config import ALLOWED_ORIGINS, RAW_PDF_DIR, PROCESSED_DIR, CHROMA_DIR
from rag.agent import answer_question_stream
from backend.database import init_db, get_session, Feedback, IndexJob
from backend.tasks import index_pdfs_task

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Biomedical RAG Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    try:
        await init_db()
    except Exception as exc:
        # Postgres not running — chat still works, feedback/upload-status endpoints won't
        import logging
        logging.getLogger("uvicorn.error").warning(
            "Database unavailable at startup (%s). "
            "Start PostgreSQL to enable feedback and job tracking.", exc
        )


# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    document_ids: list[str] = []
    history: list[ChatTurn] = []


class FeedbackRequest(BaseModel):
    trace_id: str
    question: str
    answer: str
    vote: str          # "up" | "down"


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── Chat (SSE streaming) ───────────────────────────────────────────────────────
async def _sse_stream(question: str, history: list, document_ids: list) -> AsyncIterator[str]:
    """Format agent events as SSE data lines."""
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable nginx buffering
        },
    )


# ── Documents ──────────────────────────────────────────────────────────────────
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


# ── Upload (async via Celery) ──────────────────────────────────────────────────
def _write_atomic(target: Path, content: bytes, retries: int = 5) -> None:
    """Write to a temp file then replace the target.

    On Windows a freshly-written PDF may be transiently held by an antivirus
    scan, the search indexer, or the indexing worker (pypdf opens each file),
    which makes an in-place overwrite fail with PermissionError (sharing
    violation). Writing to a sibling temp file and os.replace()-ing avoids the
    truncate-in-place conflict; the retry loop covers the replace itself.
    """
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(content)
    for attempt in range(retries):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.3 * (attempt + 1))


@app.post("/upload")
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
):
    raw_dir = Path(ROOT) / RAW_PDF_DIR
    processed_dir = Path(ROOT) / PROCESSED_DIR
    vector_dir = Path(ROOT) / CHROMA_DIR

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        if file.size and file.size > 50 * 1024 * 1024:     # 50 MB limit
            raise HTTPException(status_code=413, detail=f"{file.filename} exceeds 50 MB limit.")
        content = await file.read()
        _write_atomic(raw_dir / file.filename, content)
        saved.append(file.filename)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid PDF files received.")

    task = index_pdfs_task.delay(
        str(raw_dir),
        str(processed_dir),
        str(vector_dir),
        saved,
    )

    job = IndexJob(id=task.id, status="pending", filenames=",".join(saved))
    session.add(job)
    await session.commit()

    return {"job_id": task.id, "files": saved, "message": f"{len(saved)} PDF(s) queued for indexing."}


# ── Upload status ──────────────────────────────────────────────────────────────
@app.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    """Poll this endpoint after /upload to track indexing progress."""
    from celery.result import AsyncResult
    result = AsyncResult(job_id, app=index_pdfs_task.app)

    state = result.state         # PENDING | STARTED | PROGRESS | SUCCESS | FAILURE
    meta = result.info or {}

    if state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(meta)}

    if state == "SUCCESS":
        return {"job_id": job_id, "status": "done", "files": meta.get("files", [])}

    step = meta.get("step", "") if isinstance(meta, dict) else ""
    return {"job_id": job_id, "status": state.lower(), "step": step}


# ── Feedback ───────────────────────────────────────────────────────────────────
@app.post("/feedback")
async def save_feedback(
    req: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    if req.vote not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="vote must be 'up' or 'down'.")

    row = Feedback(
        trace_id=req.trace_id,
        question=req.question,
        answer=req.answer,
        vote=req.vote,
    )
    session.add(row)
    await session.commit()
    return {"status": "saved"}
