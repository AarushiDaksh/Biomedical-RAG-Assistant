"""
Celery worker tasks.

Start the worker with:
    celery -A backend.tasks worker --loglevel=info --pool=solo

On Windows the default "prefork" pool is broken (billiard raises
"ValueError: not enough values to unpack (expected 3, got 0)" on every task),
so this app pins the pool to "solo" below. The --pool flag above is optional
once worker_pool is set, but kept for clarity.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from celery import Celery
from celery.utils.log import get_task_logger

from rag.config import REDIS_URL

celery_app = Celery(
    "biomedical_rag",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    # Windows: prefork is unreliable; in containers and Linux we can use prefork.
    worker_pool=os.getenv("CELERY_WORKER_POOL", "solo" if os.name == "nt" else "prefork"),
)

logger = get_task_logger(__name__)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
def index_pdfs_task(self, raw_dir: str, processed_dir: str, vector_dir: str, filenames: list[str]):
    """
    Full ingestion pipeline: ingest → chunk → vectorize.
    State transitions: STARTED → PROGRESS (step) → SUCCESS | FAILURE
    """
    from rag.ingest import ingest_folder
    from rag.chunking import recursive_chunks
    from rag.vectorstore import build_chroma

    try:
        logger.info("Starting ingestion for %d file(s): %s", len(filenames), filenames)

        self.update_state(state="PROGRESS", meta={"step": "ingesting", "files": filenames})
        ingest_folder(raw_dir, str(Path(processed_dir) / "documents.jsonl"))

        self.update_state(state="PROGRESS", meta={"step": "chunking", "files": filenames})
        recursive_chunks(
            str(Path(processed_dir) / "documents.jsonl"),
            str(Path(processed_dir) / "chunks.jsonl"),
        )

        self.update_state(state="PROGRESS", meta={"step": "vectorizing", "files": filenames})
        build_chroma(
            str(Path(processed_dir) / "chunks.jsonl"),
            vector_dir,
        )

        logger.info("Indexing complete for %s", filenames)
        return {"status": "done", "files": filenames}

    except Exception as exc:
        logger.error("Indexing failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)
