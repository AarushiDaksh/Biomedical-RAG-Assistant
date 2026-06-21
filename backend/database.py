import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from rag.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    trace_id: Mapped[str] = mapped_column(String(16), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    vote: Mapped[str] = mapped_column(String(4))          # "up" | "down"
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class IndexJob(Base):
    __tablename__ = "index_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)   # Celery task id
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|processing|done|failed
    filenames: Mapped[str] = mapped_column(Text, default="")         # comma-separated
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
