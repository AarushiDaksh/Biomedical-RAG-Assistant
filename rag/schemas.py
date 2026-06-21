from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str
    title: str = "Unknown title"
    page_number: int
    chunk_id: str


class RAGAnswer(BaseModel):
    answer: str = Field(description="Grounded answer based only on retrieved context")
    citations: List[Citation] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    refusal_reason: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    page_number: int
    text: str
    score: float = 0.0
    source_url: str = ""


class GuardrailResult(BaseModel):
    allowed: bool
    reason: str = ""
    category: Literal["ok", "clinical_advice", "prompt_injection", "off_topic", "empty"] = "ok"
