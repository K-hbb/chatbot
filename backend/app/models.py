from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ===== Requests =====

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    conversation_id: Optional[str] = Field(
        default=None, description="Optional conversation thread id"
    )

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


# ===== Shared =====

class Source(BaseModel):
    id: str = Field(..., description="Internal document id (e.g., Chroma id)")
    score: float = Field(..., ge=0.0, description="Similarity score (0..1-ish)")
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ===== Responses =====

class ChatResponse(BaseModel):
    text: str
    sources: List[Source] = Field(default_factory=list)
    emergency: bool = False
    disclaimer: Optional[str] = None

class SearchHit(BaseModel):
    doc_id: str
    title: Optional[str] = None
    snippet: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    hits: List[SearchHit] = Field(default_factory=list)
