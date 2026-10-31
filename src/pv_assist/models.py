from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    point_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    document_name: str
    document_id: str
    chunks_indexed: int


class SearchRequest(BaseModel):
    question: str
    document_id: str | None = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    score: float | None = None


class AskRequest(BaseModel):
    question: str
    document_id: str | None = None


class AnswerSource(BaseModel):
    document_name: str
    document_id: str
    page_number: int
    passage: str
    source_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[AnswerSource]
    insufficient_context: bool = False
