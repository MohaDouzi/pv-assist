from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Sequence
from typing import Any

import pymupdf as fitz
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import settings
from .models import DocumentChunk, SearchResult


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: Any | None = None
        self.dimension = settings.embedding_dimension

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
            sample = next(self._model.embed(["sample text"]))
            self.dimension = len(sample)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            embeddings = list(self._get_model().embed(list(texts)))
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed for model '{self.model_name}'.") from exc
        return [list(vector) for vector in embeddings]

    def embed_single(self, text: str) -> list[float]:
        embedding = self.embed([text])
        return embedding[0]


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size or settings.embedding_dimension
        self.client = client or QdrantClient(url=settings.qdrant_url)

    def ensure_collection(self) -> None:
        try:
            if self.client.collection_exists(self.collection_name):
                collection = self.client.get_collection(self.collection_name)
                configured_vectors = collection.config.params.vectors
                collection_size = getattr(configured_vectors, "size", None)
                if collection_size != self.vector_size:
                    raise RuntimeError(
                        f"Qdrant collection '{self.collection_name}' uses vector size "
                        f"{collection_size}; the embedding model produces {self.vector_size}."
                    )
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant is unavailable while checking collection '{self.collection_name}'."
            ) from exc

    def upsert_points(self, points: Sequence[dict[str, Any]]) -> None:
        if not points:
            return
        self.ensure_collection()
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            raise RuntimeError(f"Qdrant failed to index collection '{self.collection_name}'.") from exc

    def replace_document(
        self,
        document_name: str,
        points: Sequence[dict[str, Any]],
    ) -> None:
        self.ensure_collection()
        try:
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
            point_ids = [point["id"] for point in points]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_name",
                                match=qmodels.MatchValue(value=document_name),
                            )
                        ],
                        must_not=[qmodels.HasIdCondition(has_id=point_ids)] if point_ids else None,
                    )
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Qdrant failed to replace document '{document_name}'.") from exc

    def search(
        self,
        query_vector: Sequence[float],
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_collection()
        try:
            query_filter = None
            if document_id is not None:
                query_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Qdrant search failed for collection '{self.collection_name}'.") from exc
        return [point.model_dump() if hasattr(point, "model_dump") else dict(point) for point in hits.points]


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "document"


def document_content_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def validate_pdf_bytes(file_bytes: bytes, max_size_bytes: int | None = None) -> None:
    max_allowed = max_size_bytes if max_size_bytes is not None else settings.max_upload_size_bytes
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(file_bytes) > max_allowed:
        raise ValueError(f"Uploaded file exceeds the {max_allowed} byte limit.")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover - PyMuPDF validation path
        raise ValueError("Uploaded file is not a valid PDF document.") from exc

    try:
        if doc.page_count == 0:
            raise ValueError("Uploaded PDF contains no pages.")
    finally:
        doc.close()


def extract_text_from_pdf(file_bytes: bytes) -> list[dict[str, Any]]:
    validate_pdf_bytes(file_bytes)
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[dict[str, Any]] = []
    try:
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            text = page.get_text("text", sort=True)
            pages.append({"page_number": page_number + 1, "text": text.strip()})
    finally:
        doc.close()
    return pages


def split_text_into_chunks(
    document_name: str,
    pages: Sequence[dict[str, Any]],
    chunk_size: int = 700,
    chunk_overlap: int = 150,
    document_id: str | None = None,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    content_id = document_id or hashlib.sha256(
        "\n".join(str(page["text"]) for page in pages).encode("utf-8")
    ).hexdigest()
    chunks: list[DocumentChunk] = []
    for page in pages:
        text = " ".join(page["text"].split())
        if not text:
            continue
        page_number = int(page["page_number"])
        start = 0
        chunk_index = 1
        while start < len(text):
            end = min(len(text), start + chunk_size)
            window = text[start:end].strip()
            if not window:
                break
            chunk_id = f"{content_id[:12]}-p{page_number}-c{chunk_index:04d}"
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{content_id}:{chunk_id}"))
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    point_id=point_uuid,
                    document_id=content_id,
                    document_name=document_name,
                    page_number=page_number,
                    text=window,
                    metadata={
                        "document_id": content_id,
                        "document_name": document_name,
                        "page_number": page_number,
                        "chunk_id": chunk_id,
                    },
                )
            )
            if end >= len(text):
                break
            start = max(start + chunk_size - chunk_overlap, start + 1)
            chunk_index += 1
    return chunks


class DocumentIndex:
    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()

    def ingest_document(self, document_name: str, file_bytes: bytes) -> list[DocumentChunk]:
        document_id = document_content_id(file_bytes)
        pages = extract_text_from_pdf(file_bytes)
        chunks = split_text_into_chunks(
            document_name,
            pages,
            settings.chunk_size,
            settings.chunk_overlap,
            document_id=document_id,
        )

        vectors: list[list[float]] = []
        if chunks:
            try:
                vectors = self.embedding_service.embed([chunk.text for chunk in chunks])
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to generate embeddings for '{document_name}'.") from exc

        try:
            points = [
                {
                    "id": chunk.point_id,
                    "vector": vector,
                    "payload": {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                    },
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            if vectors:
                self.vector_store.vector_size = len(vectors[0])
            self.vector_store.replace_document(document_name, points)
        except Exception as exc:  # pragma: no cover - content of runtime failure path
            raise RuntimeError(f"Failed to index '{document_name}' in Qdrant.") from exc
        return chunks

    def search(
        self,
        question: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedding_service.embed_single(question)
        self.vector_store.vector_size = len(query_vector)
        try:
            hits = self.vector_store.search(query_vector, limit=limit, document_id=document_id)
        except RuntimeError as exc:
            raise RuntimeError("Qdrant retrieval failed while searching for the answer.") from exc

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", "")),
                    document_id=str(payload.get("document_id", "")),
                    document_name=str(payload.get("document_name", "")),
                    page_number=int(payload.get("page_number", 0) or 0),
                    text=str(payload.get("text", "")),
                    score=float(hit.get("score", 0.0) or 0.0),
                )
            )
        return results
