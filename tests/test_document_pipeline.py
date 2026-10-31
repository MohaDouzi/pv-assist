from __future__ import annotations

import pymupdf as fitz
import pytest

from pv_assist.models import SearchResult
from pv_assist.services import (
    DocumentIndex,
    QdrantVectorStore,
    extract_text_from_pdf,
    split_text_into_chunks,
)


def _build_pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "This is the first page of the maintenance manual. It explains inverter diagnostics and safety checks.",
    )
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "This is the second page of the maintenance manual. It describes panel cleaning and thermal inspection steps.",
    )
    return document.tobytes()


def test_extract_text_from_pdf() -> None:
    extracted = extract_text_from_pdf(_build_pdf_bytes())

    assert len(extracted) == 2
    assert extracted[0]["page_number"] == 1
    assert "maintenance manual" in extracted[0]["text"].lower()
    assert "panel cleaning" in extracted[1]["text"].lower()


def test_split_text_into_chunks_has_overlap() -> None:
    text = " ".join([f"Sentence {index} " for index in range(1, 200)])
    chunks = split_text_into_chunks(
        "sample-manual.pdf",
        [{"page_number": 1, "text": text}],
        chunk_size=250,
        chunk_overlap=50,
    )

    assert len(chunks) > 1
    assert chunks[0].page_number == 1
    assert chunks[0].document_name == "sample-manual.pdf"
    assert chunks[0].text[-50:] == chunks[1].text[:50]


def test_split_text_into_chunks_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        split_text_into_chunks(
            "sample-manual.pdf",
            [{"page_number": 1, "text": "Some manual text."}],
            chunk_overlap=-1,
        )


def test_qdrant_vector_store_rejects_collection_dimension_mismatch() -> None:
    class FakeQdrantClient:
        def collection_exists(self, collection_name: str) -> bool:
            return True

        def get_collection(self, collection_name: str) -> object:
            from types import SimpleNamespace

            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=SimpleNamespace(size=2))
                )
            )

    vector_store = QdrantVectorStore(client=FakeQdrantClient(), vector_size=3)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="uses vector size 2"):
        vector_store.ensure_collection()


def test_qdrant_vector_store_wraps_connection_errors() -> None:
    class BrokenQdrantClient:
        def collection_exists(self, collection_name: str) -> bool:
            raise OSError("connection refused")

    vector_store = QdrantVectorStore(client=BrokenQdrantClient())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="Qdrant is unavailable"):
        vector_store.ensure_collection()


def test_document_index_search_returns_results() -> None:
    class FakeEmbeddingService:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

        def embed_single(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    class FakeVectorStore:
        def __init__(self) -> None:
            self.saved_points: list[dict] = []

        def upsert_points(self, points: list[dict]) -> None:
            self.saved_points.extend(points)

        def replace_document(self, document_name: str, points: list[dict]) -> None:
            self.saved_points = [
                point
                for point in self.saved_points
                if point["payload"]["document_name"] != document_name
            ]
            self.saved_points.extend(points)

        def search(
            self,
            query_vector: list[float],
            limit: int = 5,
            document_id: str | None = None,
        ) -> list[dict]:
            if not self.saved_points:
                return []
            return [
                {
                    "score": 0.91,
                    "payload": {
                        "chunk_id": self.saved_points[0]["payload"]["chunk_id"],
                        "document_id": self.saved_points[0]["payload"]["document_id"],
                        "document_name": self.saved_points[0]["payload"]["document_name"],
                        "page_number": self.saved_points[0]["payload"]["page_number"],
                        "text": self.saved_points[0]["payload"]["text"],
                    },
                }
            ]

    vector_store = FakeVectorStore()
    index = DocumentIndex(vector_store=vector_store, embedding_service=FakeEmbeddingService())
    pdf_bytes = _build_pdf_bytes()

    chunks = index.ingest_document("maintenance-manual.pdf", pdf_bytes)

    assert len(chunks) > 0
    results = index.search("How do I inspect an inverter?")

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].document_name == "maintenance-manual.pdf"
    assert results[0].page_number >= 1
