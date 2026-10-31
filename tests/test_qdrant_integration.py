from __future__ import annotations

import os
import uuid

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from pv_assist import api
from pv_assist.config import settings
from pv_assist.services import DocumentIndex, EmbeddingService, QdrantVectorStore


def _build_pdf_bytes(text: str, page_count: int = 1) -> bytes:
    document = fitz.open()
    for _ in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), text)
    return document.tobytes()


@pytest.mark.integration
def test_fastapi_upload_replacement_and_document_scoped_search(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant_url = os.getenv("PV_ASSIST_QDRANT_URL", settings.qdrant_url)
    client = QdrantClient(url=qdrant_url)
    collection_name = f"pv-assist-test-{uuid.uuid4().hex[:8]}"
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        vector_size=settings.embedding_dimension,
    )
    index = DocumentIndex(vector_store=vector_store, embedding_service=EmbeddingService())
    monkeypatch.setattr(api, "document_index", index)

    try:
        with TestClient(api.app) as test_client:
            original_pdf = _build_pdf_bytes(
                "INTEGRATION REVISION ONE. "
                + "The inverter isolation procedure requires opening the DC disconnect before inspection.",
                page_count=4,
            )
            upload_response = test_client.post(
                "/documents/upload",
                files={"file": ("integration-manual.pdf", original_pdf, "application/pdf")},
            )
            assert upload_response.status_code == 200, upload_response.text
            first_document = upload_response.json()
            assert len(first_document["document_id"]) == 64
            assert first_document["chunks_indexed"] > 1

            revised_pdf = _build_pdf_bytes(
                "INTEGRATION REVISION TWO. The battery arc fault reset procedure requires isolating the battery first."
            )
            revised_response = test_client.post(
                "/documents/upload",
                files={"file": ("integration-manual.pdf", revised_pdf, "application/pdf")},
            )
            assert revised_response.status_code == 200, revised_response.text
            revised_document = revised_response.json()
            assert revised_document["document_id"] != first_document["document_id"]

            collection = client.get_collection(collection_name)
            assert collection.config.params.vectors.size == index.embedding_service.dimension
            stored_points = client.count(collection_name=collection_name, exact=True).count
            assert stored_points == revised_document["chunks_indexed"]

            old_document_search = test_client.post(
                "/search",
                json={
                    "question": "How do I reset the battery arc fault?",
                    "document_id": first_document["document_id"],
                },
            )
            assert old_document_search.status_code == 200, old_document_search.text
            assert old_document_search.json()["results"] == []

            search_response = test_client.post(
                "/search",
                json={
                    "question": "How do I reset the battery arc fault?",
                    "document_id": revised_document["document_id"],
                },
            )
            assert search_response.status_code == 200, search_response.text
            results = search_response.json()["results"]
            assert results
            first_result = results[0]
            assert first_result["document_id"] == revised_document["document_id"]
            assert first_result["document_name"] == "integration-manual.pdf"
            assert first_result["page_number"] == 1
            assert "battery arc fault reset procedure" in first_result["text"].lower()
            assert first_result["score"] is not None
    finally:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
