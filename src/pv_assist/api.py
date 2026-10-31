from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import settings
from .generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    MODEL_ABSTENTION_ANSWER,
    LLMProviderError,
    create_answer_service,
    validate_answer_citations,
)
from .models import AnswerSource, AskRequest, AskResponse, SearchRequest, UploadResponse
from .services import DocumentIndex, document_content_id, validate_pdf_bytes

app = FastAPI(title=settings.app_name, version="0.1.0")
document_index = DocumentIndex()
answer_service = create_answer_service()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.app_name}!"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name, "llm_provider": settings.llm_provider}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:  # noqa: B008
    if file is None or file.filename is None:
        raise HTTPException(status_code=400, detail="A PDF file is required.")

    file_bytes = await file.read()
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        validate_pdf_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        chunks = document_index.ingest_document(file.filename, file_bytes)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return UploadResponse(
        document_name=file.filename,
        document_id=document_content_id(file_bytes),
        chunks_indexed=len(chunks),
    )


@app.post("/search")
async def search_documents(payload: SearchRequest) -> dict[str, list[dict[str, object]]]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A search question is required.")

    try:
        results = document_index.search(question, document_id=payload.document_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"results": [result.model_dump() for result in results]}


@app.post("/ask", response_model=AskResponse)
async def ask_documents(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="A question is required.")

    try:
        results = document_index.search(question, document_id=payload.document_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    sources: list[AnswerSource] = []
    for result in results:
        if result.score is not None and result.score >= settings.min_retrieval_score:
            sources.append(
                AnswerSource(
                    document_name=result.document_name,
                    document_id=result.document_id,
                    page_number=result.page_number,
                    passage=result.text,
                    source_id=f"S{len(sources) + 1}",
                )
            )
    if not sources:
        return AskResponse(answer=INSUFFICIENT_CONTEXT_ANSWER, sources=[], insufficient_context=True)

    try:
        answer = answer_service.generate(question, sources)
        validate_answer_citations(answer, sources)
    except LLMProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Answer generation is unavailable.") from exc

    return AskResponse(
        answer=answer,
        sources=sources,
        insufficient_context=answer.strip().casefold() == MODEL_ABSTENTION_ANSWER.casefold(),
    )
