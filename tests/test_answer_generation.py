from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from streamlit.testing.v1 import AppTest

from pv_assist import api, generation
from pv_assist.config import settings
from pv_assist.generation import (
    GROQ_CHAT_COMPLETIONS_URL,
    INSUFFICIENT_CONTEXT_ANSWER,
    MODEL_ABSTENTION_ANSWER,
    SYSTEM_INSTRUCTIONS,
    GroqAnswerService,
    LLMProviderError,
    MistralAnswerService,
    create_answer_service,
)
from pv_assist.models import AnswerSource, SearchResult


class FakeMistralClient:
    def __init__(self, answer: str = "Isolate the DC disconnect [manual.pdf, page 2].") -> None:
        self.answer = answer
        self.call: dict[str, object] | None = None
        self.chat = SimpleNamespace(complete=self.complete)

    def complete(self, **kwargs: object) -> SimpleNamespace:
        self.call = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))]
        )


def _source() -> AnswerSource:
    return AnswerSource(
        document_name="manual.pdf",
        document_id="a" * 64,
        page_number=2,
        passage="Before inspection, isolate the DC disconnect according to site procedure.",
        source_id="S1",
    )


def test_mistral_service_uses_only_retrieved_passages() -> None:
    client = FakeMistralClient()
    service = MistralAnswerService(client=client, model_name="test-model")

    answer = service.generate("What must be isolated?", [_source()])

    assert answer == client.answer
    assert client.call is not None
    assert client.call["model"] == "test-model"
    messages = client.call["messages"]
    assert messages[0]["role"] == "system"
    assert "Never use outside knowledge" in SYSTEM_INSTRUCTIONS
    assert "manual.pdf" in messages[1]["content"]
    assert "Before inspection, isolate the DC disconnect" in messages[1]["content"]
    assert "unrelated document content" not in messages[1]["content"]
    assert '"source_id": "S1"' in messages[1]["content"]


def test_groq_service_sends_grounded_request_and_parses_answer() -> None:
    calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> httpx.Response:
        calls.append((url, json, headers, timeout))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Isolate the DC disconnect [manual.pdf, page 2]."}}]},
        )

    service = GroqAnswerService(
        request=fake_post,
        model_name="openai/gpt-oss-20b",
        api_key=SecretStr("test-groq-key"),
    )

    answer = service.generate("What must be isolated?", [_source()])

    assert answer == "Isolate the DC disconnect [manual.pdf, page 2]."
    assert len(calls) == 1
    url, payload, headers, timeout = calls[0]
    assert url == GROQ_CHAT_COMPLETIONS_URL
    assert payload["model"] == "openai/gpt-oss-20b"
    assert "Before inspection, isolate the DC disconnect" in payload["messages"][1]["content"]
    assert headers["Authorization"] == "Bearer test-groq-key"
    assert timeout == 30.0


def test_groq_service_requires_key_only_when_generating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "groq_api_key", None)

    with pytest.raises(LLMProviderError, match="GROQ_API_KEY is not configured"):
        GroqAnswerService(request=lambda *args, **kwargs: pytest.fail("must not make a request")).generate(
            "Question?", [_source()]
        )


@pytest.mark.parametrize("provider, expected", [("groq", GroqAnswerService), ("mistral", MistralAnswerService)])
def test_create_answer_service_selects_configured_provider(
    provider: str,
    expected: type[GroqAnswerService | MistralAnswerService],
) -> None:
    assert isinstance(create_answer_service(provider), expected)


@pytest.mark.parametrize(
    "status_code, expected_message",
    [
        (401, "authentication failed"),
        (403, "denied access"),
        (429, "rate limit exceeded"),
        (503, "temporarily unavailable"),
    ],
)
def test_groq_http_errors_are_safe_and_not_retried(
    status_code: int,
    expected_message: str,
) -> None:
    calls = 0

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, json={"error": {"message": "secret response detail"}})

    service = GroqAnswerService(request=fake_post, api_key=SecretStr("test-key"))
    with pytest.raises(LLMProviderError) as error:
        service.generate("Question?", [_source()])

    assert error.value.status_code == status_code
    assert expected_message in error.value.detail
    assert "secret response detail" not in error.value.detail
    assert calls == 1


@pytest.mark.parametrize(
    "exception, status_code, expected_message",
    [
        (httpx.ReadTimeout("sensitive timeout detail"), 504, "timed out"),
        (httpx.ConnectError("sensitive connection detail"), 503, "connection error"),
    ],
)
def test_groq_transport_errors_are_safe_and_not_retried(
    exception: Exception,
    status_code: int,
    expected_message: str,
) -> None:
    calls = 0

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exception

    service = GroqAnswerService(request=fake_post, api_key=SecretStr("test-key"))
    with pytest.raises(LLMProviderError) as error:
        service.generate("Question?", [_source()])

    assert error.value.status_code == status_code
    assert expected_message in error.value.detail
    assert "sensitive" not in error.value.detail
    assert calls == 1


def test_mistral_rate_limit_error_is_safe_and_not_retried() -> None:
    class RateLimitError(Exception):
        status_code = 429

    class RateLimitedMistralClient(FakeMistralClient):
        def complete(self, **kwargs: object) -> SimpleNamespace:
            self.call = kwargs
            raise RateLimitError("private provider response")

    client = RateLimitedMistralClient()
    service = MistralAnswerService(client=client)

    with pytest.raises(LLMProviderError) as error:
        service.generate("Question?", [_source()])

    assert error.value.status_code == 429
    assert "rate limit exceeded" in error.value.detail
    assert "private provider response" not in error.value.detail


def test_mistral_client_is_configured_without_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeSDKClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                complete=lambda **request: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="grounded answer"))]
                )
            )

    monkeypatch.setattr(settings, "mistral_api_key", SecretStr("test-mistral-key"))
    monkeypatch.setattr(generation, "Mistral", FakeSDKClient)
    service = MistralAnswerService()

    assert service.generate("Question?", [_source()]) == "grounded answer"
    assert captured["retry_config"].strategy == "none"
    assert captured["retry_config"].retry_connection_errors is False


def test_mistral_service_returns_insufficient_context_without_calling_client() -> None:
    client = FakeMistralClient()
    service = MistralAnswerService(client=client)

    answer = service.generate("What must be isolated?", [])

    assert answer == INSUFFICIENT_CONTEXT_ANSWER
    assert client.call is None


def test_mistral_service_requires_key_only_when_generating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "mistral_api_key", None)
    service = MistralAnswerService()

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY is not configured"):
        service.generate("What must be isolated?", [_source()])


class StubDocumentIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.search_call: tuple[str, str | None] | None = None

    def search(self, question: str, document_id: str | None = None) -> list[SearchResult]:
        self.search_call = (question, document_id)
        return self.results


class StubAnswerService:
    def __init__(self) -> None:
        self.sources: list[AnswerSource] | None = None

    def generate(self, question: str, sources: list[AnswerSource]) -> str:
        self.sources = sources
        return "Isolate the DC disconnect [S1]."


def _search_result(score: float) -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        document_id="a" * 64,
        document_name="manual.pdf",
        page_number=2,
        text="Before inspection, isolate the DC disconnect according to site procedure.",
        score=score,
    )


def test_ask_endpoint_returns_grounded_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    index = StubDocumentIndex([_search_result(score=0.9)])
    answer_service = StubAnswerService()
    monkeypatch.setattr(api, "document_index", index)
    monkeypatch.setattr(api, "answer_service", answer_service)

    with TestClient(api.app) as client:
        response = client.post(
            "/ask",
            json={"question": "What must be isolated?", "document_id": "a" * 64},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Isolate the DC disconnect [S1]."
    assert payload["insufficient_context"] is False
    assert payload["sources"] == [
        {
            "document_name": "manual.pdf",
            "document_id": "a" * 64,
            "page_number": 2,
            "passage": "Before inspection, isolate the DC disconnect according to site procedure.",
            "source_id": "S1",
        }
    ]
    assert index.search_call == ("What must be isolated?", "a" * 64)
    assert answer_service.sources is not None
    assert answer_service.sources[0].source_id == "S1"


@pytest.mark.parametrize("results", [[], [_search_result(score=0.1)]])
def test_ask_endpoint_skips_generation_without_relevant_sources(
    monkeypatch: pytest.MonkeyPatch,
    results: list[SearchResult],
) -> None:
    index = StubDocumentIndex(results)
    answer_service = StubAnswerService()
    monkeypatch.setattr(api, "document_index", index)
    monkeypatch.setattr(api, "answer_service", answer_service)

    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "Unrelated question?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert payload["sources"] == []
    assert payload["insufficient_context"] is True
    assert answer_service.sources is None


def test_ask_endpoint_preserves_provider_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "document_index", StubDocumentIndex([_search_result(score=0.9)]))

    class RateLimitedAnswerService:
        def generate(self, question: str, sources: list[AnswerSource]) -> str:
            raise LLMProviderError("Groq", 429, "Groq rate limit exceeded (HTTP 429). No retry was attempted.")

    monkeypatch.setattr(api, "answer_service", RateLimitedAnswerService())
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "Question?"})

    assert response.status_code == 429
    assert "No retry was attempted" in response.json()["detail"]


def test_streamlit_requires_index_and_clears_previous_document_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_get(*args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json={"llm_provider": "groq"})

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        calls.append(url)
        if url.endswith("/documents/upload"):
            return httpx.Response(
                200,
                json={"document_name": "pv_maintenance_smoke.pdf", "document_id": "b" * 64, "chunks_indexed": 3},
            )
        pytest.fail("The UI must not ask automatically during reruns.")

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    repository_root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(repository_root / "src/pv_assist/ui.py").run()
    submit_button = app.get("form_submit_button")[0]
    assert submit_button.proto.disabled is True
    assert any("Answer provider: Groq" in caption.value for caption in app.caption)

    pdf_path = repository_root / "tests/fixtures/pv_maintenance_smoke.pdf"
    app.file_uploader[0].upload(pdf_path.name, pdf_path.read_bytes(), "application/pdf").run()
    assert app.get("form_submit_button")[0].proto.disabled is True

    app.session_state["answer_document_scope"] = "previous-document"
    app.session_state["last_ask_result"] = {"answer": "stale answer", "sources": [], "insufficient_context": False}
    app.run()
    assert "last_ask_result" not in app.session_state

    app.button[0].click().run()
    assert app.get("form_submit_button")[0].proto.disabled is False
    assert calls == ["http://localhost:8001/documents/upload"]

    app.session_state["last_ask_result"] = {
        "answer": "Follow the documented procedure [S1].",
        "sources": [
            {
                "source_id": "S1",
                "document_name": "pv_maintenance_smoke.pdf",
                "document_id": "b" * 64,
                "page_number": 2,
                "passage": "Use clean water and a soft, non-abrasive cloth.",
            }
        ],
        "insufficient_context": False,
    }
    app.run()
    assert any("Follow the documented procedure [pv_maintenance_smoke.pdf, page 2]." in item.value for item in app.markdown)
    assert any("does not verify factual support" in item.value for item in app.caption)


@pytest.mark.parametrize("answer", ["A claim without citations.", "The answer is [S99]."])
def test_ask_endpoint_withholds_missing_or_unknown_citations(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    monkeypatch.setattr(api, "document_index", StubDocumentIndex([_search_result(score=0.99)]))

    class InvalidAnswerService:
        def generate(self, question: str, sources: list[AnswerSource]) -> str:
            return answer

    monkeypatch.setattr(api, "answer_service", InvalidAnswerService())
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "Question?"})

    assert response.status_code == 502
    assert "was withheld" in response.json()["detail"]


def test_ask_endpoint_accepts_explicit_abstention_despite_high_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "document_index", StubDocumentIndex([_search_result(score=0.99)]))

    class AbstainingAnswerService:
        def generate(self, question: str, sources: list[AnswerSource]) -> str:
            assert sources[0].source_id == "S1"
            return MODEL_ABSTENTION_ANSWER

    monkeypatch.setattr(api, "answer_service", AbstainingAnswerService())
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "What pressure limit applies?"})

    assert response.status_code == 200
    assert response.json()["answer"] == MODEL_ABSTENTION_ANSWER
    assert response.json()["insufficient_context"] is True