from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pv_assist.config import settings
from pv_assist.generation import (
    MODEL_ABSTENTION_ANSWER,
    LLMProviderError,
    validate_answer_citations,
)
from pv_assist.models import AnswerSource
from pv_assist.services import DocumentIndex

DATASET_PATH = PROJECT_ROOT / "evaluation" / "rag_questions.json"


def load_answers(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(value, str) for value in data.values()):
        raise ValueError("Answers file must be a JSON object mapping question IDs to answer strings.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local retrieval and optional recorded citation structure; never calls an LLM."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve per question.")
    parser.add_argument(
        "--answers-json",
        type=Path,
        help="Optional JSON object mapping Q01-style IDs to recorded answer strings for citation checks.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    dataset: dict[str, Any] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    manual_path = PROJECT_ROOT / dataset["manual"]
    document_id = hashlib.sha256(manual_path.read_bytes()).hexdigest()
    answers = load_answers(args.answers_json)

    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(settings.qdrant_collection):
        print(
            f"Qdrant collection '{settings.qdrant_collection}' does not exist. "
            "Upload the evaluation PDF before running this script.",
            file=sys.stderr,
        )
        return 2

    index = DocumentIndex()
    result_rows: list[dict[str, Any]] = []
    answerable_count = 0
    page_hits = 0
    citation_valid_count = 0
    citation_checked_count = 0
    expected_abstentions = [
        item for item in dataset["questions"] if item["expected_abstention"]
    ]
    for question_item in dataset["questions"]:
        results = index.search(
            question_item["question"],
            limit=args.top_k,
            document_id=document_id,
        )
        retrieved_pages = list(dict.fromkeys(result.page_number for result in results))
        expected_pages = question_item["expected_source_pages"]
        page_hit = bool(set(expected_pages).intersection(retrieved_pages)) if expected_pages else None
        if expected_pages:
            answerable_count += 1
            page_hits += bool(page_hit)

        sources = [
            AnswerSource(
                document_name=result.document_name,
                document_id=result.document_id,
                page_number=result.page_number,
                passage=result.text,
                source_id=f"S{index_number}",
            )
            for index_number, result in enumerate(
                (
                    result
                    for result in results
                    if result.score is not None and result.score >= settings.min_retrieval_score
                ),
                start=1,
            )
        ]
        recorded_answer = answers.get(question_item["id"])
        citation_status = "not_provided"
        cited_ids: list[str] = []
        abstention_match: bool | None = None
        if recorded_answer is not None:
            citation_checked_count += 1
            abstention_match = recorded_answer.strip().casefold() == MODEL_ABSTENTION_ANSWER.casefold()
            try:
                cited_ids = validate_answer_citations(recorded_answer, sources)
                citation_status = "valid_source_identifiers"
                citation_valid_count += 1
            except LLMProviderError as exc:
                citation_status = exc.detail

        result_rows.append(
            {
                "id": question_item["id"],
                "expected_source_pages": expected_pages,
                "retrieved_source_pages": retrieved_pages,
                "expected_abstention": question_item["expected_abstention"],
                "safety_sensitive": question_item["safety_sensitive"],
                "expected_page_retrieved": page_hit,
                "citation_status": citation_status,
                "cited_source_ids": cited_ids,
                "recorded_answer_matches_abstention": abstention_match,
            }
        )

    page_hit_rate = page_hits / answerable_count if answerable_count else None
    report = {
        "evaluation_set_size": len(result_rows),
        "document_id": document_id,
        "top_k": args.top_k,
        "retrieval_metrics": {
            "page_hit_rate_at_k": page_hit_rate,
            "k": args.top_k,
            "expected_page_hits": page_hits,
            "answerable_questions": answerable_count,
            "interpretation": (
                "Fraction of answerable questions with at least one expected page among the top-k retrieved pages; "
                "this is a retrieval metric, not answer accuracy."
            ),
        },
        "answer_generation_metrics": {
            "provider_calls_made": 0,
            "recorded_answers_evaluated": citation_checked_count,
            "citation_correctness": "not measured; source-ID validity is only a structural check",
            "citation_identifier_validity": (
                "not evaluated because no recorded answers were provided"
                if citation_checked_count == 0
                else "structural source-ID membership only; not semantic support"
            ),
            "semantic_factual_correctness": "not measured; requires human review",
            "semantic_abstention_correctness": "not measured; requires human review",
            "expected_abstention_count": len(expected_abstentions),
            "expected_abstention_questions": [
                {"id": item["id"], "question": item["question"]}
                for item in expected_abstentions
            ],
        },
        "retrieval_page_hit_rate": page_hit_rate,
        "answerable_questions": answerable_count,
        "page_hits": page_hits,
        "citation_checks": citation_checked_count,
        "valid_citation_identifier_rate": (
            citation_valid_count / citation_checked_count if citation_checked_count else None
        ),
        "provider_calls_made": 0,
        "semantic_correctness": "not measured; structural checks do not prove entailment",
        "results": result_rows,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())