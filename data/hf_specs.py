from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFDatasetSpec:
    dataset_id: str
    config: str | None
    image_field: str
    question_field: str
    answers_field: str
    id_field: str | None
    parquet_patterns: list[str]


HF_DATASETS: dict[str, HFDatasetSpec] = {
    "textvqa": HFDatasetSpec(
        dataset_id="lmms-lab/textvqa",
        config=None,
        image_field="image",
        question_field="question",
        answers_field="answers",
        id_field="question_id",
        parquet_patterns=["**/*.parquet"],
    ),
    "docvqa": HFDatasetSpec(
        dataset_id="pixparse/docvqa-single-page-questions",
        config=None,
        image_field="image",
        question_field="question",
        answers_field="answers",
        id_field=None,
        parquet_patterns=["**/*.parquet"],
    ),
}
