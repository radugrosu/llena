#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import io
import json
import re
import typer
from typing import Literal

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

if __package__ is None and __name__ == "__main__":
    raise RuntimeError("Run as module: python -m scripts.process_dataset")

from data.hf_specs import HF_DATASETS


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _as_record(obj: object) -> dict[str, object]:
    if not isinstance(obj, dict):
        raise TypeError(f"Dataset record must be dict, got {type(obj).__name__}")
    out: dict[str, object] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise TypeError("Dataset record keys must be str.")
        out[k] = v
    return out


def _require_str(rec: dict[str, object], key: str) -> str:
    val = rec.get(key)
    if not isinstance(val, str):
        raise TypeError(f"Field '{key}' must be str.")
    return val


def _require_answers(rec: dict[str, object], key: str) -> list[str]:
    val = rec.get(key)
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise TypeError(f"Field '{key}' must be list[str].")
    return val


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _make_image_name(idx: int, rec: dict[str, object], id_field: str | None) -> str:
    suffix = ""
    if id_field is not None and id_field in rec:
        raw = rec.get(id_field)
        if isinstance(raw, int):
            suffix = f"_{raw}"
        elif isinstance(raw, str):
            suffix = f"_{_safe_id(raw)}"
    return f"{idx:08d}{suffix}.jpg"


def _to_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw_bytes = value.get("bytes")
        raw_path = value.get("path")
        if isinstance(raw_bytes, (bytes, bytearray)):
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if isinstance(raw_path, str):
            return Image.open(raw_path).convert("RGB")
    raise TypeError("Unsupported image value type.")


def _parquet_files(raw_dir: Path, split: str) -> list[str]:
    files = sorted(str(p) for p in raw_dir.rglob(f"{split}*.parquet"))
    if not files:
        files = sorted(str(p) for p in raw_dir.rglob("*.parquet"))
    return files


def main(
    dataset: str = opt(..., "Dataset: docvqa | textvqa"),
    split: Literal["train", "validation", "test"] = opt(
        "train", "Split name (train | validation | test)"
    ),
    raw_dir: str = opt("datasets/raw", "Input directory for parquet files"),
    out_dir: str = opt("datasets/processed", "Output directory for JSONL + images"),
    limit: int | None = opt(None, "Optional max number of samples"),
) -> None:
    dataset_l = dataset.lower()
    if dataset_l not in HF_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    spec = HF_DATASETS[dataset_l]

    raw_path = Path(raw_dir) / dataset_l
    files = _parquet_files(raw_path, split)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {raw_path}")

    ds = load_dataset("parquet", data_files={split: files}, split=split)

    out_root = Path(out_dir) / dataset_l
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / f"{split}.jsonl"

    total = len(ds)
    if limit is not None and limit > 0:
        total = min(total, limit)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, raw in tqdm(enumerate(ds), total=total, desc=f"{dataset_l}:{split}"):
            if limit is not None and limit > 0 and idx >= limit:
                break
            rec = _as_record(raw)
            image = _to_image(rec.get(spec.image_field))
            question = _require_str(rec, spec.question_field)
            answers = _require_answers(rec, spec.answers_field)
            answer = answers[0]

            image_name = _make_image_name(idx, rec, spec.id_field)
            image_path = images_dir / image_name
            if not image_path.exists():
                image.save(image_path, format="JPEG")

            line = {
                "image": image_name,
                "question": question,
                "answer": answer,
                "answers": answers,
            }
            f.write(json.dumps(line, ensure_ascii=True) + "\n")

    typer.echo(f"wrote {jsonl_path} and images under {images_dir}")


if __name__ == "__main__":
    typer.run(main)
