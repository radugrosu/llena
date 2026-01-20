from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset

from mm.types import VQASample


@dataclass(frozen=True)
class VQARecord:
    image_path: str
    question: str
    answer: str
    answers: list[str] | None = None


def _parse_record(obj: object) -> VQARecord:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict record, got {type(obj).__name__}")

    image_val = obj.get("image") if "image" in obj else obj.get("image_path")
    if not isinstance(image_val, str):
        raise TypeError("Record must include 'image' or 'image_path' as str.")

    question = obj.get("question")
    answer = obj.get("answer")
    answers_obj = obj.get("answers")
    answers: list[str] | None = None
    if answers_obj is not None:
        if not isinstance(answers_obj, list) or not all(isinstance(x, str) for x in answers_obj):
            raise TypeError("Record 'answers' must be list[str] when provided.")
        answers = answers_obj
    if answer is None and answers:
        answer = _first_str(answers, label="Record answers")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise TypeError("Record must include 'question' and 'answer' as str.")

    return VQARecord(image_path=image_val, question=question, answer=answer, answers=answers)


def _load_jsonl(path: Path) -> list[VQARecord]:
    records: list[VQARecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(_parse_record(obj))
    return records


def resolve_dataset_paths(data_dir: Path, dataset: str, split: str) -> tuple[Path, Path]:
    annotations = data_dir / dataset / f"{split}.jsonl"
    image_root = data_dir / dataset / "images"
    return annotations, image_root


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_list(obj: object, *, label: str) -> list[object]:
    if not isinstance(obj, list):
        raise TypeError(f"{label} must be list, got {type(obj).__name__}")
    return obj


def _first_str(values: Iterable[object], *, label: str) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError(f"{label} has no non-empty string values.")


def _image_relpath(image_path: str, images_root: Path) -> str:
    try:
        rel = Path(image_path).relative_to(images_root)
        return str(rel)
    except ValueError:
        return image_path


def convert_textvqa_annotations(
    *,
    annotations_path: Path,
    images_root: Path,
    split: str,
    image_template: str | None = None,
) -> list[VQARecord]:
    """
    Expected TextVQA fields:
      - question: str
      - answers: list[str]
      - image_id: int
    """
    raw = _load_json(annotations_path)
    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    items = _ensure_list(data, label="TextVQA annotations")

    template = image_template or f"COCO_{split}2014_%012d.jpg"
    records: list[VQARecord] = []

    for item in items:
        if not isinstance(item, dict):
            raise TypeError("TextVQA item must be dict.")
        question = item.get("question")
        answers = item.get("answers")
        image_id = item.get("image_id")
        if not isinstance(question, str):
            raise TypeError("TextVQA item missing 'question' str.")
        if not isinstance(answers, list):
            raise TypeError("TextVQA item missing 'answers' list.")
        if not isinstance(image_id, int):
            raise TypeError("TextVQA item missing 'image_id' int.")
        answer = _first_str(answers, label="TextVQA answers")
        image_name = template % image_id
        image_path = str(Path(image_name))
        image_path = _image_relpath(image_path, images_root)
        records.append(
            VQARecord(
                image_path=image_path,
                question=question,
                answer=answer,
                answers=answers,
            )
        )

    return records


def convert_docvqa_annotations(
    *,
    annotations_path: Path,
    images_root: Path,
) -> list[VQARecord]:
    """
    Expected DocVQA fields (common variants):
      - image or image_path: str
      - question: str
      - answers: list[str]
    """
    raw = _load_json(annotations_path)
    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    items = _ensure_list(data, label="DocVQA annotations")

    records: list[VQARecord] = []
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("DocVQA item must be dict.")
        question = item.get("question")
        answers = item.get("answers")
        image_val = item.get("image") if "image" in item else item.get("image_path")
        if not isinstance(question, str):
            raise TypeError("DocVQA item missing 'question' str.")
        if not isinstance(answers, list):
            raise TypeError("DocVQA item missing 'answers' list.")
        if not isinstance(image_val, str):
            raise TypeError("DocVQA item missing 'image' str.")
        answer = _first_str(answers, label="DocVQA answers")
        image_path = _image_relpath(image_val, images_root)
        records.append(
            VQARecord(
                image_path=image_path,
                question=question,
                answer=answer,
                answers=answers,
            )
        )

    return records


def write_jsonl(records: Iterable[VQARecord], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            line: dict[str, object] = {
                "image": rec.image_path,
                "question": rec.question,
                "answer": rec.answer,
            }
            if rec.answers is not None:
                line["answers"] = rec.answers
            f.write(json.dumps(line, ensure_ascii=True) + "\n")
            count += 1
    return count


class JsonlVQADataset(Dataset):
    def __init__(
        self,
        *,
        annotations_path: Path,
        image_root: Path,
        max_samples: int | None = None,
    ):
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        if not image_root.exists():
            raise FileNotFoundError(f"Image root not found: {image_root}")

        records = _load_jsonl(annotations_path)
        if max_samples is not None and max_samples > 0:
            records = records[:max_samples]

        self.records = records
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> VQASample:
        rec = self.records[idx]
        img_path = self.image_root / rec.image_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        with Image.open(img_path) as im:
            img = im.convert("RGB")

        sample: VQASample = {
            "image": img,
            "question": rec.question,
            "answer": rec.answer,
        }
        if rec.answers is not None:
            sample["answers"] = rec.answers
        return sample


def load_vqa_jsonl_dataset(
    *,
    dataset: str,
    data_dir: Path,
    split: str,
    max_samples: int | None,
) -> JsonlVQADataset:
    annotations, image_root = resolve_dataset_paths(data_dir, dataset, split)
    return JsonlVQADataset(
        annotations_path=annotations,
        image_root=image_root,
        max_samples=max_samples,
    )
