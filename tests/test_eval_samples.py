import json
from pathlib import Path

from PIL import Image

from scripts.eval import load_eval_samples_spec, resolve_eval_samples


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    img.save(path)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def _setup_dataset(
    base: Path,
    *,
    dataset: str,
    split: str,
    records: list[dict[str, object]],
) -> None:
    annotations = base / dataset / f"{split}.jsonl"
    images_root = base / dataset / "images"
    _write_jsonl(annotations, records)
    for rec in records:
        img_name = rec.get("image") or rec.get("image_path")
        if isinstance(img_name, str):
            _write_image(images_root / img_name)


def test_load_eval_samples_spec_defaults(tmp_path: Path) -> None:
    spec_path = tmp_path / "samples.json"
    payload = {"samples": ["img1.jpg", {"dataset": "docvqa", "split": "validation", "index": 0}]}
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    refs = load_eval_samples_spec(
        spec_path,
        default_dataset="textvqa",
        default_split="validation",
    )

    assert refs[0]["dataset"] == "textvqa"
    assert refs[0]["split"] == "validation"
    assert refs[0]["image_path"] == "img1.jpg"
    assert refs[1]["dataset"] == "docvqa"
    assert refs[1]["split"] == "validation"
    assert refs[1]["index"] == 0


def test_resolve_eval_samples_by_index_and_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _setup_dataset(
        data_dir,
        dataset="textvqa",
        split="validation",
        records=[
            {"image": "img1.jpg", "question": "q1", "answer": "a1"},
            {"image": "img2.jpg", "question": "q2", "answer": "a2"},
        ],
    )
    _setup_dataset(
        data_dir,
        dataset="docvqa",
        split="validation",
        records=[
            {"image": "doc1.jpg", "question": "doc q1", "answer": "doc a1"},
        ],
    )

    refs = [
        {"dataset": "textvqa", "split": "validation", "index": 1},
        {"dataset": "docvqa", "split": "validation", "image_path": "doc1.jpg", "question": "doc q1"},
    ]
    samples = resolve_eval_samples(refs, data_dir=data_dir)

    assert samples[0]["question"] == "q2"
    assert samples[0]["dataset"] == "textvqa"
    assert samples[0]["split"] == "validation"
    assert samples[0]["image_path"] == "img2.jpg"
    assert samples[1]["question"] == "doc q1"
    assert samples[1]["dataset"] == "docvqa"
    assert samples[1]["split"] == "validation"
    assert samples[1]["image_path"] == "doc1.jpg"
