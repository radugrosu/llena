#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import typer

from data.format import (
    convert_docvqa_annotations,
    convert_textvqa_annotations,
    write_jsonl,
)


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def main(
    dataset: str = opt(..., "Dataset: docvqa | textvqa"),
    annotations: str = opt(..., "Path to raw annotations JSON"),
    images: str = opt(..., "Path to images root"),
    out: str = opt(..., "Output JSONL path"),
    split: str = opt("train", "Split name (used for TextVQA image template)"),
    image_template: str | None = opt(
        None, "Override image template for TextVQA (e.g. COCO_train2014_%012d.jpg)"
    ),
) -> None:
    annotations_path = Path(annotations)
    images_root = Path(images)
    out_path = Path(out)

    dataset_l = dataset.lower()
    if dataset_l == "textvqa":
        records = convert_textvqa_annotations(
            annotations_path=annotations_path,
            images_root=images_root,
            split=split,
            image_template=image_template,
        )
    elif dataset_l == "docvqa":
        records = convert_docvqa_annotations(
            annotations_path=annotations_path,
            images_root=images_root,
        )
    else:
        raise ValueError(f"dataset must be docvqa|textvqa, got: {dataset}")

    count = write_jsonl(records, out_path)
    typer.echo(f"wrote {count} records to {out_path}")


if __name__ == "__main__":
    typer.run(main)
