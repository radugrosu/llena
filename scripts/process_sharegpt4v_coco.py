#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import typer
from tqdm import tqdm


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def extract_caption(entry: dict[str, object]) -> str:
    if "caption" in entry and isinstance(entry["caption"], str):
        return entry["caption"]
    if "text" in entry and isinstance(entry["text"], str):
        return entry["text"]
    conv = entry.get("conversations")
    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = msg.get("from")
            content = msg.get("value")
            if role in {"gpt", "assistant"} and isinstance(content, str):
                return content
    raise ValueError("No caption found in entry.")


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel)


def main(
    input_json: str = opt(
        "datasets/raw/sharegpt4v_coco/sharegpt4v_coco.json",
        "Filtered ShareGPT4V COCO metadata JSON",
    ),
    out_dir: str = opt(
        "datasets/processed/sharegpt4v_coco",
        "Output directory for JSONL",
    ),
    prompt: str = opt("Describe the image.", "Prompt to pair with caption"),
    limit: int | None = opt(None, "Optional cap on number of samples"),
    verify_images: bool = opt(True, "Check that image files exist"),
    symlink_images: bool = opt(True, "Symlink images into processed/images"),
) -> None:
    in_path = Path(input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")
    data = json.loads(in_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    out_path = out_root / "train.jsonl"

    base_dir = in_path.parent
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for entry in tqdm(data, desc="sharegpt4v_coco"):
            if not isinstance(entry, dict):
                continue
            img = entry.get("image")
            if not isinstance(img, str):
                continue
            caption = extract_caption(entry)
            img_path = base_dir / img
            if verify_images and not img_path.exists():
                continue
            if symlink_images and not (images_out / img_path.name).is_symlink():
                _ensure_symlink(img_path, images_out / img_path.name)
            line = {
                "image": img_path.name,
                "question": prompt,
                "answer": caption,
                "answers": [caption],
            }
            f.write(json.dumps(line, ensure_ascii=True) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    typer.echo(f"wrote {count} records to {out_path}")


if __name__ == "__main__":
    typer.run(main)
