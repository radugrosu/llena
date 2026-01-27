#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import typer


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(
    images: list[str] = typer.Argument(..., help="Image filenames to search for"),
    data_dir: str = opt("datasets/processed", "Root of processed datasets"),
) -> None:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data_dir not found: {root}")

    targets = set(images)
    matches: dict[str, list[str]] = {name: [] for name in images}

    for jsonl_path in root.glob("*/**/*.jsonl"):
        rel = jsonl_path.relative_to(root)
        dataset = rel.parts[0]
        split = jsonl_path.stem
        tag = f"{dataset}/{split}"
        try:
            for rec in _iter_jsonl(jsonl_path):
                image_val = rec.get("image") if isinstance(rec, dict) else None
                if isinstance(image_val, str) and image_val in targets:
                    matches[image_val].append(tag)
        except Exception:
            continue

    for name in images:
        rows = matches[name]
        if rows:
            typer.echo(f"{name}: {', '.join(rows)}")
        else:
            typer.echo(f"{name}: not found")


if __name__ == "__main__":
    typer.run(main)
