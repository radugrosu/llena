#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import typer

if __package__ is None and __name__ == "__main__":
    raise RuntimeError("Run as module: python -m scripts.download_dataset")

from data.download import download_hf_parquet


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def main(
    dataset: str = opt(..., "Dataset: docvqa | textvqa"),
    out_dir: str = opt("datasets/raw", "Output directory for parquet files"),
) -> None:
    out_path = Path(out_dir)
    dataset_l = dataset.lower()
    local_dir = download_hf_parquet(
        dataset=dataset_l,
        out_dir=out_path,
    )
    typer.echo(f"downloaded parquet files to {local_dir}")


if __name__ == "__main__":
    typer.run(main)
