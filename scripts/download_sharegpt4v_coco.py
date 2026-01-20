#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import urllib.request
import typer
from tqdm import tqdm


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def download_json(url: str) -> list[dict[str, object]]:
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError("Metadata JSON must be a list.")
    return data


def download_image(url: str, save_path: Path, timeout: float) -> bool:
    if save_path.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if r.status != 200:
                return False
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("wb") as f:
                f.write(r.read())
        return True
    except Exception:
        return False


def main(
    out_dir: str = opt("datasets/raw/sharegpt4v_coco", "Output directory"),
    metadata_url: str = opt(
        "https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json",
        "ShareGPT4V metadata JSON URL",
    ),
    max_workers: int = opt(16, "Number of download workers"),
    timeout: float = opt(10.0, "Download timeout seconds"),
    limit: int | None = opt(None, "Optional cap on number of images"),
) -> None:
    out_root = Path(out_dir)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"downloading metadata from: {metadata_url}")
    data = download_json(metadata_url)

    coco_entries: list[dict[str, object]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        img_path = entry.get("image")
        if not isinstance(img_path, str):
            continue
        if "coco/train2017" not in img_path:
            continue

        filename = img_path.split("/")[-1]
        new_entry = dict(entry)
        new_entry["image"] = str(Path("images") / filename)
        new_entry["source_url"] = f"http://images.cocodataset.org/train2017/{filename}"
        new_entry["filename"] = filename
        coco_entries.append(new_entry)

        if limit is not None and len(coco_entries) >= limit:
            break

    typer.echo(f"filtered {len(coco_entries)} COCO entries.")

    meta_path = out_root / "sharegpt4v_coco.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(coco_entries, f, indent=2)
    typer.echo(f"saved metadata: {meta_path}")

    def _task(entry: dict[str, object]) -> bool:
        url = entry.get("source_url")
        filename = entry.get("filename")
        if not isinstance(url, str) or not isinstance(filename, str):
            return False
        return download_image(url, images_dir / filename, timeout=timeout)

    typer.echo(f"downloading {len(coco_entries)} images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(_task, coco_entries), total=len(coco_entries), unit="img"))

    success = sum(1 for r in results if r)
    typer.echo(f"downloaded {success}/{len(coco_entries)} images to {images_dir}")


if __name__ == "__main__":
    typer.run(main)
