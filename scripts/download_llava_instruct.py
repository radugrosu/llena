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
    out_dir: str = opt("datasets/raw/llava_instruct", "Output directory"),
    coco_dir: str = opt("datasets/raw/coco2017", "Shared COCO root (images/)"),
    metadata_url: str = opt(
        "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
        "LLaVA-Instruct-150K metadata JSON URL",
    ),
    max_workers: int = opt(16, "Number of download workers"),
    timeout: float = opt(10.0, "Download timeout seconds"),
    limit: int | None = opt(None, "Optional cap on number of images"),
    download_images: bool = opt(True, "Download COCO images"),
    coco_train_base: str = opt(
        "http://images.cocodataset.org/train2017", "COCO train2017 base URL"
    ),
    coco_val_base: str = opt(
        "http://images.cocodataset.org/val2017", "COCO val2017 base URL"
    ),
) -> None:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    coco_root = Path(coco_dir)
    images_dir = coco_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "llava_instruct_150k.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Existing metadata JSON must be a list.")
        typer.echo(f"using existing metadata: {meta_path}")
    else:
        typer.echo(f"downloading metadata from: {metadata_url}")
        data = download_json(metadata_url)

    entries: list[dict[str, object]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        img = entry.get("image")
        if not isinstance(img, str):
            continue
        filename = Path(img).name
        new_entry = dict(entry)
        new_entry["image"] = filename
        new_entry["filename"] = filename
        entries.append(new_entry)
        if limit is not None and len(entries) >= limit:
            break

    if not meta_path.exists():
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        typer.echo(f"saved metadata: {meta_path}")

    if not download_images:
        typer.echo("skipping image downloads.")
        return

    def _task(entry: dict[str, object]) -> bool:
        filename = entry.get("filename")
        if not isinstance(filename, str):
            return False
        target = images_dir / filename
        if target.exists():
            return True
        url_train = f"{coco_train_base}/{filename}"
        if download_image(url_train, target, timeout=timeout):
            return True
        url_val = f"{coco_val_base}/{filename}"
        return download_image(url_val, target, timeout=timeout)

    def _missing_entries(items: list[dict[str, object]]) -> list[dict[str, object]]:
        missing: list[dict[str, object]] = []
        for entry in items:
            filename = entry.get("filename")
            if isinstance(filename, str) and not (images_dir / filename).exists():
                missing.append(entry)
        return missing

    missing = _missing_entries(entries)
    if not missing:
        typer.echo(f"all {len(entries)} images already present in {images_dir}")
        return

    typer.echo(f"downloading {len(missing)} images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(_task, missing), total=len(missing), unit="img"))

    success = sum(1 for r in results if r)
    typer.echo(f"downloaded {success}/{len(missing)} images to {images_dir}")

    if success < len(missing):
        missing = [e["filename"] for e, ok in zip(missing, results) if not ok]
        missing_path = out_root / "missing_images.txt"
        with missing_path.open("w", encoding="utf-8") as f:
            for name in missing:
                if isinstance(name, str):
                    f.write(name + "\n")
        typer.echo(f"wrote missing image list to {missing_path}")


if __name__ == "__main__":
    typer.run(main)
