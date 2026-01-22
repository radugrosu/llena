#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from pathlib import Path
import typer
from tqdm import tqdm


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, dst.parent)
    dst.symlink_to(rel)


def _strip_image_tokens(text: str) -> str:
    text = text.replace("<image>\n", "")
    text = text.replace("\n<image>", "")
    text = text.replace("<image>", "")
    return text.strip()


def _map_role(role: str) -> str | None:
    role_l = role.lower()
    if role_l in {"human", "user"}:
        return "user"
    if role_l in {"gpt", "assistant"}:
        return "assistant"
    if role_l == "system":
        return "system"
    return None


def _normalize_conversation(raw: object) -> list[dict[str, str]] | None:
    if not isinstance(raw, list):
        return None
    out: list[dict[str, str]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            return None
        role_raw = msg.get("from") if "from" in msg else msg.get("role")
        content_raw = msg.get("value") if "value" in msg else msg.get("content")
        if not isinstance(role_raw, str) or not isinstance(content_raw, str):
            return None
        role = _map_role(role_raw)
        if role is None:
            return None
        content = _strip_image_tokens(content_raw)
        if not content:
            return None
        out.append({"role": role, "content": content})
    if not _validate_turns(out):
        return None
    return out


def _validate_turns(conversation: list[dict[str, str]]) -> bool:
    if not conversation:
        return False
    idx = 0
    if conversation[0]["role"] == "system":
        idx = 1
    if idx >= len(conversation):
        return False
    expect = "user"
    for msg in conversation[idx:]:
        if msg["role"] != expect:
            return False
        expect = "assistant" if expect == "user" else "user"
    return conversation[-1]["role"] == "assistant"


def _split_indices(n: int, val_ratio: float, val_size: int | None, seed: int) -> tuple[set[int], set[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if val_size is None:
        if val_ratio <= 0:
            val_n = 0
        else:
            val_n = max(1, int(n * val_ratio))
    else:
        val_n = max(0, min(n, val_size))
    val_idx = set(indices[:val_n])
    train_idx = set(indices[val_n:])
    return train_idx, val_idx


def main(
    input_json: str = opt(
        "datasets/raw/llava_instruct/llava_instruct_150k.json",
        "LLaVA-Instruct-150K metadata JSON",
    ),
    out_dir: str = opt(
        "datasets/processed/llava_instruct",
        "Output directory for JSONL",
    ),
    coco_dir: str = opt(
        "datasets/raw/coco2017", "Shared COCO root (images/)"
    ),
    val_ratio: float = opt(0.15, "Validation split ratio"),
    val_size: int | None = opt(None, "Override validation size (absolute)"),
    seed: int = opt(42, "Shuffle seed"),
    limit: int | None = opt(None, "Optional cap on number of samples"),
    verify_images: bool = opt(True, "Check that image files exist"),
    symlink_images: bool = opt(True, "Symlink images into processed/images"),
    force: bool = opt(False, "Reprocess even if outputs exist"),
) -> None:
    in_path = Path(input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")
    data = json.loads(in_path.read_text())
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list.")

    coco_root = Path(coco_dir)
    images_root = coco_root / "images"
    if verify_images and not images_root.exists():
        raise FileNotFoundError(f"COCO images not found under: {images_root}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    train_path = out_root / "train.jsonl"
    val_path = out_root / "validation.jsonl"
    if train_path.exists() and val_path.exists() and images_out.exists() and not force:
        typer.echo(f"outputs already exist, skipping: {out_root}")
        return

    records: list[dict[str, object]] = []
    for entry in tqdm(data, desc="llava_instruct"):
        if not isinstance(entry, dict):
            continue
        img = entry.get("image")
        conv = entry.get("conversations")
        if not isinstance(img, str) or conv is None:
            continue
        filename = Path(img).name
        convo = _normalize_conversation(conv)
        if convo is None:
            continue
        img_path = images_root / filename
        if verify_images and not img_path.exists():
            continue
        if symlink_images:
            _ensure_symlink(img_path, images_out / filename)
        records.append({"image": filename, "conversations": convo})
        if limit is not None and len(records) >= limit:
            break

    train_idx, val_idx = _split_indices(
        len(records), val_ratio=val_ratio, val_size=val_size, seed=seed
    )

    train_count = 0
    val_count = 0
    with train_path.open("w", encoding="utf-8") as train_f, val_path.open(
        "w", encoding="utf-8"
    ) as val_f:
        for i, rec in enumerate(records):
            line = json.dumps(rec, ensure_ascii=True)
            if i in val_idx:
                val_f.write(line + "\n")
                val_count += 1
            else:
                train_f.write(line + "\n")
                train_count += 1

    typer.echo(f"wrote {train_count} records to {train_path}")
    typer.echo(f"wrote {val_count} records to {val_path}")


if __name__ == "__main__":
    typer.run(main)
