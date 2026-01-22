#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
import typer
import torch


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _ckpt_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "ckpt.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _load(path: Path, *, mmap: bool) -> tuple[str, int]:
    ckpt = torch.load(path, map_location="cpu", mmap=mmap)
    stage = str(ckpt.get("stage", ""))
    step = int(ckpt.get("step", -1))
    return stage, step


def main(
    ckpt: str = opt(..., "Path to ckpt.pt or step_* directory"),
    repeats: int = opt(5, "Number of timed loads per mode"),
) -> None:
    path = _ckpt_path(ckpt)

    for mmap in (True, False):
        # Warmup
        _load(path, mmap=mmap)
        t0 = time.perf_counter()
        last = ("", -1)
        for _ in range(repeats):
            last = _load(path, mmap=mmap)
        dt = time.perf_counter() - t0
        avg = dt / max(repeats, 1)
        typer.echo(f"mmap={mmap} avg={avg * 1000:.2f}ms stage={last[0]} step={last[1]}")


if __name__ == "__main__":
    typer.run(main)
