#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

from scripts import eval as eval_script
from scripts import train as train_script


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"completed": []}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _latest_step(run_dir: Path) -> Path:
    steps = sorted(run_dir.glob("step_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not steps:
        raise FileNotFoundError(f"No step_* checkpoints found in: {run_dir}")
    return steps[0]


def _is_done(state: dict[str, Any], phase: str) -> bool:
    completed = set(state.get("completed", []))
    return phase in completed


def _mark_done(state: dict[str, Any], state_file: Path, phase: str) -> None:
    state.setdefault("completed", []).append(phase)
    _write_state(state_file, state)


def main(
    cfg_profile: str = opt("cpu", "Config profile"),
    stage1_cfg: str | None = opt(None, "Stage1 config (defaults by profile)"),
    stage2_cfg: str | None = opt(None, "Stage2 config (defaults by profile)"),
    eval_samples_path: str | None = opt(None, "Eval samples JSON"),
    state_path: str = opt("artifacts/pipeline_smoke_state.json", "State file"),
    completed_dir: str = opt("artifacts/completed_runs", "Completed runs archive"),
) -> None:
    stage1_cfg = stage1_cfg or f"configs/{cfg_profile}/sharegpt4v_train_qwen2.5-0.5b_siglip224.yaml"
    stage2_cfg = stage2_cfg or f"configs/{cfg_profile}/llava_textvqa_train_qwen2.5-0.5b_siglip224.yaml"

    state_file = Path(state_path)
    state = _read_state(state_file)

    # stage1: train
    if not _is_done(state, "stage1_train"):
        train_script.run_train(config=stage1_cfg)
        run1 = Path("artifacts/latest_run.txt").read_text(encoding="utf-8").strip()
        state["run1"] = run1
        _mark_done(state, state_file, "stage1_train")

    # stage1: eval (sharegpt4v_coco validation)
    if not _is_done(state, "stage1_eval_sharegpt4v"):
        run1 = Path(str(state.get("run1", "")))
        ckpt1 = _latest_step(run1)
        eval_script.run_eval_main(ckpt=str(ckpt1), override=["data.split=validation"])
        _mark_done(state, state_file, "stage1_eval_sharegpt4v")

    # stage1: eval (textvqa validation)
    if not _is_done(state, "stage1_eval_textvqa"):
        run1 = Path(str(state.get("run1", "")))
        ckpt1 = _latest_step(run1)
        eval_script.run_eval_main(
            ckpt=str(ckpt1),
            dataset="textvqa",
            split="validation",
            eval_samples_path=eval_samples_path,
        )
        _mark_done(state, state_file, "stage1_eval_textvqa")

    # stage2: train
    if not _is_done(state, "stage2_train"):
        run1 = str(state.get("run1", ""))
        train_script.run_train(config=stage2_cfg, resume=run1)
        run2 = Path("artifacts/latest_run.txt").read_text(encoding="utf-8").strip()
        state["run2"] = run2
        _mark_done(state, state_file, "stage2_train")

    # stage2: eval (textvqa validation)
    if not _is_done(state, "stage2_eval_textvqa"):
        run2 = Path(str(state.get("run2", "")))
        ckpt2 = _latest_step(run2)
        eval_script.run_eval_main(
            ckpt=str(ckpt2),
            dataset="textvqa",
            split="validation",
            eval_samples_path=eval_samples_path,
        )
        _mark_done(state, state_file, "stage2_eval_textvqa")

    done_dir = Path(completed_dir)
    done_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = done_dir / f"pipeline_smoke_state_{stamp}.json"
    state_file.replace(dest)
    typer.echo(f"pipeline: complete (state archived to {dest})")


if __name__ == "__main__":
    typer.run(main)
