#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import typer

if __package__ is None and __name__ == "__main__":
    raise RuntimeError("Run as module: python -m scripts.e2e_smoke")

from scripts import eval as eval_mod
from scripts import train as train_mod
from mm.config import load_config
from mm.run_config import RunConfig, Stage


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _latest_step_dir(out_dir: Path) -> Path:
    step_re = re.compile(r"^step_(\d+)$")
    best_step = -1
    best_path = None
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        m = step_re.match(child.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = child
    if best_path is None:
        raise FileNotFoundError(f"No step_* checkpoints found in {out_dir}")
    return best_path


def main(
    config: str = opt(..., "Path to YAML config"),
    stage: Stage = opt("projector", "Stage: projector | peft_lora | peft_qlora"),
    max_steps: int = opt(20, "Max training steps"),
    out_dir: str = opt("artifacts/e2e_smoke", "Output directory for checkpoints"),
    eval_batch_size: int = opt(2, "Eval batch size"),
    eval_max_samples: int | None = opt(None, "Limit eval samples (None = config)"),
    eval_log_every: int = opt(50, "Eval progress log every N batches"),
    override: list[str] = opt([], "Config override(s): KEY=VALUE (repeatable)"),
) -> None:
    train_mod.main(
        config=config,
        stage=stage,
        max_steps=max_steps,
        out_dir=out_dir,
        resume=None,
        save_every=0,
        save_trainable_only=True,
        override=override,
    )

    ckpt_dir = _latest_step_dir(Path(out_dir))
    metrics, dataset = eval_mod.run_eval(
        config=config,
        stage=stage,
        ckpt=str(ckpt_dir),
        batch_size=eval_batch_size,
        max_samples=eval_max_samples,
        override=override,
        log_every=eval_log_every,
    )

    raw_cfg = load_config(config, overrides=override)
    rc = RunConfig.from_dict(raw_cfg)
    report_path = Path(rc.paths.reports_dir) / "e2e_smoke_eval.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset,
        "checkpoint": str(ckpt_dir / "ckpt.pt"),
        "metrics": metrics,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    typer.echo(f"report: {report_path}")


if __name__ == "__main__":
    typer.run(main)
