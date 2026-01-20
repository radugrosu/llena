# train.py
from __future__ import annotations

from pathlib import Path
import json
import time

import torch
import typer
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import wandb

from scripts.utils import get_git_commit

from mm.config import load_config, save_resolved_config
from mm.model import LlenaModel, LlenaModelConfig
from mm.collator import LlenaCollator
from mm.run_config import RunConfig, Stage
from data.synthetic import SyntheticVQADataset
from data.format import load_vqa_jsonl_dataset


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str, *, force_cuda: bool) -> torch.device:
    if force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but not available.")
        return torch.device("cuda")

    d = device.lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"device must be auto|cpu|cuda, got: {device}")


def build_dataset(rc: RunConfig) -> Dataset:
    if rc.data.dataset == "synthetic":
        return SyntheticVQADataset(
            num_samples=rc.data.num_samples,
            image_size=rc.data.image_size,
            seed=rc.train.seed,
        )
    if rc.data.dataset == "sharegpt4v_coco":
        max_samples = rc.data.num_samples if rc.data.num_samples > 0 else None
        return load_vqa_jsonl_dataset(
            dataset=rc.data.dataset,
            data_dir=Path(rc.data.data_dir),
            split=rc.data.split,
            max_samples=max_samples,
        )
    max_samples = rc.data.num_samples if rc.data.num_samples > 0 else None
    return load_vqa_jsonl_dataset(
        dataset=rc.data.dataset,
        data_dir=Path(rc.data.data_dir),
        split=rc.data.split,
        max_samples=max_samples,
    )


def trainable_param_summary(
    model: torch.nn.Module,
) -> dict[str, int | float | dict[str, int]]:
    total = 0
    trainable = 0
    prefixes: dict[str, int] = {}

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            prefix = name.split(".", 1)[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

    pct = 100.0 * (trainable / total) if total else 0.0
    return {"total": total, "trainable": trainable, "pct": pct, "prefixes": prefixes}


def _ckpt_cfg(model: LlenaModel) -> dict[str, object]:
    return {
        "llm_name": model.cfg.llm_name,
        "vision_name": model.cfg.vision_name,
        "num_image_tokens": model.cfg.num_image_tokens,
        "projector": model.cfg.projector,
        "peft_enable": bool(model.cfg.peft_enable),
        "qlora_enable": bool(model.cfg.qlora_enable),
    }


def save_ckpt(
    out_dir: Path,
    step: int,
    model: LlenaModel,
    optm: torch.optim.Optimizer,
    *,
    stage: Stage,
    save_trainable_only: bool,
) -> None:
    save_path = out_dir / f"step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "stage": stage,
        "step": step,
        "cfg": _ckpt_cfg(model),
        "optimizer": optm.state_dict(),
    }

    if save_trainable_only and stage in {"projector", "peft_lora", "peft_qlora"}:
        payload["projector"] = model.projector.state_dict()
        if model.cfg.peft_enable:
            if not isinstance(model.llm, PeftModel):
                raise RuntimeError("peft_enable=True but model.llm is not a PeftModel.")
            payload["adapter"] = get_peft_model_state_dict(model.llm)
    else:
        payload["model"] = model.state_dict()

    torch.save(payload, save_path / "ckpt.pt")
    typer.echo(f"ckpt: saved {save_path / 'ckpt.pt'}")


def load_ckpt(
    ckpt_path: Path,
    model: LlenaModel,
    optm: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, str]:
    ckpt = torch.load(ckpt_path, map_location=device)
    step = int(ckpt["step"])
    stage = str(ckpt["stage"])

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)

    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"], strict=True)

    if "adapter" in ckpt:
        if not model.cfg.peft_enable or not isinstance(model.llm, PeftModel):
            raise RuntimeError("Checkpoint has adapter weights but current model is not PEFT-wrapped.")
        set_peft_model_state_dict(model.llm, ckpt["adapter"])

    if "optimizer" in ckpt:
        optm.load_state_dict(ckpt["optimizer"])

    typer.echo(f"ckpt: loaded {ckpt_path}")
    return step, stage


def main(
    config: str = opt(..., "Path to YAML config"),
    stage: Stage = opt("smoke", "Stage: smoke | projector | peft_lora | peft_qlora"),
    max_steps: int = opt(200, "Max training steps"),
    out_dir: str = opt("artifacts/ckpt_run", "Output directory for checkpoints/config"),
    resume: str | None = opt(None, "Path to a ckpt.pt or a step_* directory to resume from"),
    save_every: int | None = opt(None, "Override train.save_every (int). If None, use config."),
    save_trainable_only: bool = opt(True, "If true, saves only projector/adapters for PEFT stages."),
    override: list[str] = opt([], "Config override(s): KEY=VALUE (repeatable)"),
) -> None:
    commit = get_git_commit()
    if stage not in {"smoke", "projector", "peft_lora", "peft_qlora"}:
        raise ValueError(f"Unknown stage: {stage}")

    raw_cfg = load_config(config, overrides=override)
    rc = RunConfig.from_dict(raw_cfg)

    qlora_enable = stage == "peft_qlora"
    device = get_device(rc.train.device, force_cuda=qlora_enable)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    save_resolved_config(raw_cfg, out_path / "resolved_config.yaml")
    set_seed(rc.train.seed)

    freeze_llm = stage in {"smoke", "projector"}
    peft_enable = stage in {"peft_lora", "peft_qlora"}

    mcfg = LlenaModelConfig(
        llm_name=rc.model.llm_name,
        vision_name=rc.model.vision_name,
        num_image_tokens=rc.mm.num_image_tokens,
        projector=rc.mm.projector,
        gradient_checkpointing=rc.train.gradient_checkpointing,
        freeze_vision=True,
        freeze_llm=freeze_llm,
        peft_enable=peft_enable,
        peft_r=rc.train.lora_r,
        peft_alpha=rc.train.lora_alpha,
        peft_dropout=rc.train.lora_dropout,
        peft_target_modules=list(rc.train.lora_targets),
        qlora_enable=qlora_enable,
        device="cuda" if device.type == "cuda" else "cpu",
    )

    model = LlenaModel(mcfg)

    if stage in {"smoke", "projector"}:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("projector.")

    s = trainable_param_summary(model)
    typer.echo(
        f"params: total={s['total']:,} trainable={s['trainable']:,} ({float(s['pct']):.6f}%) prefixes={s['prefixes']}"  # pyright: ignore[reportArgumentType]
    )
    typer.echo(f"effective_batch_size={rc.effective_batch_size()}")

    image_proc = SiglipImageProcessor.from_pretrained(rc.model.vision_name)

    ds = build_dataset(rc)

    collator = LlenaCollator(
        tokenizer=model.tokenizer,
        image_processor=image_proc,
        max_seq_len=rc.train.max_seq_len,
        num_image_tokens=rc.mm.num_image_tokens,
        pad_to_multiple_of=8,
    )

    dl = DataLoader(
        ds,
        batch_size=rc.train.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check freezing logic/PEFT wiring.")

    lr = rc.train.lr_projector if stage == "projector" else rc.train.lr_lora
    optm = torch.optim.AdamW(trainable_params, lr=lr)

    # bf16-only autocast on CUDA
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    accum = rc.train.grad_accum_steps
    log_every = rc.train.log_every
    save_every_eff = save_every if save_every is not None else rc.train.save_every
    max_grad_norm = rc.train.max_grad_norm

    if rc.logging.backend == "wandb":
        wandb.init(project=rc.project.name, name=rc.project.run_name, config=raw_cfg)
    elif rc.logging.backend == "mlflow":
        raise NotImplementedError("mlflow logging not implemented.")

    start_step = 0
    if resume is not None:
        ckpt_path = Path(resume)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        start_step, ckpt_stage = load_ckpt(ckpt_path, model, optm, device)
        typer.echo(f"resumed from {ckpt_path} at step={start_step} (ckpt_stage={ckpt_stage})")

    model.train()
    optm.zero_grad(set_to_none=True)

    step = start_step
    last_loss = 0.0
    for batch in dl:
        step += 1
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                mm_attention_mask=batch_t["mm_attention_mask"],
                mm_labels=batch_t["mm_labels"],
            )
            loss = out.loss / accum

        loss.backward()

        if step % accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optm.step()
            optm.zero_grad(set_to_none=True)

        if step % log_every == 0:
            last_loss = float(loss.item() * accum)
            typer.echo(f"stage={stage} step={step} loss={last_loss:.4f}")
            if rc.logging.backend == "wandb":
                wandb.log({"train/loss": last_loss, "step": step})

        if save_every_eff > 0 and step % save_every_eff == 0:
            save_ckpt(
                out_path,
                step,
                model,
                optm,
                stage=stage,
                save_trainable_only=save_trainable_only,
            )

        if step >= max_steps:
            break

    save_ckpt(
        out_path,
        step,
        model,
        optm,
        stage=stage,
        save_trainable_only=save_trainable_only,
    )
    typer.echo("done")

    report = {
        "stage": stage,
        "step": step,
        "loss": last_loss,
        "ckpt_dir": str(out_path / f"step_{step}"),
        "commit": commit,
    }
    report_path = Path(rc.paths.reports_dir) / f"train_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if rc.logging.backend == "wandb":
        wandb.finish()


if __name__ == "__main__":
    typer.run(main)
