# train.py
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Literal

import torch
import typer
from dotenv import load_dotenv
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import SiglipImageProcessor

import wandb
from data.format import VqaAsInstructDataset, load_instruct_jsonl_dataset, load_vqa_jsonl_dataset
from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator, LlenaPackedCollator
from mm.config import load_config, save_resolved_config
from mm.model import LlenaModel, LlenaModelConfig
from mm.run_config import RunConfig, Stage
from scripts.utils import get_git_commit

WandbResumePolicy = Literal["auto", "always", "never"]
WANDB_RESUME_POLICIES: frozenset[WandbResumePolicy] = frozenset({"auto", "always", "never"})


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


def build_dataset(
    rc: RunConfig,
    *,
    split: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    use_split = split or rc.data.split
    cap = max_samples if max_samples is not None else rc.data.num_samples
    if rc.data.dataset == "synthetic":
        if cap is None:
            raise ValueError("data.num_samples must be set for synthetic dataset.")
        return SyntheticVQADataset(
            num_samples=int(cap),
            image_size=224,
            seed=rc.train.seed,
        )
    if rc.data.dataset == "llava_instruct":
        return load_instruct_jsonl_dataset(
            dataset=rc.data.dataset,
            data_dir=Path(rc.data.data_dir),
            split=use_split,
            max_samples=cap,
        )
    if rc.data.dataset == "llava_textvqa":
        llava_ds = load_instruct_jsonl_dataset(
            dataset="llava_instruct",
            data_dir=Path(rc.data.data_dir),
            split=use_split,
            max_samples=None,
        )
        text_ds = load_vqa_jsonl_dataset(
            dataset="textvqa",
            data_dir=Path(rc.data.data_dir),
            split=use_split,
            max_samples=None,
        )
        mix = ConcatDataset([llava_ds, VqaAsInstructDataset(text_ds)])
        if cap is None:
            return mix
        cap_n = max(0, int(cap))
        if cap_n >= len(mix):
            return mix
        return torch.utils.data.Subset(mix, range(cap_n))
    if rc.data.dataset == "sharegpt4v_coco":
        return load_vqa_jsonl_dataset(
            dataset=rc.data.dataset,
            data_dir=Path(rc.data.data_dir),
            split=use_split,
            max_samples=cap,
        )
    return load_vqa_jsonl_dataset(
        dataset=rc.data.dataset,
        data_dir=Path(rc.data.data_dir),
        split=use_split,
        max_samples=cap,
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


def eval_loop(
    model: LlenaModel,
    dl: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dl:
            batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    pixel_values=batch_t["pixel_values"],
                    input_ids=batch_t["input_ids"],
                    mm_attention_mask=batch_t["mm_attention_mask"],
                    mm_labels=batch_t["mm_labels"],
                )
                loss = out.loss
            total_loss += float(loss.item())
            count += 1
    model.train()
    if count == 0:
        return float("nan")
    return total_loss / count


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
    global_step: int,
    model: LlenaModel,
    optm: torch.optim.Optimizer,
    *,
    stage: Stage,
    save_trainable_only: bool,
    wandb_run_id: str | None,
    wandb_project: str | None,
    run_config: dict[str, object],
) -> None:
    save_path = out_dir / f"step_{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "stage": stage,
        "step": global_step,
        "cfg": _ckpt_cfg(model),
        "optimizer": optm.state_dict(),
        "run_config": run_config,
    }
    if wandb_run_id is not None:
        payload["wandb_run_id"] = wandb_run_id
    if wandb_project is not None:
        payload["wandb_project"] = wandb_project

    if save_trainable_only and stage in {"projector", "lora", "qlora"}:
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
    *,
    expected_stage: Stage | None = None,
) -> tuple[int, str, str | None, str | None, bool]:
    ckpt = torch.load(ckpt_path, map_location=device)
    step = int(ckpt["step"])
    stage = str(ckpt["stage"])
    wandb_run_id = ckpt.get("wandb_run_id") if isinstance(ckpt.get("wandb_run_id"), str) else None
    wandb_project = ckpt.get("wandb_project") if isinstance(ckpt.get("wandb_project"), str) else None
    stage_transition = False

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)

    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"], strict=True)

    if "adapter" in ckpt:
        if not model.cfg.peft_enable or not isinstance(model.llm, PeftModel):
            raise RuntimeError("Checkpoint has adapter weights but current model is not PEFT-wrapped.")
        set_peft_model_state_dict(model.llm, ckpt["adapter"])

    if expected_stage is not None and stage != expected_stage:
        stage2 = {"lora", "qlora", "full_ft"}
        if stage == "projector" and expected_stage in stage2:
            typer.echo(f"ckpt: stage transition (ckpt_stage={stage} -> expected={expected_stage}); skipping optimizer")
            step = 0
            stage_transition = True
        elif stage in stage2 and expected_stage in stage2:
            typer.echo(
                f"ckpt: stage transition (ckpt_stage={stage} -> expected={expected_stage}); "
                "skipping optimizer (assume new stage2 data)"
            )
            step = 0
            stage_transition = True
        else:
            raise ValueError(f"ckpt: stage mismatch (ckpt_stage={stage} expected={expected_stage})")
    elif "optimizer" in ckpt:
        optm.load_state_dict(ckpt["optimizer"])

    typer.echo(f"ckpt: loaded {ckpt_path}")
    return step, stage, wandb_run_id, wandb_project, stage_transition


def parse_wandb_resume_policy(policy_raw: str) -> WandbResumePolicy:
    policy = policy_raw.strip().lower()
    if policy not in WANDB_RESUME_POLICIES:
        raise ValueError(f"wandb_resume_policy must be auto|always|never, got: {policy_raw}")
    if policy == "auto":
        return "auto"
    if policy == "always":
        return "always"
    return "never"


def should_resume_wandb_run(
    *,
    resume_wandb_id: str | None,
    stage_transition: bool,
    policy: WandbResumePolicy,
) -> bool:
    if resume_wandb_id is None:
        return False
    if policy == "always":
        return True
    if policy == "never":
        return False
    return not stage_transition


def run_train(
    *,
    config: str,
    max_steps: int | None = None,
    out_dir: str = "artifacts",
    resume: str | None = None,
    save_every: int | None = None,
    save_trainable_only: bool = True,
    wandb_resume_policy: str = "auto",
    override: list[str] | None = None,
) -> None:
    overrides = override or []
    load_dotenv()
    commit = get_git_commit()
    raw_cfg = load_config(config, overrides=overrides)
    rc = RunConfig.from_dict(raw_cfg)
    resume_policy = parse_wandb_resume_policy(wandb_resume_policy)
    raw_cfg["project"]["run_name"] = rc.project.run_name  # pyright: ignore[reportIndexIssue]

    stage = rc.train.stage_name

    qlora_enable = stage == "qlora"
    device = get_device(rc.train.device, force_cuda=qlora_enable)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"{rc.project.run_name}_{run_stamp}"
    out_path = Path(out_dir) / run_dir
    out_path.mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "latest_run.txt").write_text(str(out_path), encoding="utf-8")

    save_resolved_config(raw_cfg, out_path / "resolved_config.yaml")
    set_seed(rc.train.seed)

    freeze_llm = stage == "projector"
    peft_enable = stage in {"lora", "qlora"}

    mcfg = LlenaModelConfig(
        llm_name=rc.model.llm_name,
        vision_name=rc.model.vision_name,
        num_image_tokens=rc.mm.num_image_tokens,
        projector=rc.mm.projector,
        gradient_checkpointing=rc.train.gradient_checkpointing,
        precision=rc.train.precision,
        freeze_vision=True,
        freeze_llm=freeze_llm,
        peft_enable=peft_enable,
        peft_r=rc.train.lora_r or 0,
        peft_alpha=rc.train.lora_alpha or 0,
        peft_dropout=rc.train.lora_dropout or 0.0,
        peft_target_modules=list(rc.train.lora_targets or ()),
        qlora_enable=qlora_enable,
        device="cuda" if device.type == "cuda" else "cpu",
    )

    model = LlenaModel(mcfg)

    if stage == "projector":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("projector.")

    s = trainable_param_summary(model)
    typer.echo(
        f"params: total={s['total']:,} trainable={s['trainable']:,} ({float(s['pct']):.6f}%) prefixes={s['prefixes']}"  # pyright: ignore[reportArgumentType]
    )
    typer.echo(f"effective_batch_size={rc.effective_batch_size()}")

    typer.echo("init: loading image processor")
    image_proc = SiglipImageProcessor.from_pretrained(rc.model.vision_name)

    ds = build_dataset(rc)
    collator_cls = LlenaPackedCollator if rc.data.dataset in {"llava_instruct", "llava_textvqa"} else LlenaCollator
    collator = collator_cls(
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

    eval_every = rc.train.eval_every
    eval_max_samples = rc.train.eval_max_samples
    val_dl: DataLoader | None = None
    if eval_every > 0:
        val_ds = build_dataset(rc, split="validation", max_samples=eval_max_samples or None)
        val_dl = DataLoader(
            val_ds,
            batch_size=rc.train.micro_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check freezing logic/PEFT wiring.")

    lr = rc.train.lr
    optm = torch.optim.AdamW(trainable_params, lr=lr)

    accum = rc.train.grad_accum_steps
    warmup_ratio = rc.train.warmup_ratio
    if max_steps is None:
        max_steps = rc.train.max_steps
    if max_steps is not None:
        # max_steps is expressed in optimizer-step units.
        target_optim_steps = int(max_steps)
        target_micro_steps = target_optim_steps * accum
    else:
        # Epoch mode: drop tail micro-batches that do not form a full accumulation window.
        total_micro_steps = len(dl) * rc.train.epochs
        target_micro_steps = (total_micro_steps // accum) * accum
        dropped_micro_steps = total_micro_steps - target_micro_steps
        if dropped_micro_steps > 0:
            typer.echo(f"train: dropping tail micro-batches ({dropped_micro_steps}) to keep full accumulation windows")
        target_optim_steps = target_micro_steps // accum
    warmup_steps = int(target_optim_steps * warmup_ratio)

    def lr_scale(step_idx: int) -> float:
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        denom = max(1, target_optim_steps - warmup_steps)
        progress = (step_idx - warmup_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if rc.train.lr_schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule: {rc.train.lr_schedule}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optm, lr_scale)

    # autocast precision on CUDA
    use_amp = device.type == "cuda" and rc.train.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if rc.train.precision == "bf16" else torch.float16
    use_scaler = rc.train.precision == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)  # pyright: ignore[reportPrivateImportUsage]

    log_every = rc.train.log_every
    save_every_eff = save_every if save_every is not None else rc.train.save_every
    max_grad_norm = rc.train.max_grad_norm

    start_step = 0
    global_step = 0
    resume_wandb_id: str | None = None
    resume_wandb_project: str | None = None
    resume_stage_transition = False
    if resume is not None:
        ckpt_path = Path(resume)
        if ckpt_path.is_dir():
            direct_ckpt = ckpt_path / "ckpt.pt"
            if direct_ckpt.exists():
                ckpt_path = direct_ckpt
            elif not ckpt_path.name.startswith("step"):
                step_paths = sorted(
                    ckpt_path.glob("step_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not step_paths:
                    raise FileNotFoundError(f"No step_* checkpoints found in: {ckpt_path}")
                ckpt_path = step_paths[0] / "ckpt.pt"
            else:
                ckpt_path = ckpt_path / "ckpt.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        start_step, ckpt_stage, resume_wandb_id, resume_wandb_project, resume_stage_transition = load_ckpt(
            ckpt_path, model, optm, device, expected_stage=stage
        )
        typer.echo(f"resumed from {ckpt_path} at step={start_step} (ckpt_stage={ckpt_stage})")
        if start_step > 0:
            scheduler.last_epoch = start_step - 1
            global_step = start_step

    wandb_run_id: str | None = None
    wandb_project: str | None = None
    if rc.logging.backend == "wandb":
        should_resume_wandb = should_resume_wandb_run(
            resume_wandb_id=resume_wandb_id,
            stage_transition=resume_stage_transition,
            policy=resume_policy,
        )
        if should_resume_wandb and resume_wandb_id is not None:
            project_name = resume_wandb_project or rc.project.name
            typer.echo(f"wandb: resuming existing run id={resume_wandb_id} (policy={resume_policy})")
            wandb.init(
                id=resume_wandb_id,
                resume="allow",
                project=project_name,
                name=rc.project.run_name,
                config=raw_cfg,
            )
        else:
            if resume_wandb_id is not None and resume_policy == "auto" and resume_stage_transition:
                typer.echo("wandb: stage transition resume detected; starting a new run (policy=auto)")
            elif resume_wandb_id is not None and resume_policy == "never":
                typer.echo("wandb: checkpoint has run id but policy=never; starting a new run")
            wandb.init(project=rc.project.name, name=rc.project.run_name, config=raw_cfg)
        if wandb.run is not None:
            wandb_run_id = wandb.run.id
            wandb_project = wandb.run.project
        wandb.define_metric("global_step", hidden=True)
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("val/*", step_metric="global_step")
        wandb.define_metric("textvqa_eval/*", step_metric="global_step")
    elif rc.logging.backend == "mlflow":
        raise NotImplementedError("mlflow logging not implemented.")

    model.train()
    optm.zero_grad(set_to_none=True)

    step = start_step * accum
    last_loss = 0.0
    last_log_time = time.perf_counter()
    last_log_tokens = 0
    last_log_samples = 0

    for epoch in range(rc.train.epochs):
        for batch in dl:
            if step >= target_micro_steps:
                break
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

            scaler.scale(loss).backward()

            last_log_tokens += int(batch_t["mm_attention_mask"].sum().item())
            last_log_samples += batch_t["mm_attention_mask"].size(0)

            if step % accum == 0:
                if max_grad_norm > 0:
                    if use_scaler:
                        scaler.unscale_(optm)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                if use_scaler:
                    scaler.step(optm)
                    scaler.update()
                else:
                    optm.step()
                scheduler.step()
                optm.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    now = time.perf_counter()
                    dt = max(now - last_log_time, 1e-8)
                    tokens_per_s = last_log_tokens / dt
                    samples_per_s = last_log_samples / dt

                    last_loss = float(loss.item() * accum)
                    typer.echo(
                        f"stage={stage} epoch={epoch + 1}/{rc.train.epochs} global_step={global_step} "
                        f"loss={last_loss:.4f} samples/s={samples_per_s:.2f} tok/s={tokens_per_s:.0f}"
                    )
                    if rc.logging.backend == "wandb":
                        wandb.log(
                            {
                                "global_step": global_step,
                                "train/loss": last_loss,
                                "train/samples_per_s": samples_per_s,
                                "train/tokens_per_s": tokens_per_s,
                                "train/lr": optm.param_groups[0]["lr"],
                                **({"train/scale": float(scaler.get_scale())} if use_scaler else {}),
                            }
                        )
                    last_log_time = now
                    last_log_tokens = 0
                    last_log_samples = 0

                if eval_every > 0 and val_dl is not None and global_step % eval_every == 0:
                    typer.echo(f"val: starting eval at global_step={global_step}")
                    val_loss = eval_loop(
                        model,
                        val_dl,
                        device,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                    typer.echo(f"val: global_step={global_step} loss={val_loss:.4f}")
                    if rc.logging.backend == "wandb":
                        wandb.log({"global_step": global_step, "val/loss": val_loss})

                if save_every_eff > 0 and global_step % save_every_eff == 0:
                    save_ckpt(
                        out_path,
                        global_step,
                        model,
                        optm,
                        stage=stage,
                        save_trainable_only=save_trainable_only,
                        wandb_run_id=wandb_run_id,
                        wandb_project=wandb_project,
                        run_config=raw_cfg,
                    )

            if global_step >= target_optim_steps:
                break
        if step >= target_micro_steps or global_step >= target_optim_steps:
            break

    save_ckpt(
        out_path,
        global_step,
        model,
        optm,
        stage=stage,
        save_trainable_only=save_trainable_only,
        wandb_run_id=wandb_run_id,
        wandb_project=wandb_project,
        run_config=raw_cfg,
    )

    typer.echo("done")

    report = {
        "stage": stage,
        "step": step,
        "global_step": global_step,
        "loss": last_loss,
        "ckpt_dir": str(out_path / f"step_{global_step}"),
        "commit": commit,
    }
    report_path = Path(rc.paths.reports_dir) / f"train_{rc.project.run_name}_step{global_step}_{stage}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if rc.logging.backend == "wandb":
        wandb.finish()


def main(
    config: str = opt(..., "Path to YAML config"),
    max_steps: int | None = opt(None, "Max training steps (overrides config)"),
    out_dir: str = opt("artifacts", "Base output directory for checkpoints/config"),
    resume: str | None = opt(None, "Path to a ckpt.pt or a step_* directory to resume from"),
    save_every: int | None = opt(None, "Override train.save_every (int). If None, use config."),
    save_trainable_only: bool = opt(True, "If true, saves only projector/adapters for PEFT stages."),
    wandb_resume_policy: str = opt(
        "auto",
        "W&B resume behavior: auto (same-stage only), always, or never.",
    ),
    override: list[str] = opt([], "Config override(s): KEY=VALUE (repeatable)"),
) -> None:
    run_train(
        config=config,
        max_steps=max_steps,
        out_dir=out_dir,
        resume=resume,
        save_every=save_every,
        save_trainable_only=save_trainable_only,
        wandb_resume_policy=wandb_resume_policy,
        override=override,
    )


if __name__ == "__main__":
    typer.run(main)
