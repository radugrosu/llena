from __future__ import annotations

import re
from pathlib import Path
import json
from typing import cast

import torch
import typer
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor
from dotenv import load_dotenv
from peft import PeftModel, set_peft_model_state_dict
import wandb

from scripts.utils import get_git_commit

from data.format import load_vqa_jsonl_dataset
from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator
from mm.model import LlenaModel, LlenaModelConfig
from mm.run_config import RunConfig, Stage
from mm.types import VQASample


def opt(default: object, help: str):
    return typer.Option(default, help=help)


def _apply_overrides(cfg: dict[str, object], overrides: list[str]) -> dict[str, object]:
    if not overrides:
        return cfg
    from mm.config import _set_dotted_key  # type: ignore[reportPrivateImportUsage]

    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        _set_dotted_key(out, k.strip(), v.strip())
    return out


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


def build_dataset(rc: RunConfig, *, max_samples: int | None) -> Dataset:
    if rc.data.dataset == "synthetic":
        num_samples = rc.data.num_samples if max_samples is None else max_samples
        return SyntheticVQADataset(
            num_samples=num_samples,
            image_size=rc.data.image_size,
            seed=rc.train.seed,
        )
    if rc.data.dataset in {"llava_instruct", "llava_textvqa"}:
        raise ValueError("llava_instruct is not supported for eval.")
    cap = rc.data.num_samples if max_samples is None else max_samples
    return load_vqa_jsonl_dataset(
        dataset=rc.data.dataset,
        data_dir=Path(rc.data.data_dir),
        split=rc.data.split,
        max_samples=cap,
    )


def _ckpt_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "ckpt.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _load_ckpt_meta(path: str) -> dict[str, object]:
    ckpt = torch.load(_ckpt_path(path), map_location="cpu", mmap=True)
    return ckpt


def _detect_stage(path: str) -> Stage:
    ckpt = _load_ckpt_meta(path)
    stage = str(ckpt.get("stage", ""))
    if stage in ("smoke", "projector", "peft_lora", "peft_qlora", "full_ft"):
        return stage
    else:
        raise ValueError(f"Checkpoint stage is unknown or missing: {stage!r}")


def _detect_step(path: str) -> int:
    ckpt = _load_ckpt_meta(path)
    step = ckpt.get("step")
    if isinstance(step, (int, float)):
        return int(step)
    raise ValueError(f"No valid step found: {step}")


def load_eval_ckpt(path: str, model: LlenaModel, device: torch.device) -> None:
    ckpt = torch.load(_ckpt_path(path), map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)

    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"], strict=True)

    if "adapter" in ckpt:
        if not model.cfg.peft_enable or not isinstance(model.llm, PeftModel):
            raise RuntimeError("Checkpoint has adapter weights but current model is not PEFT-wrapped.")
        set_peft_model_state_dict(model.llm, ckpt["adapter"])
    typer.echo(f"ckpt: loaded {_ckpt_path(path)}")


_PUNCT = re.compile(r"[^0-9a-zA-Z]+")


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT.sub(" ", text)
    text = " ".join(text.split())
    return text


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    norm_pred = normalize_answer(pred)
    matches = 0
    for ans in answers:
        if normalize_answer(ans) == norm_pred:
            matches += 1
    return min(1.0, matches / 3.0)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def anls_score(pred: str, answers: list[str]) -> float:
    norm_pred = normalize_answer(pred)
    best = 0.0
    for ans in answers:
        norm_ans = normalize_answer(ans)
        if not norm_ans:
            continue
        dist = _levenshtein(norm_pred, norm_ans)
        sim = 1.0 - (dist / max(len(norm_ans), 1))
        if sim < 0.5:
            sim = 0.0
        if sim > best:
            best = sim
    return best


def eval_loop(
    model: LlenaModel,
    dl: DataLoader,
    device: torch.device,
    dataset: str,
    log_every: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    em = 0
    vqa_acc = 0.0
    anls = 0.0

    with torch.no_grad():
        for step, (batch, answers_list) in enumerate(dl, start=1):
            batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

            out = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                mm_attention_mask=batch_t["mm_attention_mask"],
                mm_labels=batch_t["mm_labels"],
            )
            loss = float(out.loss)
            total_loss += loss * len(answers_list)
            total += len(answers_list)

            logits = out.logits
            pred_ids = logits.argmax(dim=-1)
            mm_labels = batch_t["mm_labels"]

            for i, answers in enumerate(answers_list):
                mask = mm_labels[i] != -100
                pred_seq = pred_ids[i][mask].tolist()
                pred_text = model.tokenizer.decode(pred_seq, skip_special_tokens=True)
                if dataset == "textvqa":
                    vqa_acc += vqa_accuracy(pred_text, answers)
                elif dataset == "docvqa":
                    anls += anls_score(pred_text, answers)
                else:
                    if normalize_answer(pred_text) == normalize_answer(answers[0]):
                        em += 1

            if log_every > 0 and step % log_every == 0:
                avg_loss = total_loss / total if total else 0.0
                typer.echo(
                    f"eval step={step} count={total} avg_loss={avg_loss:.4f}",
                    err=True,
                )

    avg_loss = total_loss / total if total else 0.0
    em_rate = em / total if total else 0.0
    vqa_rate = vqa_acc / total if total else 0.0
    anls_rate = anls / total if total else 0.0
    return {
        "avg_loss": avg_loss,
        "exact_match": em_rate,
        "vqa_accuracy": vqa_rate,
        "anls": anls_rate,
        "count": float(total),
    }


def run_eval(
    *,
    stage: Stage,
    ckpt: str,
    batch_size: int | None,
    max_samples: int | None,
    override: list[str],
    log_every: int,
) -> tuple[dict[str, float], str]:
    commit = get_git_commit()
    ckpt_meta = _load_ckpt_meta(ckpt)
    raw_cfg = ckpt_meta.get("run_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("Checkpoint missing run_config; cannot run eval without config.")
    raw_cfg = _apply_overrides(raw_cfg, override)
    rc = RunConfig.from_dict(raw_cfg)
    base_run_name = raw_cfg["project"]["run_name"]
    if not isinstance(base_run_name, str) or not base_run_name:
        raise ValueError("project.run_name must be a non-empty string in checkpoint config")

    qlora_enable = stage == "peft_qlora"
    device = get_device(rc.train.device, force_cuda=qlora_enable)

    mcfg = LlenaModelConfig(
        llm_name=rc.model.llm_name,
        vision_name=rc.model.vision_name,
        num_image_tokens=rc.mm.num_image_tokens,
        projector=rc.mm.projector,
        gradient_checkpointing=False,
        freeze_vision=True,
        freeze_llm=True,
        peft_enable=stage in {"peft_lora", "peft_qlora"},
        peft_r=rc.train.lora_r,
        peft_alpha=rc.train.lora_alpha,
        peft_dropout=rc.train.lora_dropout,
        peft_target_modules=list(rc.train.lora_targets),
        qlora_enable=qlora_enable,
        device="cuda" if device.type == "cuda" else "cpu",
    )

    typer.echo("eval: building model")
    model = LlenaModel(mcfg)

    typer.echo("eval: loading checkpoint")
    load_eval_ckpt(ckpt, model, device)

    run_id: str | None = None
    if rc.logging.backend == "wandb":
        meta = _load_ckpt_meta(ckpt)
        run_id_val = meta.get("wandb_run_id")
        project_val = meta.get("wandb_project")
        if not isinstance(run_id_val, str) or not run_id_val:
            typer.echo("eval: wandb_run_id missing in checkpoint; skipping W&B logging.")
        else:
            run_id = run_id_val
            project_name = str(project_val) if isinstance(project_val, str) and project_val else rc.project.name
            wandb.init(
                id=run_id,
                resume="allow",
                project=project_name,
                name=base_run_name,
                config=raw_cfg,
            )
            wandb.define_metric("global_step", hidden=True)
            wandb.define_metric("eval/*", step_metric="global_step")
    elif rc.logging.backend == "mlflow":
        raise NotImplementedError("mlflow logging not implemented.")

    typer.echo("eval: loading image processor")
    image_proc = SiglipImageProcessor.from_pretrained(rc.model.vision_name)
    if batch_size is None:
        batch_size = rc.eval.batch_size
    if max_samples is None:
        max_samples = rc.eval.max_samples

    typer.echo(f"eval: building dataset split={rc.data.split} max_samples={max_samples}")
    ds = build_dataset(rc, max_samples=max_samples)

    collator = LlenaCollator(
        tokenizer=model.tokenizer,
        image_processor=image_proc,
        max_seq_len=rc.train.max_seq_len,
        num_image_tokens=rc.mm.num_image_tokens,
        pad_to_multiple_of=8,
    )

    def _eval_collate(
        batch: list[VQASample],
    ) -> tuple[dict[str, torch.Tensor], list[list[str]]]:
        out = collator(batch)
        answers_list: list[list[str]] = []
        for ex in batch:
            if "answers" in ex and ex["answers"]:
                answers_list.append(ex["answers"])
            else:
                answers_list.append([ex["answer"]])
        return out, answers_list

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_eval_collate,
        num_workers=0,
    )

    typer.echo("eval: starting eval loop")
    metrics = eval_loop(model, dl, device, rc.data.dataset, log_every)
    if rc.logging.backend == "wandb" and run_id is not None:
        payload = {f"eval/{k}": v for k, v in metrics.items() if k != "count"}
        ckpt_step = _detect_step(ckpt)
        if ckpt_step >= 0:
            payload["global_step"] = float(ckpt_step)
        wandb.log(payload)
    report = {
        "dataset": rc.data.dataset,
        "split": rc.data.split,
        "checkpoint": ckpt,
        "metrics": metrics,
        "commit": commit,
    }
    ckpt_step = _detect_step(ckpt)
    report_path = (
        Path(rc.paths.reports_dir)
        / f"eval_{rc.project.run_name}_step{ckpt_step}_{rc.data.dataset}_{rc.data.split}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if rc.logging.backend == "wandb" and run_id is not None:
        wandb.finish()

    return metrics, rc.data.dataset


def main(
    stage: str = opt("auto", "Stage: auto | smoke | projector | peft_lora | peft_qlora | full_ft"),
    ckpt: str = opt(..., "Path to a ckpt.pt or a step_* directory"),
    batch_size: int | None = opt(None, "Eval batch size (None = config)"),
    max_samples: int | None = opt(None, "Limit number of samples (None = config)"),
    dataset: str | None = opt(None, "Override data.dataset for eval"),
    split: str | None = opt(None, "Override data.split for eval"),
    override: list[str] = opt([], "Config override(s): KEY=VALUE (repeatable)"),
    log_every: int = opt(100, "Log progress every N batches (0 disables)"),
) -> None:
    load_dotenv()
    typer.echo("eval: starting", err=True)
    if stage == "auto":
        stage = _detect_stage(ckpt)
        typer.echo(f"eval: detected stage={stage}", err=True)
    if stage not in {"smoke", "projector", "peft_lora", "peft_qlora", "full_ft"}:
        raise ValueError(f"Unknown stage: {stage}")
    if dataset is not None:
        override.append(f"data.dataset={dataset}")
    if split is not None:
        override.append(f"data.split={split}")
    metrics, dataset = run_eval(
        stage=cast(Stage, stage),
        ckpt=ckpt,
        batch_size=batch_size,
        max_samples=max_samples,
        override=override,
        log_every=log_every,
    )
    msg = f"eval: count={int(metrics['count'])} avg_loss={metrics['avg_loss']:.4f}"
    if dataset == "textvqa":
        msg += f" vqa_acc={metrics['vqa_accuracy']:.4f}"
    elif dataset == "docvqa":
        msg += f" anls={metrics['anls']:.4f}"
    else:
        msg += f" exact_match={metrics['exact_match']:.4f}"
    typer.echo(msg)
    typer.echo("eval: done", err=True)


if __name__ == "__main__":
    typer.run(main)
