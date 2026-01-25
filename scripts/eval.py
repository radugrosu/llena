from __future__ import annotations

import re
from pathlib import Path
import json
from typing import Literal

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
from mm.model import LlenaModel, LlenaModelConfig, _select_or_pad_tokens
from mm.run_config import RunConfig
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
            image_size=224,
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
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
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

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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


def _pad_batch_1d(
    sequences: list[list[int]],
    *,
    pad_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("pad_batch_1d: sequences is empty")
    max_len = max(len(s) for s in sequences)
    batch = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    attn = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        t = len(seq)
        batch[i, :t] = torch.tensor(seq, dtype=torch.long)
        attn[i, :t] = 1
    return batch, attn


def _collate_eval_teacher(
    collator: LlenaCollator,
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


def _collate_eval_generate(
    image_proc: SiglipImageProcessor,
    batch: list[VQASample],
) -> tuple[torch.Tensor, list[str], list[list[str]]]:
    images = [ex["image"] for ex in batch]
    questions = [ex["question"] for ex in batch]
    answers_list: list[list[str]] = []
    for ex in batch:
        if "answers" in ex and ex["answers"]:
            answers_list.append(ex["answers"])
        else:
            answers_list.append([ex["answer"]])
    vision = image_proc(images=images, return_tensors="pt")
    return vision["pixel_values"], questions, answers_list


def eval_loop_generate(
    model: LlenaModel,
    dl: DataLoader,
    device: torch.device,
    dataset: str,
    log_every: int,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_new_tokens: int = 128,
    repetition_penalty: float = 1.0,
) -> dict[str, float]:
    model.eval()
    total = 0
    em = 0.0
    vqa_acc = 0.0
    anls = 0.0

    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")

    with torch.no_grad():
        for step, (images, questions, answers_list) in enumerate(dl, start=1):
            pixel_values = images.to(device)
            prompt_ids: list[list[int]] = []
            for q in questions:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors=None,
                )
                prompt_ids.append(ids)

            input_ids, attention_mask = _pad_batch_1d(prompt_ids, pad_value=int(pad_id))
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            v_out = model.vision(pixel_values=pixel_values)
            v_tokens = v_out.last_hidden_state
            if model.freeze_vision:
                v_tokens = v_tokens.detach()
            v_tokens = _select_or_pad_tokens(v_tokens, model.num_image_tokens)
            proj_dtype = next(model.projector.parameters()).dtype
            v_tokens = v_tokens.to(dtype=proj_dtype)
            img_embeds = model.projector(v_tokens)

            text_embeds = model.llm.get_input_embeddings()(input_ids)  # pyright: ignore[reportCallIssue]
            if img_embeds.dtype != text_embeds.dtype:
                img_embeds = img_embeds.to(dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)

            prefix_mask = torch.ones(
                (attention_mask.size(0), model.num_image_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            mm_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                gen_ids = model.llm.generate(  # pyright: ignore[reportCallIssue]
                    inputs_embeds=inputs_embeds,
                    attention_mask=mm_attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prefix_len = inputs_embeds.size(1)
            gen_new = gen_ids[:, prefix_len:].tolist()
            preds = tokenizer.batch_decode(gen_new, skip_special_tokens=True)

            for pred_text, answers in zip(preds, answers_list):
                if dataset == "textvqa":
                    vqa_acc += vqa_accuracy(pred_text, answers)
                elif dataset == "docvqa":
                    anls += anls_score(pred_text, answers)
                else:
                    if normalize_answer(pred_text) == normalize_answer(answers[0]):
                        em += 1.0
                total += 1

            if log_every > 0 and step % log_every == 0:
                typer.echo(f"eval step={step} count={total}", err=True)

    vqa_rate = vqa_acc / total if total else 0.0
    anls_rate = anls / total if total else 0.0
    em_rate = em / total if total else 0.0
    return {
        "avg_loss": float("nan"),
        "exact_match": em_rate,
        "vqa_accuracy": vqa_rate,
        "anls": anls_rate,
        "count": float(total),
    }


def run_eval(
    *,
    ckpt: str,
    batch_size: int | None,
    max_samples: int | None,
    override: list[str],
    log_every: int,
    eval_mode: Literal["teacher", "generate"],
) -> tuple[dict[str, float], str]:
    commit = get_git_commit()
    ckpt_meta = _load_ckpt_meta(ckpt)
    stage = str(ckpt_meta.get("stage", ""))
    if stage not in ("projector", "lora", "qlora", "full_ft"):
        raise ValueError(f"Checkpoint stage is unknown or missing: {stage!r}")
    raw_cfg = ckpt_meta.get("run_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("Checkpoint missing run_config; cannot run eval without config.")
    raw_cfg = _apply_overrides(raw_cfg, override)
    rc = RunConfig.from_dict(raw_cfg)
    base_run_name = raw_cfg["project"]["run_name"]  # pyright: ignore[reportIndexIssue]
    if not isinstance(base_run_name, str) or not base_run_name:
        raise ValueError("project.run_name must be a non-empty string in checkpoint config")

    qlora_enable = stage == "qlora"
    device = get_device(rc.train.device, force_cuda=qlora_enable)

    mcfg = LlenaModelConfig(
        llm_name=rc.model.llm_name,
        vision_name=rc.model.vision_name,
        num_image_tokens=rc.mm.num_image_tokens,
        projector=rc.mm.projector,
        gradient_checkpointing=False,
        freeze_vision=True,
        freeze_llm=True,
        peft_enable=stage in {"lora", "qlora"},
        peft_r=rc.train.lora_r or 0,
        peft_alpha=rc.train.lora_alpha or 0,
        peft_dropout=rc.train.lora_dropout or 0.0,
        peft_target_modules=list(rc.train.lora_targets or ()),
        qlora_enable=qlora_enable,
        device="cuda" if device.type == "cuda" else "cpu",
    )

    typer.echo("eval: building model")
    model = LlenaModel(mcfg)

    typer.echo("eval: loading checkpoint")
    load_eval_ckpt(ckpt, model, device)

    run_id: str | None = None
    if rc.logging.backend == "wandb":
        run_id_val = ckpt_meta.get("wandb_run_id")
        project_val = ckpt_meta.get("wandb_project")
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

    # autocast precision on CUDA
    use_amp = device.type == "cuda" and rc.train.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if rc.train.precision == "bf16" else torch.float16

    typer.echo("eval: loading image processor")
    image_proc = SiglipImageProcessor.from_pretrained(rc.model.vision_name)
    if batch_size is None:
        batch_size = rc.eval.batch_size
    if max_samples is None:
        max_samples = rc.train.eval_max_samples if rc.train.eval_max_samples > 0 else rc.eval.max_samples
        typer.echo(f"eval: max_samples from config = {max_samples}")

    typer.echo(f"eval: building dataset split={rc.data.split} max_samples={max_samples}")
    ds = build_dataset(rc, max_samples=max_samples)

    if eval_mode == "teacher":
        collator = LlenaCollator(
            tokenizer=model.tokenizer,
            image_processor=image_proc,
            max_seq_len=rc.train.max_seq_len,
            num_image_tokens=rc.mm.num_image_tokens,
            pad_to_multiple_of=8,
        )

        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: _collate_eval_teacher(collator, b),
            num_workers=0,
        )

        typer.echo("eval: starting eval loop (teacher)")
        metrics = eval_loop(
            model,
            dl,
            device,
            rc.data.dataset,
            log_every,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: _collate_eval_generate(image_proc, b),
            num_workers=0,
        )

        typer.echo("eval: starting eval loop (generate)")
        metrics = eval_loop_generate(
            model,
            dl,
            device,
            rc.data.dataset,
            log_every,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    ckpt_step = int(ckpt_meta["step"])  # pyright: ignore[reportArgumentType]
    if rc.logging.backend == "wandb" and run_id is not None:
        payload = {f"eval/{k}": v for k, v in metrics.items() if k != "count"}
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
    ckpt: str = opt(..., "Path to a ckpt.pt or a step_* directory"),
    batch_size: int | None = opt(None, "Eval batch size (None = config)"),
    max_samples: int | None = opt(None, "Limit number of samples (None = config)"),
    dataset: str | None = opt(None, "Override data.dataset for eval"),
    split: str | None = opt(None, "Override data.split for eval"),
    override: list[str] = opt([], "Config override(s): KEY=VALUE (repeatable)"),
    log_every: int = opt(100, "Log progress every N batches (0 disables)"),
    eval_mode: Literal["teacher", "generate"] = opt("generate", "Eval mode: generate | teacher"),
) -> None:
    load_dotenv()
    typer.echo("eval: starting", err=True)
    if dataset is not None:
        override.append(f"data.dataset={dataset}")
    if split is not None:
        override.append(f"data.split={split}")
    metrics, dataset = run_eval(
        ckpt=ckpt,
        batch_size=batch_size,
        max_samples=max_samples,
        override=override,
        log_every=log_every,
        eval_mode=eval_mode,
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
