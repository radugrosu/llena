from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import typer
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor

from data.format import load_vqa_jsonl_dataset
from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator, LlenaPackedCollator
from mm.config import load_config
from mm.model import LlenaModel, LlenaModelConfig
from mm.run_config import RunConfig
from scripts import eval as eval_script
from scripts import train as train_script

ProbeMode = Literal["train", "val", "eval_teacher", "eval_generate"]


def opt(default: object, help: str):
    return typer.Option(default, help=help)


@dataclass(frozen=True)
class ProbeResult:
    mode: ProbeMode
    max_ok_batch: int
    tested_up_to: int
    recommended_batch: int
    note: str | None = None


def _clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    markers = [
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
    ]
    return any(m in msg for m in markers)


def _search_max_batch(
    *,
    try_batch: Callable[[int], bool],
    start_batch: int,
    max_batch: int,
) -> tuple[int, int]:
    if start_batch <= 0:
        raise ValueError(f"start_batch must be > 0, got: {start_batch}")
    if max_batch <= 0:
        raise ValueError(f"max_batch must be > 0, got: {max_batch}")
    if start_batch > max_batch:
        start_batch = max_batch

    tested_up_to = start_batch
    if not try_batch(start_batch):
        return 0, tested_up_to

    low = start_batch
    high = start_batch
    while high < max_batch:
        nxt = min(max_batch, high * 2)
        tested_up_to = max(tested_up_to, nxt)
        if not try_batch(nxt):
            high = nxt
            break
        low = nxt
        high = nxt
    else:
        return low, tested_up_to

    left = low + 1
    right = high - 1
    best = low
    while left <= right:
        mid = (left + right) // 2
        tested_up_to = max(tested_up_to, mid)
        if try_batch(mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1

    return best, tested_up_to


def _build_train_model(rc: RunConfig, device: torch.device) -> LlenaModel:
    stage = rc.train.stage_name
    qlora_enable = stage == "qlora"
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
    return model


def _build_eval_model(rc: RunConfig, device: torch.device) -> LlenaModel:
    stage = rc.train.stage_name
    qlora_enable = stage == "qlora"
    mcfg = LlenaModelConfig(
        llm_name=rc.model.llm_name,
        vision_name=rc.model.vision_name,
        num_image_tokens=rc.mm.num_image_tokens,
        projector=rc.mm.projector,
        gradient_checkpointing=False,
        precision=rc.train.precision,
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
    return LlenaModel(mcfg)


def _eval_dataset_name(default_name: str, override_name: str | None) -> str:
    if override_name is not None:
        return override_name
    if default_name in {"llava_instruct", "llava_textvqa"}:
        return "textvqa"
    return default_name


def _build_eval_dataset(
    *,
    rc: RunConfig,
    dataset_name: str,
    split: str,
    max_samples: int,
) -> Dataset:
    if dataset_name == "synthetic":
        return SyntheticVQADataset(
            num_samples=max_samples,
            image_size=224,
            seed=rc.train.seed,
        )
    return load_vqa_jsonl_dataset(
        dataset=dataset_name,  # type: ignore[arg-type]
        data_dir=Path(rc.data.data_dir),
        split=split,
        max_samples=max_samples,
    )


def _execute_probe_trial(
    *,
    mode: ProbeMode,
    batch_size: int,
    device: torch.device,
    run_trial: Callable[[int], None],
) -> bool:
    try:
        run_trial(batch_size)
        return True
    except Exception as exc:  # noqa: BLE001
        if is_oom_error(exc):
            typer.echo(f"probe[{mode}]: batch_size={batch_size} -> OOM")
            return False
        raise
    finally:
        _clear_device_cache(device)


def _recommend(max_ok_batch: int, safety_margin: float) -> int:
    if max_ok_batch <= 0:
        return 0
    return max(1, int(math.floor(max_ok_batch * safety_margin)))


def run_probe(
    *,
    config: str,
    out_json: str | None,
    max_batch: int,
    start_batch: int,
    probe_samples: int,
    safety_margin: float,
    eval_dataset: str | None,
    eval_split: str,
    eval_max_generated_tokens: int | None,
    ckpt: str | None,
    probe_train: bool,
    probe_val: bool,
    probe_eval_teacher: bool,
    probe_eval_generate: bool,
    override: list[str] | None = None,
) -> list[ProbeResult]:
    if safety_margin <= 0.0 or safety_margin > 1.0:
        raise ValueError(f"safety_margin must be in (0, 1], got: {safety_margin}")
    if probe_samples <= 0:
        raise ValueError(f"probe_samples must be > 0, got: {probe_samples}")

    overrides = override or []
    load_dotenv()
    raw_cfg = load_config(config, overrides=overrides)
    rc = RunConfig.from_dict(raw_cfg)

    qlora_enable = rc.train.stage_name == "qlora"
    device = train_script.get_device(rc.train.device, force_cuda=qlora_enable)
    use_amp = device.type == "cuda" and rc.train.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if rc.train.precision == "bf16" else torch.float16
    pad_to_multiple = 8

    run_any = probe_train or probe_val or probe_eval_teacher or probe_eval_generate
    if not run_any:
        raise ValueError("No probe modes selected.")

    sample_cap = max(probe_samples, max_batch)
    image_proc = SiglipImageProcessor.from_pretrained(rc.model.vision_name)
    results: list[ProbeResult] = []

    train_model: LlenaModel | None = None
    eval_model: LlenaModel | None = None
    train_collator: LlenaCollator | LlenaPackedCollator | None = None
    ds_eval: Dataset | None = None
    eval_upper = 0
    try:
        if probe_train or probe_val:
            typer.echo("probe: building training model")
            train_model = _build_train_model(rc, device)
            collator_cls = LlenaPackedCollator if rc.data.dataset in {"llava_instruct", "llava_textvqa"} else LlenaCollator
            train_collator = collator_cls(
                tokenizer=train_model.tokenizer,
                image_processor=image_proc,
                max_seq_len=rc.train.max_seq_len,
                num_image_tokens=rc.mm.num_image_tokens,
                pad_to_multiple_of=pad_to_multiple,
            )

        if probe_train and train_model is not None and train_collator is not None:
            typer.echo("probe[train]: preparing dataset")
            ds_train = train_script.build_dataset(rc, max_samples=sample_cap)
            train_upper = min(max_batch, len(ds_train))
            if train_upper <= 0:
                raise ValueError("probe[train]: dataset is empty.")
            if start_batch > train_upper:
                typer.echo(
                    f"probe[train]: start_batch={start_batch} > available={train_upper}; clamping to available size."
                )
            train_model.train()

            def train_trial(batch_size: int) -> None:
                dl = DataLoader(
                    ds_train,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=train_collator,
                    num_workers=0,
                )
                batch = next(iter(dl))
                batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                train_model.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = train_model(
                        pixel_values=batch_t["pixel_values"],
                        input_ids=batch_t["input_ids"],
                        mm_attention_mask=batch_t["mm_attention_mask"],
                        mm_labels=batch_t["mm_labels"],
                    )
                    loss = out.loss
                loss.backward()
                train_model.zero_grad(set_to_none=True)

            max_ok, tested_up_to = _search_max_batch(
                try_batch=lambda bs: _execute_probe_trial(
                    mode="train",
                    batch_size=bs,
                    device=device,
                    run_trial=train_trial,
                ),
                start_batch=min(start_batch, train_upper),
                max_batch=train_upper,
            )
            results.append(
                ProbeResult(
                    mode="train",
                    max_ok_batch=max_ok,
                    tested_up_to=tested_up_to,
                    recommended_batch=_recommend(max_ok, safety_margin),
                )
            )

        if probe_val and train_model is not None and train_collator is not None:
            typer.echo("probe[val]: preparing dataset")
            ds_val = train_script.build_dataset(rc, split="validation", max_samples=sample_cap)
            val_upper = min(max_batch, len(ds_val))
            if val_upper <= 0:
                raise ValueError("probe[val]: validation dataset is empty.")
            if start_batch > val_upper:
                typer.echo(f"probe[val]: start_batch={start_batch} > available={val_upper}; clamping to available size.")

            def val_trial(batch_size: int) -> None:
                dl = DataLoader(
                    ds_val,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=train_collator,
                    num_workers=0,
                )
                batch = next(iter(dl))
                batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                train_model.eval()
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = train_model(
                            pixel_values=batch_t["pixel_values"],
                            input_ids=batch_t["input_ids"],
                            mm_attention_mask=batch_t["mm_attention_mask"],
                            mm_labels=batch_t["mm_labels"],
                        )
                        _ = float(out.loss.item())
                train_model.train()

            max_ok, tested_up_to = _search_max_batch(
                try_batch=lambda bs: _execute_probe_trial(
                    mode="val",
                    batch_size=bs,
                    device=device,
                    run_trial=val_trial,
                ),
                start_batch=min(start_batch, val_upper),
                max_batch=val_upper,
            )
            results.append(
                ProbeResult(
                    mode="val",
                    max_ok_batch=max_ok,
                    tested_up_to=tested_up_to,
                    recommended_batch=_recommend(max_ok, safety_margin),
                )
            )

        if probe_eval_teacher or probe_eval_generate:
            typer.echo("probe[eval]: building eval model")
            eval_model = _build_eval_model(rc, device)
            if ckpt is not None:
                typer.echo(f"probe[eval]: loading checkpoint {ckpt}")
                eval_script.load_eval_ckpt(ckpt, eval_model, device)

            use_eval_dataset = _eval_dataset_name(rc.data.dataset, eval_dataset)
            if eval_dataset is None and rc.data.dataset in {"llava_instruct", "llava_textvqa"}:
                typer.echo(f"probe[eval]: dataset {rc.data.dataset} is instruct-style; using dataset={use_eval_dataset}.")
            ds_eval = _build_eval_dataset(
                rc=rc,
                dataset_name=use_eval_dataset,
                split=eval_split,
                max_samples=sample_cap,
            )
            eval_upper = min(max_batch, len(ds_eval))
            if eval_upper <= 0:
                raise ValueError("probe[eval]: eval dataset is empty.")
            if start_batch > eval_upper:
                typer.echo(
                    f"probe[eval]: start_batch={start_batch} > available={eval_upper}; clamping to available size."
                )

        if probe_eval_teacher and eval_model is not None and ds_eval is not None:
            eval_collator = LlenaCollator(
                tokenizer=eval_model.tokenizer,
                image_processor=image_proc,
                max_seq_len=rc.train.max_seq_len,
                num_image_tokens=rc.mm.num_image_tokens,
                pad_to_multiple_of=pad_to_multiple,
            )

            def eval_teacher_trial(batch_size: int) -> None:
                dl = DataLoader(
                    ds_eval,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda b: eval_script._collate_eval_teacher(eval_collator, b),
                    num_workers=0,
                )
                batch, _answers = next(iter(dl))
                batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                eval_model.eval()
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = eval_model(
                            pixel_values=batch_t["pixel_values"],
                            input_ids=batch_t["input_ids"],
                            mm_attention_mask=batch_t["mm_attention_mask"],
                            mm_labels=batch_t["mm_labels"],
                        )
                        _ = float(out.loss.item())

            max_ok, tested_up_to = _search_max_batch(
                try_batch=lambda bs: _execute_probe_trial(
                    mode="eval_teacher",
                    batch_size=bs,
                    device=device,
                    run_trial=eval_teacher_trial,
                ),
                start_batch=min(start_batch, eval_upper),
                max_batch=eval_upper,
            )
            results.append(
                ProbeResult(
                    mode="eval_teacher",
                    max_ok_batch=max_ok,
                    tested_up_to=tested_up_to,
                    recommended_batch=_recommend(max_ok, safety_margin),
                )
            )

        if probe_eval_generate and eval_model is not None and ds_eval is not None:
            max_gen_tokens = eval_max_generated_tokens or rc.eval.max_generated_tokens
            tokenizer = eval_model.tokenizer
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad_id is None:
                raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")

            def eval_generate_trial(batch_size: int) -> None:
                dl = DataLoader(
                    ds_eval,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda b: eval_script._collate_eval_generate(image_proc, b),
                    num_workers=0,
                )
                images, questions, _answers = next(iter(dl))
                eval_model.eval()
                with torch.no_grad():
                    _ = eval_script._generate_batch(
                        eval_model,
                        images,
                        questions,
                        device=device,
                        tokenizer=tokenizer,
                        pad_id=int(pad_id),
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        max_generated_tokens=max_gen_tokens,
                        repetition_penalty=1.0,
                    )

            max_ok, tested_up_to = _search_max_batch(
                try_batch=lambda bs: _execute_probe_trial(
                    mode="eval_generate",
                    batch_size=bs,
                    device=device,
                    run_trial=eval_generate_trial,
                ),
                start_batch=min(start_batch, eval_upper),
                max_batch=eval_upper,
            )
            results.append(
                ProbeResult(
                    mode="eval_generate",
                    max_ok_batch=max_ok,
                    tested_up_to=tested_up_to,
                    recommended_batch=_recommend(max_ok, safety_margin),
                    note=f"max_generated_tokens={max_gen_tokens}",
                )
            )
    finally:
        del train_model
        del eval_model
        _clear_device_cache(device)

    typer.echo("\nBatch size probe results")
    for item in results:
        note = f" ({item.note})" if item.note else ""
        typer.echo(
            f"- {item.mode}: max_ok={item.max_ok_batch} recommended={item.recommended_batch} "
            f"tested_up_to={item.tested_up_to}{note}"
        )

    as_dict = [
        {
            "mode": r.mode,
            "max_ok_batch": r.max_ok_batch,
            "recommended_batch": r.recommended_batch,
            "tested_up_to": r.tested_up_to,
            **({"note": r.note} if r.note else {}),
        }
        for r in results
    ]
    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": as_dict}, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"probe: wrote {out_path}")
    return results


def main(
    config: str = opt(..., "Path to training config YAML."),
    out_json: str | None = opt(None, "Optional output JSON path."),
    max_batch: int = opt(256, "Upper bound to probe per mode."),
    start_batch: int = opt(1, "Initial batch size to test."),
    probe_samples: int = opt(256, "Number of dataset samples to make available for probing."),
    safety_margin: float = opt(0.9, "Recommended batch = floor(max_ok * safety_margin)."),
    eval_dataset: str | None = opt(None, "Eval dataset override (default auto-maps instruct datasets to textvqa)."),
    eval_split: str = opt("validation", "Eval split for eval-teacher/eval-generate probes."),
    eval_max_generated_tokens: int | None = opt(None, "Override generated tokens for eval-generate probing."),
    ckpt: str | None = opt(None, "Optional checkpoint path for eval model probing."),
    probe_train: bool = opt(True, "Probe train micro-batch size (forward+backward)."),
    probe_val: bool = opt(True, "Probe validation batch size (teacher-forced forward)."),
    probe_eval_teacher: bool = opt(True, "Probe eval teacher batch size."),
    probe_eval_generate: bool = opt(True, "Probe eval generate batch size."),
    override: list[str] = opt([], "Config override(s): KEY=VALUE (repeatable)."),
) -> None:
    run_probe(
        config=config,
        out_json=out_json,
        max_batch=max_batch,
        start_batch=start_batch,
        probe_samples=probe_samples,
        safety_margin=safety_margin,
        eval_dataset=eval_dataset,
        eval_split=eval_split,
        eval_max_generated_tokens=eval_max_generated_tokens,
        ckpt=ckpt,
        probe_train=probe_train,
        probe_val=probe_val,
        probe_eval_teacher=probe_eval_teacher,
        probe_eval_generate=probe_eval_generate,
        override=override,
    )


if __name__ == "__main__":
    typer.run(main)
