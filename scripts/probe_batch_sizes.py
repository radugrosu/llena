from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal, Sized, cast

import torch
import typer
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor

from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator
from mm.config import load_config
from mm.model import LlenaModel, LlenaModelConfig
from mm.run_config import RunConfig, derive_vision_params
from mm.types import ChatTokenizer
from scripts import eval as eval_script
from scripts import train as train_script

ProbeMode = Literal["train", "validation", "eval_teacher", "eval_generate"]
EvalMode = Literal["teacher", "generate"]

DEFAULT_MAX_BATCH = 256
DEFAULT_START_BATCH = 1
DEFAULT_PROBE_SAMPLES = 256
DEFAULT_SAFETY_MARGIN = 0.9


def opt(default: object, help: str):
    return typer.Option(default, help=help)


@dataclass(frozen=True)
class ProbeResult:
    mode: ProbeMode
    max_ok_batch: int
    tested_up_to: int
    recommended_batch: int
    gradient_checkpointing: bool
    liger_kernel: bool
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
    except Exception as exc:
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


def _prepare_bounds(*, mode: ProbeMode, dataset: Dataset, start_batch: int, max_batch: int) -> tuple[int, int]:
    upper = min(max_batch, len(cast(Sized, dataset)))
    if upper <= 0:
        raise ValueError(f"probe[{mode}]: dataset is empty.")
    if start_batch > upper:
        typer.echo(f"probe[{mode}]: start_batch={start_batch} > available={upper}; clamping to available size.")
    return min(start_batch, upper), upper


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
        liger_kernel=rc.train.liger_kernel,
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
        liger_kernel=rc.train.liger_kernel,
        device="cuda" if device.type == "cuda" else "cpu",
    )
    return LlenaModel(mcfg)


def _chat_prompt_len(tokenizer: ChatTokenizer, question: str) -> int:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        return_tensors=None,
    )
    return len(ids)


def _build_full_sequence_question(tokenizer: ChatTokenizer, max_seq_len: int) -> str:
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got: {max_seq_len}")

    base = "Describe the visual scene precisely and explain the rectangle color."
    filler = " detail"
    if _chat_prompt_len(tokenizer, base) >= max_seq_len:
        return base

    low = 0
    high = 1
    while _chat_prompt_len(tokenizer, f"{base}{filler * high}") < max_seq_len:
        low = high
        high *= 2
        if high > 2**20:
            raise RuntimeError(f"Unable to build a full-sequence probe prompt for max_seq_len={max_seq_len}")

    while low + 1 < high:
        mid = (low + high) // 2
        if _chat_prompt_len(tokenizer, f"{base}{filler * mid}") >= max_seq_len:
            high = mid
        else:
            low = mid

    return f"{base}{filler * high}"


def _build_probe_dataset(
    *,
    tokenizer: ChatTokenizer,
    max_seq_len: int,
    num_samples: int,
    seed: int,
    image_size: int,
) -> Dataset:
    question = _build_full_sequence_question(tokenizer, max_seq_len)
    return SyntheticVQADataset(
        num_samples=num_samples,
        image_size=image_size,
        seed=seed,
        fixed_question=question,
    )


def decide_best_training_batch_size(
    *,
    rc: RunConfig,
    model: LlenaModel,
    collator: LlenaCollator,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batch: int,
    start_batch: int,
    safety_margin: float,
    gradient_checkpointing: bool,
    liger_kernel: bool,
) -> ProbeResult:
    start, upper = _prepare_bounds(mode="train", dataset=dataset, start_batch=start_batch, max_batch=max_batch)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("probe[train]: model has no trainable parameters.")

    def train_trial(batch_size: int) -> None:
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        batch = next(iter(dl))
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                pixel_values=batch_t["pixel_values"],
                input_ids=batch_t["input_ids"],
                mm_attention_mask=batch_t["mm_attention_mask"],
                mm_labels=batch_t["mm_labels"],
            )
            loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    max_ok, tested_up_to = _search_max_batch(
        try_batch=lambda bs: _execute_probe_trial(
            mode="train",
            batch_size=bs,
            device=device,
            run_trial=train_trial,
        ),
        start_batch=start,
        max_batch=upper,
    )
    return ProbeResult(
        mode="train",
        max_ok_batch=max_ok,
        tested_up_to=tested_up_to,
        recommended_batch=_recommend(max_ok, safety_margin),
        gradient_checkpointing=gradient_checkpointing,
        liger_kernel=liger_kernel,
        note=f"stage={rc.train.stage_name}",
    )


def decide_best_validation_batch_size(
    *,
    model: LlenaModel,
    collator: LlenaCollator,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batch: int,
    start_batch: int,
    safety_margin: float,
    gradient_checkpointing: bool,
    liger_kernel: bool,
) -> ProbeResult:
    start, upper = _prepare_bounds(mode="validation", dataset=dataset, start_batch=start_batch, max_batch=max_batch)

    def val_trial(batch_size: int) -> None:
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        batch = next(iter(dl))
        batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    pixel_values=batch_t["pixel_values"],
                    input_ids=batch_t["input_ids"],
                    mm_attention_mask=batch_t["mm_attention_mask"],
                    mm_labels=batch_t["mm_labels"],
                )
                _ = float(out.loss.item())
        model.train(was_training)

    max_ok, tested_up_to = _search_max_batch(
        try_batch=lambda bs: _execute_probe_trial(
            mode="validation",
            batch_size=bs,
            device=device,
            run_trial=val_trial,
        ),
        start_batch=start,
        max_batch=upper,
    )
    return ProbeResult(
        mode="validation",
        max_ok_batch=max_ok,
        tested_up_to=tested_up_to,
        recommended_batch=_recommend(max_ok, safety_margin),
        gradient_checkpointing=gradient_checkpointing,
        liger_kernel=liger_kernel,
    )


def decide_best_evaluation_batch_size(
    *,
    mode: EvalMode,
    rc: RunConfig,
    model: LlenaModel,
    image_proc: SiglipImageProcessor,
    dataset: Dataset,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batch: int,
    start_batch: int,
    safety_margin: float,
    gradient_checkpointing: bool,
    liger_kernel: bool,
) -> ProbeResult:
    if mode == "teacher":
        probe_mode: ProbeMode = "eval_teacher"
    else:
        probe_mode = "eval_generate"

    start, upper = _prepare_bounds(mode=probe_mode, dataset=dataset, start_batch=start_batch, max_batch=max_batch)

    if mode == "teacher":
        collator = LlenaCollator(
            tokenizer=model.tokenizer,
            image_processor=image_proc,
            max_seq_len=rc.train.max_seq_len,
            num_image_tokens=rc.mm.num_image_tokens,
            pad_to_multiple_of=8,
        )

        def eval_teacher_trial(batch_size: int) -> None:
            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: eval_script._collate_eval_teacher(collator, b),
                num_workers=0,
            )
            batch, _answers = next(iter(dl))
            batch_t = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(
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
            start_batch=start,
            max_batch=upper,
        )
        return ProbeResult(
            mode="eval_teacher",
            max_ok_batch=max_ok,
            tested_up_to=tested_up_to,
            recommended_batch=_recommend(max_ok, safety_margin),
            gradient_checkpointing=gradient_checkpointing,
            liger_kernel=liger_kernel,
        )

    max_gen_tokens = rc.eval.max_generated_tokens
    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")

    def eval_generate_trial(batch_size: int) -> None:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: eval_script._collate_eval_generate(image_proc, b),
            num_workers=0,
        )
        images, questions, _answers = next(iter(dl))
        model.eval()
        with torch.no_grad():
            _ = eval_script._generate_batch(
                model,
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
        start_batch=start,
        max_batch=upper,
    )
    return ProbeResult(
        mode="eval_generate",
        max_ok_batch=max_ok,
        tested_up_to=tested_up_to,
        recommended_batch=_recommend(max_ok, safety_margin),
        gradient_checkpointing=gradient_checkpointing,
        liger_kernel=liger_kernel,
        note=f"max_generated_tokens={max_gen_tokens}",
    )


def run_probe(
    *,
    config_path: str,
    probe_mode: ProbeMode,
    gradient_checkpointing: bool,
    liger_kernel: bool,
) -> ProbeResult:
    overrides = [
        f"train.gradient_checkpointing={'true' if gradient_checkpointing else 'false'}",
        f"train.liger_kernel={'true' if liger_kernel else 'false'}",
    ]

    load_dotenv()
    raw_cfg = load_config(config_path, overrides=overrides)
    rc = RunConfig.from_dict(raw_cfg)
    image_size, _ = derive_vision_params(rc.model.vision_name)

    qlora_enable = rc.train.stage_name == "qlora"
    device = train_script.get_device(rc.train.device, force_cuda=qlora_enable)
    use_amp = device.type == "cuda" and rc.train.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if rc.train.precision == "bf16" else torch.float16

    image_proc = SiglipImageProcessor.from_pretrained(rc.model.vision_name)

    train_model: LlenaModel | None = None
    eval_model: LlenaModel | None = None
    try:
        if probe_mode in {"train", "validation"}:
            typer.echo("probe: building training model")
            train_model = _build_train_model(rc, device)
            probe_ds = _build_probe_dataset(
                tokenizer=train_model.tokenizer,
                max_seq_len=rc.train.max_seq_len,
                num_samples=DEFAULT_PROBE_SAMPLES,
                seed=rc.train.seed,
                image_size=image_size,
            )
            train_collator = LlenaCollator(
                tokenizer=train_model.tokenizer,
                image_processor=image_proc,
                max_seq_len=rc.train.max_seq_len,
                num_image_tokens=rc.mm.num_image_tokens,
                pad_to_multiple_of=8,
            )

            if probe_mode == "train":
                result = decide_best_training_batch_size(
                    rc=rc,
                    model=train_model,
                    collator=train_collator,
                    dataset=probe_ds,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    max_batch=DEFAULT_MAX_BATCH,
                    start_batch=DEFAULT_START_BATCH,
                    safety_margin=DEFAULT_SAFETY_MARGIN,
                    gradient_checkpointing=gradient_checkpointing,
                    liger_kernel=liger_kernel,
                )
            else:
                result = decide_best_validation_batch_size(
                    model=train_model,
                    collator=train_collator,
                    dataset=probe_ds,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    max_batch=DEFAULT_MAX_BATCH,
                    start_batch=DEFAULT_START_BATCH,
                    safety_margin=DEFAULT_SAFETY_MARGIN,
                    gradient_checkpointing=gradient_checkpointing,
                    liger_kernel=liger_kernel,
                )

        else:
            typer.echo("probe: building eval model")
            eval_model = _build_eval_model(rc, device)
            probe_ds = _build_probe_dataset(
                tokenizer=eval_model.tokenizer,
                max_seq_len=rc.train.max_seq_len,
                num_samples=DEFAULT_PROBE_SAMPLES,
                seed=rc.train.seed,
                image_size=image_size,
            )

            eval_mode: EvalMode = "teacher" if probe_mode == "eval_teacher" else "generate"
            result = decide_best_evaluation_batch_size(
                mode=eval_mode,
                rc=rc,
                model=eval_model,
                image_proc=image_proc,
                dataset=probe_ds,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                max_batch=DEFAULT_MAX_BATCH,
                start_batch=DEFAULT_START_BATCH,
                safety_margin=DEFAULT_SAFETY_MARGIN,
                gradient_checkpointing=gradient_checkpointing,
                liger_kernel=liger_kernel,
            )
    finally:
        del train_model
        del eval_model
        _clear_device_cache(device)

    note = f" ({result.note})" if result.note else ""
    typer.echo(
        f"probe[{result.mode}]: max_ok={result.max_ok_batch} recommended={result.recommended_batch} "
        f"tested_up_to={result.tested_up_to} gradient_checkpointing={result.gradient_checkpointing} "
        f"liger_kernel={result.liger_kernel}{note}"
    )
    return result


def main(
    config_path: str = opt(..., "Path to training config YAML."),
    probe_mode: ProbeMode = opt("train", "Probe mode: train|validation|eval_teacher|eval_generate."),
    gradient_checkpointing: bool = opt(False, "Override train.gradient_checkpointing for this probe run."),
    liger_kernel: bool = opt(False, "Override train.liger_kernel for this probe run."),
) -> None:
    run_probe(
        config_path=config_path,
        probe_mode=probe_mode,
        gradient_checkpointing=gradient_checkpointing,
        liger_kernel=liger_kernel,
    )


if __name__ == "__main__":
    typer.run(main)
