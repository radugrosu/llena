import math

import pytest
import torch
from transformers import SiglipImageProcessor

from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator
from mm.model import LlenaModel, LlenaModelConfig


def _is_unsupported_cuda_arch_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = (
        "no kernel image is available",
        "not compatible with your gpu",
        "not supported on this architecture",
        "cuda error: invalid device function",
        "compute capability",
    )
    return any(m in msg for m in markers)


def _cuda_sm_major() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        major, _minor = torch.cuda.get_device_capability()
        return major
    except RuntimeError:
        return None


def _make_batch(model: LlenaModel, device: torch.device) -> dict[str, torch.Tensor]:
    proc = SiglipImageProcessor.from_pretrained(model.cfg.vision_name)
    ds = SyntheticVQADataset(num_samples=2, image_size=224, seed=7)
    batch = [ds[0], ds[1]]
    collator = LlenaCollator(
        tokenizer=model.tokenizer,
        image_processor=proc,
        max_seq_len=64,
        num_image_tokens=model.cfg.num_image_tokens,
        pad_to_multiple_of=None,
    )
    out = collator(batch)
    return {k: v.to(device) for k, v in out.items() if torch.is_tensor(v)}


@torch.no_grad()
def test_peft_lora_forward_cpu() -> None:
    cfg = LlenaModelConfig(
        llm_name="Qwen/Qwen2.5-0.5B-Instruct",
        vision_name="google/siglip-base-patch16-224",
        num_image_tokens=8,
        projector="mlp2",
        freeze_vision=True,
        freeze_llm=False,
        gradient_checkpointing=False,
        device="cpu",
        peft_enable=True,
        peft_r=8,
        peft_alpha=16,
        peft_dropout=0.0,
        qlora_enable=False,
    )
    model = LlenaModel(cfg)
    model.eval()

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(n.startswith("projector.") or "lora" in n for n in trainable)

    batch = _make_batch(model, torch.device("cpu"))
    out = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        mm_attention_mask=batch["mm_attention_mask"],
        mm_labels=batch["mm_labels"],
    )
    assert math.isfinite(float(out.loss))


@torch.no_grad()
def test_peft_qlora_forward_cuda() -> None:
    sm_major = _cuda_sm_major()
    if sm_major is None:
        pytest.skip("CUDA is required for QLoRA test.")
    if sm_major < 7:
        pytest.skip("QLoRA test requires a newer CUDA architecture (SM70+).")
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        pytest.skip("bitsandbytes is required for QLoRA test.")

    cfg = LlenaModelConfig(
        llm_name="Qwen/Qwen2.5-0.5B-Instruct",
        vision_name="google/siglip-base-patch16-224",
        num_image_tokens=8,
        projector="mlp2",
        freeze_vision=True,
        freeze_llm=False,
        gradient_checkpointing=False,
        device="cuda",
        peft_enable=True,
        peft_r=8,
        peft_alpha=16,
        peft_dropout=0.0,
        qlora_enable=True,
    )
    try:
        model = LlenaModel(cfg)
        model.eval()

        batch = _make_batch(model, torch.device("cuda"))
        out = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            mm_attention_mask=batch["mm_attention_mask"],
            mm_labels=batch["mm_labels"],
        )
    except RuntimeError as exc:
        if _is_unsupported_cuda_arch_error(exc):
            pytest.skip(f"Skipping QLoRA on unsupported CUDA architecture: {exc}")
        raise
    assert math.isfinite(float(out.loss))
