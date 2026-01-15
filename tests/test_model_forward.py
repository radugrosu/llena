import math
import torch
from transformers import AutoImageProcessor

from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator
from mm.model import LlenaModel, LlenaModelConfig


@torch.no_grad()
def test_model_forward_shapes_and_loss_finite() -> None:
    cfg = LlenaModelConfig(
        llm_name="Qwen/Qwen2.5-0.5B-Instruct",
        vision_name="google/siglip-base-patch16-224",
        num_image_tokens=32,
        projector="mlp2",
        freeze_vision=True,
        freeze_llm=True,
        gradient_checkpointing=False,
        device="cuda",
    )
    model = LlenaModel(cfg)
    model.eval()

    proc = AutoImageProcessor.from_pretrained(cfg.vision_name)
    ds = SyntheticVQADataset(num_samples=2, image_size=224, seed=1)
    batch = [ds[0], ds[1]]

    collator = LlenaCollator(
        tokenizer=model.tokenizer,
        image_processor=proc,
        max_seq_len=128,
        num_image_tokens=cfg.num_image_tokens,
        pad_to_multiple_of=None,
    )
    out = collator(batch)

    outputs = model(
        pixel_values=out["pixel_values"],
        input_ids=out["input_ids"],
        mm_attention_mask=out["mm_attention_mask"],
        mm_labels=out["mm_labels"],
    )

    assert hasattr(outputs, "loss")
    loss = float(outputs.loss)
    assert math.isfinite(loss)

    # logits length should match mm sequence length
    logits = outputs.logits
    assert logits.shape[1] == out["mm_attention_mask"].shape[1]
