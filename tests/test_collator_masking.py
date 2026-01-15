import torch
from transformers import AutoImageProcessor, AutoTokenizer

from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator


def test_collator_masks_prompt_and_mm_prefix() -> None:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vision_name = "google/siglip-base-patch16-224"

    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    proc = AutoImageProcessor.from_pretrained(vision_name)

    ds = SyntheticVQADataset(num_samples=4, image_size=224, seed=123)
    batch = [ds[i] for i in range(2)]

    collator = LlenaCollator(
        tokenizer=tok,
        image_processor=proc,
        max_seq_len=256,
        num_image_tokens=64,
        pad_to_multiple_of=None,
    )

    out = collator(batch)

    assert out["pixel_values"].dim() == 4
    assert out["input_ids"].dim() == 2
    assert out["labels"].shape == out["input_ids"].shape

    # mm prefix length
    assert out["mm_labels"].shape[1] == out["labels"].shape[1] + 64
    assert out["mm_attention_mask"].shape[1] == out["attention_mask"].shape[1] + 64

    # prefix labels should be -100, prefix attn should be 1
    prefix_labels = out["mm_labels"][:, :64]
    prefix_attn = out["mm_attention_mask"][:, :64]
    assert torch.all(prefix_labels == -100)
    assert torch.all(prefix_attn == 1)

    # At least some labels should be unmasked (answer tokens exist)
    assert torch.any(out["labels"] != -100)
