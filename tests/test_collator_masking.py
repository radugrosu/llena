import torch
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor

from data.synthetic import SyntheticVQADataset
from mm.collator import LlenaCollator, LlenaPackedCollator, _encode_chat_packed, _encode_chat_sft
from mm.types import ChatMessage, InstructSample


def test_collator_masks_prompt_and_mm_prefix() -> None:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vision_name = "google/siglip-base-patch16-224"

    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    proc = SiglipImageProcessor.from_pretrained(vision_name)

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
    text_labels = out["mm_labels"][:, 64:]
    assert text_labels.shape == out["input_ids"].shape

    # mm prefix length
    assert out["mm_labels"].shape[1] == text_labels.shape[1] + 64
    assert out["mm_attention_mask"].shape[1] == out["attention_mask"].shape[1] + 64

    # prefix labels should be -100, prefix attn should be 1
    prefix_labels = out["mm_labels"][:, :64]
    prefix_attn = out["mm_attention_mask"][:, :64]
    assert torch.all(prefix_labels == -100)
    assert torch.all(prefix_attn == 1)

    # At least some labels should be unmasked (answer tokens exist)
    assert torch.any(text_labels != -100)


def test_packed_collator_masks_assistant_turns() -> None:
    llm_name = "Qwen/Qwen2.5-0.5B-Instruct"
    vision_name = "google/siglip-base-patch16-224"

    tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
    proc = SiglipImageProcessor.from_pretrained(vision_name)

    img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    conversation: list[ChatMessage] = [
        {"role": "user", "content": "What color is the ball?"},
        {"role": "assistant", "content": "Red."},
        {"role": "user", "content": "And what shape is it?"},
        {"role": "assistant", "content": "Circle."},
    ]
    batch: list[InstructSample] = [{"image": img, "conversation": conversation}]

    collator = LlenaPackedCollator(
        tokenizer=tok,
        image_processor=proc,
        max_seq_len=256,
        num_image_tokens=32,
        pad_to_multiple_of=None,
    )

    out = collator(batch)

    labels = out["mm_labels"][0, 32:]
    input_ids = out["input_ids"][0]
    assert torch.any(labels != -100)

    kept = input_ids[labels != -100].tolist()
    decoded = tok.decode(kept, skip_special_tokens=True).lower()
    assert "red" in decoded
    assert "circle" in decoded


def test_encode_chat_sft_matches_packed_for_single_turn() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    question = "What is shown in the image?"
    answer = "A red ball."
    max_len = 256

    sft_ids, sft_labels = _encode_chat_sft(tok, question, answer, max_len)
    packed_ids, packed_labels = _encode_chat_packed(
        tok,
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        max_len,
    )

    assert sft_ids == packed_ids
    assert sft_labels == packed_labels
