from dataclasses import dataclass

import torch
from transformers import SiglipImageProcessor

from mm.types import ChatTokenizer, VQASample


def _ensure_pad_token(tokenizer: ChatTokenizer) -> int:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("pad_token_id is still None after setting pad_token.")
    return int(tokenizer.pad_token_id)


def _encode_chat_sft(
    tokenizer: ChatTokenizer,
    question: str,
    answer: str,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """
    Qwen2.5-Instruct path only:
    - prompt_ids: user msg + assistant header (generation prompt)
    - full_ids: user msg + assistant msg (answer)
    - labels mask prompt tokens as -100
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        add_generation_prompt=False,
        tokenize=True,
        return_tensors=None,
    )

    if max_len <= 0:
        raise ValueError("max_len must be > 0")
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    if len(full_ids) <= max_len:
        return full_ids, labels
    return full_ids[:max_len], labels[:max_len]


def _pad_batch_1d(
    sequences: list[list[int]],
    pad_value: int,
    pad_to_multiple_of: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(len(s) == 0 for s in sequences):
        raise ValueError("Encountered empty sequence in batch.")
    max_len = max((len(s) for s in sequences), default=0)
    if (
        pad_to_multiple_of is not None
        and pad_to_multiple_of > 0
        and max_len % pad_to_multiple_of != 0
    ):
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    batch = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    attn = torch.zeros((len(sequences), max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        t = len(seq)
        batch[i, :t] = torch.tensor(seq, dtype=torch.long)
        attn[i, :t] = 1

    return batch, attn


def prepend_mm_prefix(
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    num_image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_image_tokens <= 0:
        return labels, attention_mask

    bsz = labels.size(0)
    prefix_labels = torch.full(
        (bsz, num_image_tokens), -100, dtype=labels.dtype, device=labels.device
    )
    prefix_attn = torch.ones(
        (bsz, num_image_tokens),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    mm_labels = torch.cat([prefix_labels, labels], dim=1)
    mm_attention_mask = torch.cat([prefix_attn, attention_mask], dim=1)
    return mm_labels, mm_attention_mask


@dataclass
class LlenaCollator:
    tokenizer: ChatTokenizer
    image_processor: SiglipImageProcessor
    max_seq_len: int
    num_image_tokens: int = 256
    pad_to_multiple_of: int | None = 8

    def __call__(self, batch: list[VQASample]) -> dict[str, torch.Tensor]:
        pad_id = _ensure_pad_token(self.tokenizer)

        images = [ex["image"] for ex in batch]
        vision = self.image_processor(images=images, return_tensors="pt")
        pixel_values = vision["pixel_values"]

        input_id_seqs: list[list[int]] = []
        label_seqs: list[list[int]] = []

        for ex in batch:
            full_ids, labels = _encode_chat_sft(
                self.tokenizer,
                ex["question"],
                ex["answer"],
                self.max_seq_len,
            )
            input_id_seqs.append(full_ids)
            label_seqs.append(labels)

        input_ids, attention_mask = _pad_batch_1d(
            input_id_seqs,
            pad_value=pad_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        labels_t, _ = _pad_batch_1d(
            label_seqs,
            pad_value=-100,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels_t = labels_t.to(torch.long)

        mm_labels, mm_attention_mask = prepend_mm_prefix(
            labels_t, attention_mask, self.num_image_tokens
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_t,
            "mm_labels": mm_labels,
            "mm_attention_mask": mm_attention_mask,
        }
