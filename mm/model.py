from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2ForCausalLM,
    SiglipVisionModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from mm.projectors import build_projector
from mm.types import ChatTokenizer


def _select_or_pad_tokens(tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    bsz, n, d = tokens.shape
    if n >= num_tokens:
        return tokens[:, :num_tokens, :]
    pad = tokens[:, -1:, :].expand(bsz, num_tokens - n, d)
    return torch.cat([tokens, pad], dim=1)


@dataclass
class LlenaModelConfig:
    llm_name: str
    vision_name: str
    num_image_tokens: int
    projector: Literal["mlp2"]
    device: Literal["cpu", "cuda"]
    gradient_checkpointing: bool
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_implementation: str | None = None
    freeze_vision: bool = True
    freeze_llm: bool = False
    peft_enable: bool = False
    peft_r: int = 16
    peft_alpha: int = 32
    peft_dropout: float = 0.05
    peft_target_modules: list[str] | None = None
    qlora_enable: bool = False


class LlenaModel(nn.Module):
    """
    Specific stack:
      - Vision: SiglipVisionModel
      - LLM: Qwen2ForCausalLM (Qwen2.5 family)
      - Optional: PEFT LoRA / QLoRA
    """

    tokenizer: ChatTokenizer
    vision: SiglipVisionModel
    llm: Qwen2ForCausalLM | PeftModel
    projector: nn.Module

    @staticmethod
    def _resolve_dtype(
        *,
        device: Literal["cpu", "cuda"],
        precision: Literal["bf16", "fp16", "fp32"],
    ) -> torch.dtype:
        if device == "cpu":
            return torch.float32
        if precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("precision='bf16' requested but CUDA BF16 is not supported on this device.")
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision == "fp32":
            return torch.float32

    def __init__(self, cfg: LlenaModelConfig):
        super().__init__()
        target_device = torch.device("cuda" if cfg.device == "cuda" else "cpu")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")

        self.cfg = cfg
        self.freeze_vision = bool(cfg.freeze_vision)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.vision = SiglipVisionModel.from_pretrained(
            cfg.vision_name,
            attn_implementation=cfg.attn_implementation,
        )

        if cfg.qlora_enable:
            if cfg.device != "cuda":
                raise ValueError("QLoRA requires device='cuda'")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for QLoRA but not available.")

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._resolve_dtype(device="cuda", precision=cfg.precision),
            )
            llm = Qwen2ForCausalLM.from_pretrained(
                cfg.llm_name,
                quantization_config=bnb,
                device_map={"": 0},
                attn_implementation=cfg.attn_implementation,
            )
            llm = prepare_model_for_kbit_training(
                llm,
                use_gradient_checkpointing=cfg.gradient_checkpointing,
            )
            self.llm = llm
        else:
            llm_dtype = self._resolve_dtype(device=cfg.device, precision=cfg.precision)
            llm = Qwen2ForCausalLM.from_pretrained(
                cfg.llm_name,
                torch_dtype=llm_dtype,
                attn_implementation=cfg.attn_implementation,
            )
            if cfg.gradient_checkpointing:
                llm.gradient_checkpointing_enable()
                llm.config.use_cache = False
            self.llm = llm

        if cfg.peft_enable:
            if cfg.peft_target_modules is None:
                cfg.peft_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            for p in self.llm.parameters():
                p.requires_grad = False

            lora_cfg = LoraConfig(
                r=cfg.peft_r,
                lora_alpha=cfg.peft_alpha,
                lora_dropout=cfg.peft_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=cfg.peft_target_modules,
            )
            self.llm = get_peft_model(self.llm, lora_cfg)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

        # Projector: match dtype to LLM embedding weights
        vision_dim = int(self.vision.config.hidden_size)
        llm_dim = int(self.llm.config.hidden_size)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        self.projector = build_projector(cfg.projector, vision_dim=vision_dim, text_dim=llm_dim)

        self.num_image_tokens = int(cfg.num_image_tokens)

        # Freeze vision
        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
            self.vision.eval()

        # Freeze LLM only if NOT using PEFT
        if cfg.freeze_llm and not cfg.peft_enable:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure projector dtype matches LLM embed dtype (important!)
        embed_dtype = self.llm.get_input_embeddings().weight.dtype  # pyright: ignore[reportCallIssue]
        self.projector.to(dtype=embed_dtype)  # pyright: ignore[reportCallIssue, reportArgumentType]

        if cfg.qlora_enable:
            cast(nn.Module, self.vision).to(target_device)
            self.projector.to(target_device)
        else:
            self.to(target_device)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        mm_attention_mask: torch.Tensor,
        mm_labels: torch.Tensor,
    ) -> CausalLMOutputWithPast:
        v_out = self.vision(pixel_values=pixel_values)
        v_tokens = v_out.last_hidden_state
        if self.freeze_vision:
            v_tokens = v_tokens.detach()

        v_tokens = _select_or_pad_tokens(v_tokens, self.num_image_tokens)
        proj_dtype = next(self.projector.parameters()).dtype
        v_tokens = v_tokens.to(dtype=proj_dtype)
        img_embeds = self.projector(v_tokens)

        text_embeds = self.llm.get_input_embeddings()(input_ids)  # pyright: ignore[reportCallIssue]
        if img_embeds.dtype != text_embeds.dtype:
            img_embeds = img_embeds.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=mm_attention_mask,
            labels=mm_labels,
        )
        return out  # type: ignore[return-value]
