# mm/run_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Stage = Literal["smoke", "projector", "peft_lora", "peft_qlora"]


@dataclass(frozen=True)
class ModelConfig:
    llm_name: str
    vision_name: str


@dataclass(frozen=True)
class MmConfig:
    num_image_tokens: int
    projector: Literal["mlp2"]


@dataclass(frozen=True)
class DataConfig:
    num_samples: int
    image_size: int


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: Literal["auto", "cpu", "cuda"]

    micro_batch_size: int
    grad_accum_steps: int
    max_seq_len: int

    log_every: int
    save_every: int
    max_grad_norm: float

    gradient_checkpointing: bool

    # Learning rates
    lr_projector: float
    lr_lora: float

    # LoRA config (Qwen2.5 targets)
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]


@dataclass(frozen=True)
class PathsConfig:
    artifacts_dir: str
    reports_dir: str


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    mm: MmConfig
    data: DataConfig
    train: TrainConfig
    paths: PathsConfig

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        # --- Model ---
        model_d = d["model"]
        model = ModelConfig(
            llm_name=str(model_d["llm_name"]),
            vision_name=str(model_d["vision_name"]),
        )

        # --- MM ---
        mm_d = d["mm"]
        projector = str(mm_d["projector"])
        if projector != "mlp2":
            raise ValueError(f"Only projector=mlp2 is supported, got: {projector}")

        mm = MmConfig(
            num_image_tokens=int(mm_d["num_image_tokens"]),
            projector="mlp2",
        )

        # --- Data ---
        data_d = d["data"]
        data = DataConfig(
            num_samples=int(data_d["num_samples"]),
            image_size=int(data_d["image_size"]),
        )

        # --- Paths ---
        paths_d = d["paths"]
        paths = PathsConfig(
            artifacts_dir=str(paths_d["artifacts_dir"]),
            reports_dir=str(paths_d["reports_dir"]),
        )

        # --- Train ---
        train_d = d["train"]
        device = str(train_d.get("device", "auto")).lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                f"train.device must be one of auto|cpu|cuda, got: {device}"
            )

        # bf16-only policy: we don't accept config precision knobs here.
        # If you have 'precision' in yaml, we ignore it on purpose.

        lora_targets_raw = train_d["lora_targets"]
        if not isinstance(lora_targets_raw, list) or not all(
            isinstance(x, str) for x in lora_targets_raw
        ):
            raise ValueError("train.lora_targets must be a list[str]")

        # keep it specific to Qwen2.5 names
        allowed_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        for t in lora_targets_raw:
            if t not in allowed_targets:
                raise ValueError(
                    f"Unsupported LoRA target module: {t}. Allowed: {sorted(allowed_targets)}"
                )

        train = TrainConfig(
            seed=int(train_d["seed"]),
            device=device,  # type: ignore[arg-type]
            micro_batch_size=int(train_d["micro_batch_size"]),
            grad_accum_steps=int(train_d["grad_accum_steps"]),
            max_seq_len=int(train_d["max_seq_len"]),
            log_every=int(train_d["log_every"]),
            save_every=int(train_d["save_every"]),
            max_grad_norm=float(train_d["max_grad_norm"]),
            gradient_checkpointing=bool(train_d["gradient_checkpointing"]),
            lr_projector=float(train_d["lr_projector"]),
            lr_lora=float(train_d["lr_lora"]),
            lora_r=int(train_d["lora_r"]),
            lora_alpha=int(train_d["lora_alpha"]),
            lora_dropout=float(train_d["lora_dropout"]),
            lora_targets=tuple(lora_targets_raw),
        )

        return RunConfig(model=model, mm=mm, data=data, train=train, paths=paths)

    def effective_batch_size(self) -> int:
        return self.train.micro_batch_size * self.train.grad_accum_steps
