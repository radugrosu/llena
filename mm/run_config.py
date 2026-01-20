# mm/run_config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


Stage = Literal["smoke", "projector", "peft_lora", "peft_qlora"]
DatasetName = Literal["synthetic", "docvqa", "textvqa", "sharegpt4v_coco"]
LogBackend = Literal["none", "wandb", "mlflow"]


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
    dataset: DatasetName
    data_dir: str
    split: str
    num_samples: int
    image_size: int


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: Literal["auto", "cpu", "cuda"]

    batch_size: int
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
class ProjectConfig:
    name: str
    run_name: str


@dataclass(frozen=True)
class LoggingConfig:
    backend: LogBackend
    wandb_project: str | None


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool
    max_samples: int
    batch_size: int


@dataclass(frozen=True)
class RunConfig:
    project: ProjectConfig
    model: ModelConfig
    mm: MmConfig
    data: DataConfig
    train: TrainConfig
    paths: PathsConfig
    logging: LoggingConfig
    eval: EvalConfig

    @staticmethod
    def from_dict(d: dict) -> "RunConfig":
        # --- Project ---
        project_d = d.get("project")
        if not isinstance(project_d, dict):
            raise ValueError("project must be a dict with name and run_name")
        name = project_d.get("name")
        run_name = project_d.get("run_name")
        if not isinstance(name, str) or not name:
            raise ValueError("project.name must be a non-empty string")
        if not isinstance(run_name, str) or not run_name:
            raise ValueError("project.run_name must be a non-empty string")
        project = ProjectConfig(name=name, run_name=run_name)

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
        dataset = str(data_d["dataset"]).lower()
        if dataset not in {"synthetic", "docvqa", "textvqa", "sharegpt4v_coco"}:
            raise ValueError(
                "data.dataset must be one of synthetic|docvqa|textvqa|sharegpt4v_coco, "
                f"got: {data_d['dataset']}"
            )
        data_dir = str(data_d.get("data_dir", "datasets/processed"))
        split = str(data_d.get("split", "train"))
        data = DataConfig(
            dataset=cast(DatasetName, dataset),
            data_dir=data_dir,
            split=split,
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
            raise ValueError(f"train.device must be one of auto|cpu|cuda, got: {device}")

        # bf16-only policy: we don't accept config precision knobs here.
        # If you have 'precision' in yaml, we ignore it on purpose.
        lora_targets_raw = train_d["lora_targets"]
        if not isinstance(lora_targets_raw, list) or not all(isinstance(x, str) for x in lora_targets_raw):
            raise ValueError("train.lora_targets must be a list[str]")

        # keep it specific to Qwen2.5 names
        allowed_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        for t in lora_targets_raw:
            if t not in allowed_targets:
                raise ValueError(f"Unsupported LoRA target module: {t}. Allowed: {sorted(allowed_targets)}")

        batch_size = int(train_d["batch_size"])
        micro_batch_size = int(train_d["micro_batch_size"])
        if batch_size <= 0 or micro_batch_size <= 0:
            raise ValueError("train.batch_size and train.micro_batch_size must be > 0")
        if batch_size % micro_batch_size != 0:
            raise ValueError("train.batch_size must be divisible by train.micro_batch_size")
        grad_accum_steps = batch_size // micro_batch_size

        train = TrainConfig(
            seed=int(train_d["seed"]),
            device=device,  # type: ignore[arg-type]
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
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

        # --- Logging ---
        logging_d = d.get("logging", {})
        backend = str(logging_d.get("backend", "none")).lower()
        if backend not in {"none", "wandb", "mlflow"}:
            raise ValueError(f"logging.backend must be one of none|wandb|mlflow, got: {backend}")
        logging_cfg = LoggingConfig(
            backend=cast(LogBackend, backend),
            wandb_project=(
                str(logging_d["wandb_project"])
                if "wandb_project" in logging_d and logging_d["wandb_project"] is not None
                else None
            ),
        )

        # --- Eval ---
        eval_d = d.get("eval")
        if not isinstance(eval_d, dict):
            raise ValueError("eval must be a dict with enabled, max_samples, batch_size")
        if "batch_size" not in eval_d:
            raise ValueError("eval.batch_size is required")
        eval_cfg = EvalConfig(
            enabled=bool(eval_d.get("enabled", True)),
            max_samples=int(eval_d.get("max_samples", 0)),
            batch_size=int(eval_d["batch_size"]),
        )

        return RunConfig(
            project=project,
            model=model,
            mm=mm,
            data=data,
            train=train,
            paths=paths,
            logging=logging_cfg,
            eval=eval_cfg,
        )

    def effective_batch_size(self) -> int:
        return self.train.micro_batch_size * self.train.grad_accum_steps
