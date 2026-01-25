# mm/run_config.py
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, cast


Stage = Literal["projector", "lora", "qlora", "full_ft"]
DatasetName = Literal[
    "synthetic",
    "docvqa",
    "textvqa",
    "sharegpt4v_coco",
    "llava_instruct",
    "llava_textvqa",
]
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


def _normalize_llm_token(llm_name: str) -> str:
    lower = llm_name.lower()
    token = lower.split("/")[-1]
    token = re.sub(r"-(instruct|chat|base)$", "", token)
    token = re.sub(r"[^a-z0-9.\-]+", "", token)
    return token or "llm"


def _normalize_vision_token(vision_name: str) -> str:
    lower = vision_name.lower()
    if "siglip" in lower:
        numbers = re.findall(r"(\d{3,4})", lower)
        if numbers:
            return f"siglip{numbers[-1]}"
        return "siglip"

    token = lower.split("/")[-1]
    token = re.sub(r"[^a-z0-9.]+", "", token)
    return token or "vision"


def _derive_image_size(vision_name: str) -> int:
    lower = vision_name.lower()
    match = re.search(r"(\d{3,4})(?=\D|$)", lower)
    if match is None:
        raise ValueError(f"Unsupported vision model (missing image size): {vision_name}")
    return int(match.group(1))


def _derive_num_image_tokens(vision_name: str, image_size: int) -> int:
    lower = vision_name.lower()
    match = re.search(r"patch(\d+)", lower)
    if match is None:
        raise ValueError(f"Unsupported vision model (missing patch size): {vision_name}")
    patch = int(match.group(1))
    if image_size % patch != 0:
        raise ValueError(f"image_size must be divisible by patch size ({patch}) for {vision_name}, got {image_size}")
    tokens = (image_size // patch) ** 2
    return tokens


def _compute_run_name(*, data: DataConfig, model: ModelConfig) -> str:
    split = data.split.lower()
    dataset = data.dataset
    llm = _normalize_llm_token(model.llm_name)
    vision = _normalize_vision_token(model.vision_name)
    return f"{dataset}_{split}_{llm}_{vision}"


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    device: Literal["auto", "cpu", "cuda"]

    batch_size: int
    micro_batch_size: int
    grad_accum_steps: int
    max_seq_len: int
    epochs: int
    max_steps: int | None

    log_every: int
    save_every: int
    eval_every: int
    eval_max_samples: int
    max_grad_norm: float

    gradient_checkpointing: bool

    lr: float
    lr_schedule: Literal["cosine"]
    warmup_ratio: float

    stage_name: Stage
    lora_r: int | None
    lora_alpha: int | None
    lora_dropout: float | None
    lora_targets: tuple[str, ...] | None


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
            raise ValueError("project must be a dict with name and optional run_name")
        name: str = project_d["name"]
        run_name = project_d.get("run_name")

        # --- Model ---
        model_d = d["model"]
        model = ModelConfig(
            llm_name=str(model_d["llm_name"]),
            vision_name=str(model_d["vision_name"]),
        )

        # --- Data ---
        data_d = d["data"]
        dataset = str(data_d["dataset"]).lower()
        if dataset not in {
            "synthetic",
            "docvqa",
            "textvqa",
            "sharegpt4v_coco",
            "llava_instruct",
            "llava_textvqa",
        }:
            raise ValueError(
                "data.dataset must be one of synthetic|docvqa|textvqa|sharegpt4v_coco|llava_instruct|llava_textvqa, "
                f"got: {data_d['dataset']}"
            )
        data_dir = str(data_d.get("data_dir", "datasets/processed"))
        split = str(data_d.get("split", "train"))
        data = DataConfig(
            dataset=cast(DatasetName, dataset),
            data_dir=data_dir,
            split=split,
            num_samples=int(data_d["num_samples"]),
        )

        # --- MM ---
        mm_d = d["mm"]
        projector = str(mm_d["projector"])
        if projector != "mlp2":
            raise ValueError(f"Only projector=mlp2 is supported, got: {projector}")

        image_size = _derive_image_size(model.vision_name)
        num_image_tokens = _derive_num_image_tokens(model.vision_name, image_size)
        mm = MmConfig(
            num_image_tokens=num_image_tokens,
            projector="mlp2",
        )

        # --- Paths ---
        paths_d = d["paths"]
        paths = PathsConfig(
            artifacts_dir=str(paths_d["artifacts_dir"]),
            reports_dir=str(paths_d["reports_dir"]),
        )

        # --- Train ---
        train_d = d["train"]
        stage_d = train_d.get("stage")
        if not isinstance(stage_d, dict):
            raise ValueError("train.stage must be a dict with name and optional params")
        stage_name_raw = stage_d.get("name")
        if not isinstance(stage_name_raw, str):
            raise ValueError("train.stage.name must be a string")
        stage_name_raw = stage_name_raw.lower()
        if stage_name_raw == "lora":
            stage_name = "lora"
        elif stage_name_raw == "qlora":
            stage_name = "qlora"
        elif stage_name_raw in {"projector", "full_ft"}:
            stage_name = cast(Stage, stage_name_raw)
        else:
            raise ValueError(f"train.stage.name must be projector|lora|qlora|full_ft, got: {stage_name_raw}")
        stage_params = stage_d.get("params") or {}
        if not isinstance(stage_params, dict):
            raise ValueError("train.stage.params must be a dict")
        device = str(train_d.get("device", "auto")).lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"train.device must be one of auto|cpu|cuda, got: {device}")

        # bf16-only policy: we don't accept config precision knobs here.
        # If you have 'precision' in yaml, we ignore it on purpose.
        lora_targets_raw = stage_params.get("lora_targets")
        if stage_name in {"lora", "qlora"}:
            for key in ("lora_r", "lora_alpha", "lora_dropout"):
                if key not in stage_params:
                    raise ValueError(f"train.stage.params.{key} is required for stage={stage_name}")
            if not isinstance(lora_targets_raw, list) or not all(isinstance(x, str) for x in lora_targets_raw):
                raise ValueError("train.stage.params.lora_targets must be a list[str]")

            # keep it specific to Qwen2.5 names
            allowed_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
            for t in lora_targets_raw:
                if t not in allowed_targets:
                    raise ValueError(f"Unsupported LoRA target module: {t}. Allowed: {sorted(allowed_targets)}")
        else:
            if stage_params:
                raise ValueError("train.stage.params must be empty for non-PEFT stages")

        batch_size = int(train_d["batch_size"])
        micro_batch_size = int(train_d["micro_batch_size"])
        if batch_size <= 0 or micro_batch_size <= 0:
            raise ValueError("train.batch_size and train.micro_batch_size must be > 0")
        if batch_size % micro_batch_size != 0:
            raise ValueError("train.batch_size must be divisible by train.micro_batch_size")
        grad_accum_steps = batch_size // micro_batch_size

        lr_schedule = str(train_d.get("lr_schedule", "")).lower()
        if lr_schedule != "cosine":
            raise ValueError(f"train.lr_schedule must be 'cosine', got: {lr_schedule}")
        warmup_ratio = float(train_d.get("warmup_ratio", 0.0))
        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            raise ValueError("train.warmup_ratio must be between 0 and 1")

        max_steps_raw = train_d.get("max_steps", None)
        max_steps = int(max_steps_raw) if max_steps_raw is not None else None

        train = TrainConfig(
            seed=int(train_d["seed"]),
            device=device,  # type: ignore[arg-type]
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps,
            max_seq_len=int(train_d["max_seq_len"]),
            epochs=int(train_d.get("epochs", 1)),
            max_steps=max_steps,
            log_every=int(train_d["log_every"]),
            save_every=int(train_d["save_every"]),
            eval_every=int(train_d.get("eval_every", 0)),
            eval_max_samples=int(train_d.get("eval_max_samples", 0)),
            max_grad_norm=float(train_d["max_grad_norm"]),
            gradient_checkpointing=bool(train_d["gradient_checkpointing"]),
            lr=float(train_d["lr"]),
            lr_schedule="cosine",
            warmup_ratio=warmup_ratio,
            stage_name=cast(Stage, stage_name),
            lora_r=int(stage_params["lora_r"]) if "lora_r" in stage_params else None,
            lora_alpha=int(stage_params["lora_alpha"]) if "lora_alpha" in stage_params else None,
            lora_dropout=float(stage_params["lora_dropout"]) if "lora_dropout" in stage_params else None,
            lora_targets=tuple(lora_targets_raw) if isinstance(lora_targets_raw, list) else None,
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
        if eval_d is None:
            eval_cfg = EvalConfig(enabled=False, max_samples=0, batch_size=1)
        elif not isinstance(eval_d, dict):
            raise ValueError("eval must be a dict with enabled, max_samples, batch_size")
        else:
            if "batch_size" not in eval_d:
                raise ValueError("eval.batch_size is required")
            eval_cfg = EvalConfig(
                enabled=bool(eval_d.get("enabled", True)),
                max_samples=int(eval_d.get("max_samples", 0)),
                batch_size=int(eval_d["batch_size"]),
            )

        if run_name is None:
            run_name = _compute_run_name(data=data, model=model)

        project = ProjectConfig(name=name, run_name=run_name)

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
