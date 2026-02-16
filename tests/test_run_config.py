from types import SimpleNamespace

import mm.run_config as run_config
from mm.run_config import RunConfig, derive_vision_params
import pytest


@pytest.fixture(autouse=True)
def _mock_siglip_vision_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_config.SiglipVisionConfig,
        "from_pretrained",
        classmethod(lambda cls, vision_name: SimpleNamespace(image_size=224, patch_size=16)),
    )


def _base_cfg() -> dict:
    return {
        "project": {"name": "llena"},
        "paths": {"artifacts_dir": "artifacts", "reports_dir": "reports"},
        "model": {
            "llm_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "vision_name": "google/siglip-base-patch16-224",
        },
        "mm": {"num_image_tokens": 256, "projector": "mlp2"},
        "data": {
            "dataset": "textvqa",
            "data_dir": "datasets/processed",
            "split": "train",
        },
        "eval": {
            "mode": "teacher",
            "batch_size": 4,
        },
        "train": {
            "seed": 42,
            "device": "cpu",
            "gradient_checkpointing": False,
            "lr_schedule": "cosine",
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "max_seq_len": 512,
            "epochs": 1,
            "max_steps": 5,
            "batch_size": 2,
            "micro_batch_size": 1,
            "log_every": 10,
            "save_every": 0,
            "eval_every": 0,
            "eval_max_samples": 0,
            "lr": 1.0e-4,
            "stage": {
                "name": "lora",
                "params": {
                    "lora_r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
            },
        },
    }


def test_run_name_auto_from_config() -> None:
    cfg = _base_cfg()
    rc = RunConfig.from_dict(cfg)
    assert rc.project.run_name == "textvqa_train_qwen2.5-0.5b_siglip224"

    cfg = _base_cfg()
    cfg["data"]["split"] = "validation"
    rc = RunConfig.from_dict(cfg)
    assert rc.project.run_name == "textvqa_validation_qwen2.5-0.5b_siglip224"


def test_mm_num_image_tokens_derived_from_vision_config() -> None:
    cfg = _base_cfg()
    cfg["mm"]["num_image_tokens"] = 999
    rc = RunConfig.from_dict(cfg)
    assert rc.mm.num_image_tokens == 196


def test_derive_vision_params_fail_fast_on_invalid_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_config.SiglipVisionConfig,
        "from_pretrained",
        classmethod(lambda cls, vision_name: SimpleNamespace(image_size=224, patch_size=0)),
    )
    with pytest.raises(ValueError, match="Invalid patch_size"):
        derive_vision_params("google/siglip-base-patch16-224")


def test_run_config_fail_fast_when_vision_config_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(cls: type, vision_name: str) -> object:
        raise OSError("config not found")

    monkeypatch.setattr(run_config.SiglipVisionConfig, "from_pretrained", classmethod(_raise))
    with pytest.raises(ValueError, match="Failed to load SigLIP vision config"):
        RunConfig.from_dict(_base_cfg())


def test_eval_generate_batch_size_multiplier_defaults_and_validation() -> None:
    cfg = _base_cfg()
    rc = RunConfig.from_dict(cfg)
    assert rc.eval.generate_batch_size_multiplier == 8
    assert rc.train.liger_kernel is False
    assert rc.train.num_workers == 4
    assert rc.train.pin_memory is True
    assert rc.train.persistent_workers is False
    assert rc.eval.num_workers == 4
    assert rc.eval.pin_memory is True
    assert rc.eval.persistent_workers is False

    cfg = _base_cfg()
    cfg["eval"]["generate_batch_size_multiplier"] = 12
    rc = RunConfig.from_dict(cfg)
    assert rc.eval.generate_batch_size_multiplier == 12

    cfg = _base_cfg()
    cfg["eval"]["generate_batch_size_multiplier"] = 0
    with pytest.raises(ValueError, match="eval.generate_batch_size_multiplier must be > 0"):
        RunConfig.from_dict(cfg)


def test_train_liger_kernel_parse() -> None:
    cfg = _base_cfg()
    cfg["train"]["liger_kernel"] = True
    rc = RunConfig.from_dict(cfg)
    assert rc.train.liger_kernel is True


def test_dataloader_worker_fields_parse_and_validate() -> None:
    cfg = _base_cfg()
    cfg["train"]["num_workers"] = 4
    cfg["train"]["pin_memory"] = True
    cfg["train"]["persistent_workers"] = True
    cfg["eval"]["num_workers"] = 2
    cfg["eval"]["pin_memory"] = True
    cfg["eval"]["persistent_workers"] = True

    rc = RunConfig.from_dict(cfg)
    assert rc.train.num_workers == 4
    assert rc.train.pin_memory is True
    assert rc.train.persistent_workers is True
    assert rc.eval.num_workers == 2
    assert rc.eval.pin_memory is True
    assert rc.eval.persistent_workers is True

    cfg = _base_cfg()
    cfg["train"]["num_workers"] = 0
    cfg["train"]["persistent_workers"] = True
    with pytest.raises(ValueError, match="train.persistent_workers requires train.num_workers > 0"):
        RunConfig.from_dict(cfg)

    cfg = _base_cfg()
    cfg["eval"]["num_workers"] = 0
    cfg["eval"]["persistent_workers"] = True
    with pytest.raises(ValueError, match="eval.persistent_workers requires eval.num_workers > 0"):
        RunConfig.from_dict(cfg)
