from mm.run_config import RunConfig


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
            "num_samples": 0,
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
