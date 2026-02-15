from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import scripts.train as train_script


class _DummyProjector:
    def __init__(self) -> None:
        self.loaded = False

    def load_state_dict(self, _state: dict[str, Any], strict: bool = True) -> None:
        self.loaded = strict


class _DummyModel:
    def __init__(self) -> None:
        self.projector = _DummyProjector()
        self.cfg = SimpleNamespace(peft_enable=False)
        self.model_loaded = False

    def load_state_dict(self, _state: dict[str, Any], strict: bool = True) -> None:
        self.model_loaded = strict


class _DummyOptimizer:
    def __init__(self) -> None:
        self.load_calls = 0

    def load_state_dict(self, _state: dict[str, Any]) -> None:
        self.load_calls += 1


def test_load_ckpt_same_stage_loads_optimizer_and_marks_no_transition(monkeypatch) -> None:
    ckpt = {
        "step": 7,
        "stage": "lora",
        "optimizer": {"state": {}},
        "wandb_run_id": "run_same_stage",
        "wandb_project": "llena",
    }

    monkeypatch.setattr(train_script.torch, "load", lambda *_args, **_kwargs: ckpt)

    model = _DummyModel()
    optm = _DummyOptimizer()
    step, stage, run_id, project, stage_transition = train_script.load_ckpt(
        Path("unused.ckpt"),
        model,  # pyright: ignore[reportArgumentType]
        optm,  # pyright: ignore[reportArgumentType]
        torch.device("cpu"),
        expected_stage="lora",
    )

    assert step == 7
    assert stage == "lora"
    assert run_id == "run_same_stage"
    assert project == "llena"
    assert stage_transition is False
    assert optm.load_calls == 1


def test_load_ckpt_stage_transition_skips_optimizer_and_marks_transition(monkeypatch) -> None:
    ckpt = {
        "step": 12,
        "stage": "projector",
        "optimizer": {"state": {}},
        "wandb_run_id": "run_stage1",
        "wandb_project": "llena",
    }

    monkeypatch.setattr(train_script.torch, "load", lambda *_args, **_kwargs: ckpt)

    model = _DummyModel()
    optm = _DummyOptimizer()
    step, stage, run_id, project, stage_transition = train_script.load_ckpt(
        Path("unused.ckpt"),
        model,  # pyright: ignore[reportArgumentType]
        optm,  # pyright: ignore[reportArgumentType]
        torch.device("cpu"),
        expected_stage="lora",
    )

    assert step == 0
    assert stage == "projector"
    assert run_id == "run_stage1"
    assert project == "llena"
    assert stage_transition is True
    assert optm.load_calls == 0


def test_parse_wandb_resume_policy_validation() -> None:
    assert train_script.parse_wandb_resume_policy("auto") == "auto"
    assert train_script.parse_wandb_resume_policy("always") == "always"
    assert train_script.parse_wandb_resume_policy("never") == "never"

    with pytest.raises(ValueError, match="wandb_resume_policy must be auto\\|always\\|never"):
        train_script.parse_wandb_resume_policy("bad-value")


@pytest.mark.parametrize(
    ("resume_id", "stage_transition", "policy", "expected"),
    [
        ("run_id", False, "auto", True),
        ("run_id", True, "auto", False),
        ("run_id", True, "always", True),
        ("run_id", False, "never", False),
        (None, False, "always", False),
    ],
)
def test_should_resume_wandb_run(
    resume_id: str | None,
    stage_transition: bool,
    policy: train_script.WandbResumePolicy,
    expected: bool,
) -> None:
    assert (
        train_script.should_resume_wandb_run(
            resume_wandb_id=resume_id,
            stage_transition=stage_transition,
            policy=policy,
        )
        == expected
    )


def test_checkpoint_run_mode_message_fresh() -> None:
    assert (
        train_script.checkpoint_run_mode_message(
            resume_ckpt_path=None,
            start_step=0,
            ckpt_stage=None,
            stage_transition=False,
        )
        == "ckpt: run_mode=new resume_checkpoint=none"
    )


def test_checkpoint_run_mode_message_resumed() -> None:
    assert (
        train_script.checkpoint_run_mode_message(
            resume_ckpt_path=Path("artifacts/run/step_100/ckpt.pt"),
            start_step=100,
            ckpt_stage="lora",
            stage_transition=False,
        )
        == "ckpt: run_mode=resumed resume_checkpoint=artifacts/run/step_100/ckpt.pt start_step=100 ckpt_stage=lora"
    )


def test_checkpoint_run_mode_message_stage_transition() -> None:
    assert (
        train_script.checkpoint_run_mode_message(
            resume_ckpt_path=Path("artifacts/run/step_100/ckpt.pt"),
            start_step=0,
            ckpt_stage="projector",
            stage_transition=True,
        )
        == "ckpt: run_mode=new_from_checkpoint "
        "resume_checkpoint=artifacts/run/step_100/ckpt.pt reason=stage_transition ckpt_stage=projector start_step=0"
    )


def test_wandb_run_mode_message_resumed() -> None:
    assert (
        train_script.wandb_run_mode_message(
            should_resume=True,
            resume_wandb_id="run_123",
            stage_transition=False,
            policy="auto",
        )
        == "wandb: run_mode=resumed existing_run_id=run_123 policy=auto"
    )


@pytest.mark.parametrize(
    ("resume_id", "stage_transition", "policy", "expected_reason"),
    [
        (None, False, "auto", "no_existing_run_id"),
        ("run_123", False, "never", "policy_never"),
        ("run_123", True, "auto", "stage_transition_auto"),
        ("run_123", False, "auto", "resume_disabled"),
    ],
)
def test_wandb_run_mode_message_new_reasons(
    resume_id: str | None,
    stage_transition: bool,
    policy: train_script.WandbResumePolicy,
    expected_reason: str,
) -> None:
    assert (
        train_script.wandb_run_mode_message(
            should_resume=False,
            resume_wandb_id=resume_id,
            stage_transition=stage_transition,
            policy=policy,
        )
        == f"wandb: run_mode=new reason={expected_reason} policy={policy}"
    )
