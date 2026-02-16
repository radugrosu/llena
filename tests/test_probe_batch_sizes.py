import pytest

import scripts.probe_batch_sizes as probe_script


def test_search_max_batch_finds_threshold() -> None:
    threshold = 37

    def try_batch(bs: int) -> bool:
        return bs <= threshold

    max_ok, tested_up_to = probe_script._search_max_batch(
        try_batch=try_batch,
        start_batch=1,
        max_batch=128,
    )

    assert max_ok == threshold
    assert tested_up_to >= threshold


def test_search_max_batch_returns_zero_when_start_fails() -> None:
    max_ok, tested_up_to = probe_script._search_max_batch(
        try_batch=lambda _bs: False,
        start_batch=2,
        max_batch=64,
    )
    assert max_ok == 0
    assert tested_up_to == 2


def test_search_max_batch_caps_start_to_max_batch() -> None:
    max_ok, tested_up_to = probe_script._search_max_batch(
        try_batch=lambda bs: bs <= 8,
        start_batch=32,
        max_batch=8,
    )
    assert max_ok == 8
    assert tested_up_to == 8


def test_is_oom_error_detects_common_markers() -> None:
    assert probe_script.is_oom_error(RuntimeError("CUDA out of memory."))
    assert probe_script.is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not probe_script.is_oom_error(RuntimeError("unrelated failure"))


def test_recommend_applies_margin_and_floor() -> None:
    assert probe_script._recommend(64, 0.9) == 57
    assert probe_script._recommend(1, 0.9) == 1
    assert probe_script._recommend(0, 0.9) == 0


def test_run_probe_apply_liger_kernel_adds_override(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(config: str, overrides: list[str]) -> dict:
        captured["config"] = config
        captured["overrides"] = list(overrides)
        return {}

    def fake_from_dict(_cfg: dict) -> object:
        raise RuntimeError("stop after override capture")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(fake_from_dict))

    with pytest.raises(RuntimeError, match="stop after override capture"):
        probe_script.run_probe(
            config="dummy.yaml",
            out_json=None,
            max_batch=1,
            start_batch=1,
            probe_samples=1,
            safety_margin=1.0,
            eval_dataset=None,
            eval_split="validation",
            eval_max_generated_tokens=None,
            ckpt=None,
            probe_train=True,
            probe_val=False,
            probe_eval_teacher=False,
            probe_eval_generate=False,
            apply_liger_kernel=True,
            override=["data.split=train"],
        )

    assert captured["config"] == "dummy.yaml"
    assert captured["overrides"] == ["data.split=train", "train.liger_kernel=true"]


def test_run_probe_without_apply_liger_kernel_keeps_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(_config: str, overrides: list[str]) -> dict:
        captured["overrides"] = list(overrides)
        return {}

    def fake_from_dict(_cfg: dict) -> object:
        raise RuntimeError("stop after override capture")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(fake_from_dict))

    with pytest.raises(RuntimeError, match="stop after override capture"):
        probe_script.run_probe(
            config="dummy.yaml",
            out_json=None,
            max_batch=1,
            start_batch=1,
            probe_samples=1,
            safety_margin=1.0,
            eval_dataset=None,
            eval_split="validation",
            eval_max_generated_tokens=None,
            ckpt=None,
            probe_train=True,
            probe_val=False,
            probe_eval_teacher=False,
            probe_eval_generate=False,
            apply_liger_kernel=False,
            override=["data.split=train"],
        )

    assert captured["overrides"] == ["data.split=train"]
