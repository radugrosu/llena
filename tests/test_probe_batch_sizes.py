import pytest

import scripts.probe_batch_sizes as probe_script


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(
        self,
        conversation,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        return_dict: bool = False,
        return_tensors: None = None,
    ):
        _ = add_generation_prompt, tokenize, return_dict, return_tensors
        text = " ".join(msg["content"] for msg in conversation)
        # include small fixed overhead to emulate chat special tokens
        n = len(text.split()) + 4
        return list(range(n))


def test_build_full_sequence_question_reaches_target_length() -> None:
    tokenizer = _FakeTokenizer()
    q = probe_script._build_full_sequence_question(tokenizer, max_seq_len=64)
    assert probe_script._chat_prompt_len(tokenizer, q) >= 64


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


def test_run_probe_adds_runtime_overrides_true(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(config_path: str, overrides: list[str]) -> dict:
        captured["config_path"] = config_path
        captured["overrides"] = list(overrides)
        return {}

    def fake_from_dict(_cfg: dict) -> object:
        raise RuntimeError("stop after override capture")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(fake_from_dict))

    with pytest.raises(RuntimeError, match="stop after override capture"):
        probe_script.run_probe(
            config_path="dummy.yaml",
            probe_mode="train",
            gradient_checkpointing=True,
            liger_kernel=True,
        )

    assert captured["config_path"] == "dummy.yaml"
    assert captured["overrides"] == [
        "train.gradient_checkpointing=true",
        "train.liger_kernel=true",
    ]


def test_run_probe_adds_runtime_overrides_false(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(_config_path: str, overrides: list[str]) -> dict:
        captured["overrides"] = list(overrides)
        return {}

    def fake_from_dict(_cfg: dict) -> object:
        raise RuntimeError("stop after override capture")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(fake_from_dict))

    with pytest.raises(RuntimeError, match="stop after override capture"):
        probe_script.run_probe(
            config_path="dummy.yaml",
            probe_mode="train",
            gradient_checkpointing=False,
            liger_kernel=False,
        )

    assert captured["overrides"] == [
        "train.gradient_checkpointing=false",
        "train.liger_kernel=false",
    ]
