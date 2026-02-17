from types import SimpleNamespace
from typing import cast
import pytest

from mm.types import ChatTokenizer
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
    tokenizer = cast(ChatTokenizer, _FakeTokenizer())
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


def test_measure_throughput_counts_tokens(monkeypatch) -> None:
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr(probe_script.time, "perf_counter", lambda: next(ticks))

    calls = {"count": 0}

    def run_step() -> int:
        calls["count"] += 1
        return 128

    tps = probe_script.measure_throughput(
        run_step=run_step,
        num_steps=5,
        warmup_steps=2,
        device=probe_script.torch.device("cpu"),
    )

    assert calls["count"] == 7
    assert tps == pytest.approx((5 * 128) / 2.0)


def test_active_gpu_make_and_size_cpu() -> None:
    make, size_gb = probe_script._active_gpu_make_and_size(probe_script.torch.device("cpu"))
    assert make == "cpu"
    assert size_gb is None


def test_active_gpu_make_and_size_cuda_without_runtime(monkeypatch) -> None:
    monkeypatch.setattr(probe_script.torch.cuda, "is_available", lambda: False)
    make, size_gb = probe_script._active_gpu_make_and_size(probe_script.torch.device("cuda"))
    assert make == "cpu"
    assert size_gb is None


def test_effective_probe_precision_cpu_forces_fp32() -> None:
    precision = probe_script._effective_probe_precision(
        configured_precision="bf16",
        device=probe_script.torch.device("cpu"),
    )
    assert precision == "fp32"


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


def test_run_probe_matrix_collects_all_combinations(monkeypatch) -> None:
    calls: list[tuple[str, bool, bool, bool]] = []
    logged: dict[str, object] = {}

    def fake_run_probe(
        *,
        config_path: str,
        probe_mode: probe_script.ProbeMode,
        gradient_checkpointing: bool,
        liger_kernel: bool,
        log_wandb: bool,
    ) -> probe_script.ProbeResult:
        calls.append((probe_mode, gradient_checkpointing, liger_kernel, log_wandb))
        return probe_script.ProbeResult(
            mode=probe_mode,
            max_ok_batch=8,
            tested_up_to=16,
            recommended_batch=7,
            precision="bf16",
            gradient_checkpointing=gradient_checkpointing,
            liger_kernel=liger_kernel,
            gpu_make="A100",
            gpu_size_gb=80.0,
        )

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(config_path: str, overrides: list[str]) -> dict:
        logged["config_path"] = config_path
        logged["overrides"] = list(overrides)
        return {"ok": True}

    class _FakeRunConfig:
        @staticmethod
        def from_dict(_cfg: dict) -> object:
            return "run-config"

    def fake_log_probe_matrix_to_wandb(
        *, rc: object, config_path: str, results: list[probe_script.ProbeResult]
    ) -> None:
        logged["rc"] = rc
        logged["matrix_config_path"] = config_path
        logged["results"] = results

    monkeypatch.setattr(probe_script, "run_probe", fake_run_probe)
    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script, "RunConfig", _FakeRunConfig)
    monkeypatch.setattr(probe_script, "_log_probe_matrix_to_wandb", fake_log_probe_matrix_to_wandb)

    results = probe_script.run_probe_matrix(config_path="cfg.yaml")

    expected_count = len(probe_script.MATRIX_PROBE_MODES) * 2 * 2
    assert len(results) == expected_count
    assert len(calls) == expected_count
    assert all(log_wandb is False for _, _, _, log_wandb in calls)
    assert logged["config_path"] == "cfg.yaml"
    assert logged["overrides"] == [
        "train.gradient_checkpointing=false",
        "train.liger_kernel=false",
    ]
    assert logged["matrix_config_path"] == "cfg.yaml"
    assert logged["rc"] == "run-config"
    assert len(logged["results"]) == expected_count  # pyright: ignore[reportArgumentType]


def test_run_probe_skips_liger_on_cpu(monkeypatch) -> None:
    fake_rc = SimpleNamespace(
        model=SimpleNamespace(vision_name="google/siglip-base-patch16-224"),
        train=SimpleNamespace(stage_name="projector", device="cpu", precision="bf16"),
    )

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(_config_path: str, overrides: list[str]) -> dict:
        _ = overrides
        return {}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("should not be called for skipped CPU+liger probes")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(lambda _cfg: fake_rc))
    monkeypatch.setattr(probe_script, "derive_vision_params", lambda _vision_name: (224, 196))
    monkeypatch.setattr(probe_script.train_script, "get_device", lambda _dev, force_cuda: probe_script.torch.device("cpu"))
    monkeypatch.setattr(probe_script.SiglipImageProcessor, "from_pretrained", staticmethod(fail_if_called))

    result = probe_script.run_probe(
        config_path="dummy.yaml",
        probe_mode="train",
        gradient_checkpointing=False,
        liger_kernel=True,
        log_wandb=False,
    )

    assert result.mode == "train"
    assert result.liger_kernel is True
    assert result.max_ok_batch is None
    assert result.recommended_batch is None
    assert result.tested_up_to is None
    assert result.tokens_per_sec is None
    assert result.note is not None and "skipped" in result.note


def test_run_probe_skips_qlora_when_cuda_unavailable(monkeypatch) -> None:
    fake_rc = SimpleNamespace(
        model=SimpleNamespace(vision_name="google/siglip-base-patch16-224"),
        train=SimpleNamespace(stage_name="qlora", device="auto", precision="bf16"),
    )

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(_config_path: str, overrides: list[str]) -> dict:
        _ = overrides
        return {}

    def fake_get_device(_device: str, *, force_cuda: bool):
        assert force_cuda is True
        raise RuntimeError("CUDA required but not available.")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(lambda _cfg: fake_rc))
    monkeypatch.setattr(probe_script, "derive_vision_params", lambda _vision_name: (224, 196))
    monkeypatch.setattr(probe_script.train_script, "get_device", fake_get_device)

    result = probe_script.run_probe(
        config_path="dummy.yaml",
        probe_mode="train",
        gradient_checkpointing=False,
        liger_kernel=False,
        log_wandb=False,
    )

    assert result.max_ok_batch is None
    assert result.recommended_batch is None
    assert result.tested_up_to is None
    assert result.precision == "fp32"
    assert result.note is not None and "CUDA required but not available" in result.note


def test_run_probe_skips_when_device_cuda_but_runtime_unavailable(monkeypatch) -> None:
    fake_rc = SimpleNamespace(
        model=SimpleNamespace(vision_name="google/siglip-base-patch16-224"),
        train=SimpleNamespace(stage_name="projector", device="cuda", precision="bf16"),
    )

    def fake_load_dotenv() -> None:
        return None

    def fake_load_config(_config_path: str, overrides: list[str]) -> dict:
        _ = overrides
        return {}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("should not be called for skipped CUDA-unavailable probes")

    monkeypatch.setattr(probe_script, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(probe_script, "load_config", fake_load_config)
    monkeypatch.setattr(probe_script.RunConfig, "from_dict", staticmethod(lambda _cfg: fake_rc))
    monkeypatch.setattr(probe_script, "derive_vision_params", lambda _vision_name: (224, 196))
    monkeypatch.setattr(probe_script.train_script, "get_device", lambda _dev, force_cuda: probe_script.torch.device("cuda"))
    monkeypatch.setattr(probe_script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(probe_script.SiglipImageProcessor, "from_pretrained", staticmethod(fail_if_called))

    result = probe_script.run_probe(
        config_path="dummy.yaml",
        probe_mode="train",
        gradient_checkpointing=False,
        liger_kernel=False,
        log_wandb=False,
    )

    assert result.max_ok_batch is None
    assert result.recommended_batch is None
    assert result.tested_up_to is None
    assert result.note is not None and "train.device resolved to cuda but CUDA is unavailable" in result.note
