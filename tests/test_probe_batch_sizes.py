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
