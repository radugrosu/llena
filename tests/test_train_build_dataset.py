from types import SimpleNamespace
from typing import cast

from torch.utils.data import Dataset

from mm.run_config import RunConfig
import scripts.train as train_script


class _ListDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def _rc_for_llava_textvqa() -> object:
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset="llava_textvqa",
            data_dir="unused",
            split="train",
            num_samples=None,
        ),
        train=SimpleNamespace(seed=42),
    )


def _rc_for_synthetic() -> object:
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset="synthetic",
            data_dir="unused",
            split="train",
            num_samples=3,
        ),
        train=SimpleNamespace(seed=7),
        model=SimpleNamespace(vision_name="google/siglip-base-patch16-256"),
    )


def test_build_dataset_llava_textvqa_caps_to_dataset_len(monkeypatch) -> None:
    def fake_load_instruct_jsonl_dataset(*, dataset: str, data_dir, split: str, max_samples):
        assert dataset == "llava_instruct"
        return _ListDataset(
            [
                {
                    "image": None,
                    "conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                },
                {
                    "image": None,
                    "conversation": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}],
                },
            ]
        )

    def fake_load_vqa_jsonl_dataset(*, dataset: str, data_dir, split: str, max_samples):
        assert dataset == "textvqa"
        return _ListDataset(
            [
                {"image": None, "question": "q1", "answer": "a1"},
                {"image": None, "question": "q2", "answer": "a2"},
                {"image": None, "question": "q3", "answer": "a3"},
            ]
        )

    monkeypatch.setattr(train_script, "load_instruct_jsonl_dataset", fake_load_instruct_jsonl_dataset)
    monkeypatch.setattr(train_script, "load_vqa_jsonl_dataset", fake_load_vqa_jsonl_dataset)

    mock_conf = cast(RunConfig, _rc_for_llava_textvqa())
    ds = train_script.build_dataset(mock_conf, max_samples=10)
    assert len(ds) == 5  # pyright: ignore[reportArgumentType]
    for i in range(len(ds)):  # pyright: ignore[reportArgumentType]
        _ = ds[i]


def test_build_dataset_synthetic_uses_derived_image_size(monkeypatch) -> None:
    monkeypatch.setattr(train_script, "derive_vision_params", lambda _vision_name: (384, 576))
    mock_conf = cast(RunConfig, _rc_for_synthetic())
    ds = train_script.build_dataset(mock_conf, max_samples=None)
    assert hasattr(ds, "image_size")
    assert getattr(ds, "image_size") == 384
