from types import SimpleNamespace

from torch.utils.data import Dataset

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


def test_build_dataset_llava_textvqa_caps_to_dataset_len(monkeypatch) -> None:
    def fake_load_instruct_jsonl_dataset(*, dataset: str, data_dir, split: str, max_samples):
        assert dataset == "llava_instruct"
        return _ListDataset(
            [
                {"image": None, "conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
                {"image": None, "conversation": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},
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

    ds = train_script.build_dataset(_rc_for_llava_textvqa(), max_samples=10)
    assert len(ds) == 5
    for i in range(len(ds)):
        _ = ds[i]

