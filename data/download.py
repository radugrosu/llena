from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from data.hf_specs import HF_DATASETS


def expected_layout(data_dir: Path, dataset: str) -> tuple[Path, Path]:
    """
    Expected local layout:
      <data_dir>/<dataset>/{split}.jsonl
      <data_dir>/<dataset>/images/...
    """
    annotations_dir = data_dir / dataset
    images_dir = data_dir / dataset / "images"
    return annotations_dir, images_dir


def download_hf_parquet(
    *,
    dataset: str,
    out_dir: Path,
) -> Path:
    if dataset not in HF_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    spec = HF_DATASETS[dataset]

    local_dir = out_dir / dataset
    snapshot_download(
        repo_id=spec.dataset_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=spec.parquet_patterns,
    )
    return local_dir
