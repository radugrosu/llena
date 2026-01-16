# Llena — Mini‑LLaVA for Document/UI VQA

Llena is a small, reproducible **LLaVA‑style multimodal LLM** focused on **document/UI visual question answering** (DocVQA/TextVQA). It pairs **SigLIP** (vision) with **Qwen2.5 Instruct** (LLM) via a lightweight **MLP projector**, and supports **projector warmup → LoRA → QLoRA** training stages with clean configs and tests.

Key goals:
- Model‑agnostic wrapper that can swap LLM sizes via config.
- Deterministic, stage‑driven training (projector‑only, LoRA, QLoRA).
- Simple data pipeline with JSONL + images.
- Evaluation harness for TextVQA (VQA accuracy) and DocVQA (ANLS).

## Quickstart

```bash
uv sync
uv run python -m scripts.train --config configs/local/textvqa_train.yaml --stage projector --max-steps 50
uv run python -m scripts.eval --config configs/local/textvqa_eval.yaml
```

## Data pipeline

We standardize datasets into a simple JSONL schema to keep training/eval logic minimal.
Each line is a single record:

```json
{"image": "00000000_34602.jpg", "question": "What does the sign say?", "answer": "STOP", "answers": ["STOP", "..."]}
```

### Download parquet shards (HF)

```bash
uv run python -m scripts.download_dataset --dataset textvqa --out-dir datasets/raw
uv run python -m scripts.download_dataset --dataset docvqa --out-dir datasets/raw
```

### Process parquet → JSONL + images

```bash
uv run python -m scripts.process_dataset --dataset textvqa --split validation --raw-dir datasets/raw --out-dir datasets/processed
uv run python -m scripts.process_dataset --dataset docvqa --split validation --raw-dir datasets/raw --out-dir datasets/processed
```

Expected on‑disk layout:

```
datasets/raw/
  textvqa/...
  docvqa/...
datasets/processed/
  textvqa/
    validation.jsonl
    images/...
  docvqa/
    validation.jsonl
    images/...
```

## Training

Stage‑driven runs:

```bash
uv run python -m scripts.train --config configs/cloud/textvqa_train.yaml --stage projector --max-steps 1000
uv run python -m scripts.train --config configs/cloud/textvqa_train.yaml --stage peft_lora --max-steps 1000
```

## Evaluation

```bash
uv run python -m scripts.eval --config configs/local/textvqa_eval.yaml
uv run python -m scripts.eval --config configs/local/docvqa_eval.yaml
```

Eval outputs:
- TextVQA: VQA accuracy (soft‑match over 10 answers)
- DocVQA: ANLS (normalized edit‑distance similarity)

## Notebooks

For interactive inspection (tokenizer, collator, forward pass), use:
- `notebooks/inspect_pipeline.ipynb`

## Notes

- CPU runs are for pipeline validation only.
- Cloud configs assume CUDA and log to W&B.
