# Project plan: Swappable Mini-LLaVA for Document/UI Understanding

## 0) Goal

Build a **LLaVA-style multimodal LLM** that can answer questions about **documents / UI screenshots** (DocVQA/TextVQA style), with a design that lets you **swap LLM backbones (3B ↔ 7B)** by changing config only.

Primary signals for training-oriented roles:

- clean training pipeline (stages, freezing schedules, reproducibility)
- ablations + metrics
- (optional) short distributed scaling demo

---

## 1) Scope and deliverables

### Deliverable A — Model

A multimodal wrapper with:

- **frozen vision encoder** (SigLIP/CLIP ViT)
- **projector/resampler** that maps vision features → LLM hidden size
- **causal LLM** backbone loaded via HF (default Qwen2.5-1.5B, later Qwen2.5-7B)
- **embedding injection** approach so the wrapper is model-agnostic

**Swappability requirement**

- only `llm_name` changes in config; projector dims adapt from `llm.config.hidden_size`.

### Deliverable B — Training pipeline

A reproducible training setup with:

- config-driven runs (YAMLs)
- deterministic seeds, checkpointing, resume support
- logging (W&B/MLflow)
- mixed precision (bf16), QLoRA/LoRA support
- scripted stage transitions (freeze/unfreeze schedules)

### Deliverable C — Data pipeline

Scripts to:

- download / load selected public datasets
- convert each sample into a unified **multimodal conversation format**
- optionally generate a small custom dataset (200–500 samples) of UI/doc screenshots + QA, including **unanswerable** cases
- store metadata + hashes for dedup / tracking

### Deliverable D — Evaluation harness

Single `eval.py` that:

- runs on each dataset split
- reports task metrics (dataset-specific) + a shared “hallucination/abstention” metric on the custom set
- produces a small “failure gallery” artifact (examples + model outputs)

### Deliverable E — Demo

A simple Gradio demo:

- upload image (doc/UI screenshot)
- ask question
- show answer + optional “confidence/abstain” behavior

### Deliverable F — Write-up

A short report:

- architecture diagram
- training stages + compute used
- ablations and conclusions
- known failure modes and next steps

---

## 2) Task definition

### Primary task

**Document/UI visual question answering**:

- input: image + instruction question
- output: short grounded answer (often text spans / entities / yes-no / short phrases)

### Secondary capability (differentiator)

**Abstention / unanswerable detection**:

- model should refuse when the answer isn’t present in the image
- evaluate hallucination rate vs abstain precision/recall on a labeled subset

---

## 3) Datasets

### Public datasets (choose 2 to start)

- **DocVQA** (document QA)
- **TextVQA** (text in the wild)

Optional later:

- **ChartQA** (charts)

### Custom dataset

- 200–500 curated or synthetic UI/doc screenshots
- QA pairs + label for “answerable/unanswerable”
- used primarily for robustness + qualitative analysis

---

## 4) Model architecture requirements

### Vision encoder

- pretrained ViT (SigLIP/CLIP)
- frozen by default

### Multimodal connector

- baseline: **2-layer MLP projector**
- optional ablation: **Perceiver resampler** (fixed latents)

### Token/embedding integration

- inject **image embeddings as “virtual tokens”** (no tokenizer vocab hacks required)
- concatenate `[image_embeds] + [text_embeds]` into `inputs_embeds`
- ensure training labels mask out image positions appropriately

### LLM backbone

- Hugging Face `AutoModelForCausalLM`
- default dev: **~3B instruct model**
- final: **~7B instruct model**
- training via **LoRA/QLoRA** (configurable target modules)

---

## 5) Training stages (MVP)

### Stage 0 — Smoke test (dev GPU)

- tiny subset, few hundred steps
- verify loss decreases, resume works, eval runs, outputs look sane

### Stage 1 — Projector warmup (cheap)

- freeze vision + LLM
- train projector only on image→caption style data
- goal: align vision features into LLM space

### Stage 2 — Multimodal instruction tuning (main)

- freeze vision
- train projector + LoRA/QLoRA on LLM
- mix DocVQA/TextVQA (+ small caption/instruct mix for fluency)

### Stage 3 — Domain specialization + ablations

Run short, controlled comparisons:

- projector MLP vs Perceiver
- LoRA rank (e.g., 8 vs 16)
- unfreeze strategy (LoRA-only vs LoRA + top-k blocks)

---

## 6) Evaluation requirements

### Quantitative

- dataset metrics (per dataset standard)
- shared metrics:
  - exact match / normalized match (where applicable)
  - abstention precision/recall on custom “unanswerable” set
  - hallucination rate (answers when unanswerable)

### Qualitative

- curated failure set + model outputs
- error taxonomy: OCR misses, layout reasoning, distractors, counting, etc.

### Reproducibility

- each run produces:
  - config snapshot
  - git commit hash
  - dataset version/hash
  - checkpoint + eval JSON

---

## 7) Compute/budget strategy

### Development

- cheap GPU (L4/4090-class) for iteration:
  - dataset + collator + forward pass + short training runs

### Training

- 1× H100 (or similar) for the “real” run:
  - Stage 1 + Stage 2 + ablations

### Optional scaling demo

- short 2–3 hour multi-GPU run (DDP/FSDP/DeepSpeed) to show scaling competence

Budget target: $200–400 total, with most spend on the final single-GPU training.

---

## 8) Engineering requirements

### Repo structure (suggested)

- `mm/` model wrapper + projectors
- `data/` dataset ingestion + formatting + custom data generation
- `configs/` run configs (3B/7B)
- `train.py`, `eval.py`, `demo.py`
- `scripts/` launchers, downloaders
- `tests/` unit tests for collation, masking, shapes, resume

### Must-have tests

- masking correctness (image tokens ignored in loss)
- same batch produces identical loss with fixed seed
- checkpoint resume equivalence (loss continuity)
- swapping 1.5B↔7B loads + runs forward without code changes

---

## 9) Success criteria

- Working demo + documented training pipeline
- Measurable improvement over a baseline (e.g., projector-only vs projector+LoRA)
- At least 2 ablations with clear conclusions
- Clean write-up + reproducible runs
- Seamless LLM swap via config

---

## 10) Non-goals (to keep scope sane)

- training an LLM from scratch
- full vision encoder fine-tuning (unless time/budget left)
- chasing SOTA; focus on correctness, rigor, and clarity
