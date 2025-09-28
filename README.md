# VLM Studies

Project to understand multimodal AI and build an experimentation "laboratory" around visual-language models.

## Goals
- Build intuition for multimodal model internals and training dynamics.
- Fine-tune Qwen2-VL-2B on NLVR2 using parameter-efficient methods (LoRA / Q-LoRA).
- Create reproducible pipelines for evaluation, analysis, and ablations across visual reasoning tasks.

## Environment Setup
### Local (Windows + VS Code)
1. Install Python 3.10 or 3.11.
2. Create a virtual environment: `python -m venv .venv`.
3. Select `.venv` inside VS Code (`Python: Select Interpreter`) or activate manually with `\.venv\Scripts\activate` in PowerShell.
4. Install dependencies: `pip install -r requirements.txt`.
5. Run `accelerate config` when you first invoke training/eval scripts locally (CPU/lightweight runs only).

### Remote GPU (Colab or Lambda Cloud)
1. Clone this repository after mounting the appropriate storage (e.g., Google Drive at `/content/drive/MyDrive/`).
2. `pip install -r requirements.txt` (skip `torch` if a GPU build is already present).
3. Configure `accelerate` for a single GPU with bf16 mixed precision.
4. Persist artifacts under `/content/drive/MyDrive/VLM_Studies_Files/` using subdirectories such as `checkpoints/`, `logs/`, and `datasets/`.

## Data Strategy
- Primary dataset: NLVR2 (≈107k examples, ~6–7 GB of images).
- Start with Hugging Face streaming (`datasets.load_dataset("pingzhili/nlvr2", streaming=True)`) for iteration speed.
- Cache full copies to Drive/Lambda storage only when stable throughput is required; keep a manifest documenting version, split sizes, and SHA checks.
- Dataset preprocessing utilities live in `vlm_datasets/` and expose a toggle between streaming and cached modes.
- Each example retains its NLVR2 `uid` plus metadata (image dimensions, URLs when available). Use `scripts/inspect_sample.py` to surface details and optionally export the paired images for manual review.

## Training & Logging
- Training scripts use TRL + PEFT for LoRA adapters, saving adapter weights and optimizer state only.
- TensorBoard is the default experiment logger; run `%tensorboard --logdir logs/tb` in Colab or `tensorboard --logdir logs/tb` locally.
- Store configs under `training/configs/` and `eval/configs/` to reproduce runs; record seeds, prompt templates, and dataset variants.

## Manual Exploration
- `scripts/sample_gallery.py` surfaces NLVR2 examples (by UID or random draw) and can save paired images under `/content/drive/MyDrive/VLM_Studies_Files/analysis/...`.
- `scripts/model_playground.py` runs free-form prompts through Qwen2-VL with adjustable decoding knobs; append responses to JSONL logs for later study.
- `scripts/inspect_sample.py` reveals metadata for a specific UID and optionally exports the raw images.
- Notebook helpers expose a persistent UID cache (`utils/uid_cache.py`) so you can bookmark interesting examples and cycle through them during interactive exploration.
- `colab/Playground.ipynb` wires the pieces together for Drive-mounted experimentation in Colab.

- `scripts/sample_gallery.py` surfaces NLVR2 examples (by UID or random draw) and can save paired images under `/content/drive/MyDrive/VLM_Studies_Files/analysis/...`.
- `scripts/model_playground.py` runs free-form prompts through Qwen2-VL with adjustable decoding knobs; append responses to JSONL logs for later study.
- `scripts/inspect_sample.py` reveals metadata for a specific UID and optionally exports the raw images.
- `colab/Playground.ipynb` wires the pieces together for Drive-mounted experimentation in Colab.

## Repository Layout (work in progress)
```
vlm_datasets/ # Dataset preparation and prompt templating
training/    # Fine-tuning loops, configs, and utilities
evals/       # Evaluation scripts and analysis notebooks
scripts/     # Entry-point scripts for Colab/Lambda runs
utils/       # Shared helpers (prompt builders, etc.)
colab/       # Colab notebooks for setup and exploration
Setup/       # Planning notes and references
```

## Next Milestones
- Implement NLVR2 preprocessing pipeline (streaming + cached paths).
- Add zero-shot evaluation script for Qwen2-VL-2B on NLVR2 dev.
- Deliver fine-tuning smoke test script runnable on a single GPU.
- Automate TensorBoard logging and Drive checkpoint management.
- Build sample-inspection workflows that map evaluation outcomes back to raw NLVR2 images for error analysis.
