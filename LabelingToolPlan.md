# Tool-Augmented NLVR2 Research Plan

## Motivation
Qwen2-VL-2B handles many NLVR2 prompts, but it struggles on fine-grained visual tasks such as counting televisions or describing unique content on each screen. These failures point to a broader weakness: the model lacks mechanisms to explicitly mark and reference important regions while reasoning. Equipping the model with a labeling tool (draw bounding boxes + textual tags) can:
- Ground intermediate reasoning steps in visual evidence.
- Reduce hallucinations when multiple similar objects appear.
- Provide measurable intermediate supervision (correct boxes) that enables reinforcement learning and tool-use experiments.

## Overall Trajectory
1. Build an offline labeling tool and demonstrate that the model can call it correctly via supervised fine-tuning.
2. Wrap the tool in a verifiable environment (ground-truth boxes) to compute rewards.
3. Apply reinforcement learning / iterative refinement so the model learns to place boxes and reason with them autonomously.
4. Evaluate whether tool-augmented reasoning improves NLVR2 performance and generalizes to related tasks (multi-object questions, attribute verification).

## Phase 1 – Tooling & Offline Supervision
### Deliverables
- Python module that renders bounding boxes with labels (Pillow/Matplotlib). Must return tensors compatible with the Qwen processor.
- Prompt schema using `<think>`, `<label_tool>`, `<labeled_image>` blocks. Coordinates stored as pixel integers or normalized floats.
- Scripted data generator that produces “box placement transcripts” from ground-truth annotations.
- LoRA fine-tuning run that teaches the model to emit correct tool calls, consume labeled images, and output final answers.

### Tasks
- Implement `tools/box_labeler.py` with `draw_boxes(image, boxes, labels)` function returning PIL image + metadata.
- Extend tokenizer utilities to ingest rendered labeled images back into the conversation for Qwen2-VL.
- Identify bootstrap datasets:
  - Synthetic: COCO, Objects365 subsets, synthetic geometric shapes for easy verifiable labels.
  - NLVR2 slices where object names appear (e.g., “television”, “dog”, “boat”).
- Auto-generate supervised sequences:
  - Step-by-step chain of thought describing each object.
  - Emit `<label_tool>` tokens with coordinates & text.
  - Append `<labeled_image>` token (image with overlays) to the conversation history.
- Adapt `training/sft.py` to make tool tokens part of the vocabulary (special IDs) and allow multi-image turns.

## Phase 2 – Environment & Rewarding
### Deliverables
- Deterministic reward function (IoU + label match).
- Episode runner that executes `think → tool call → labeled image` cycles until the model outputs a final answer.
- Logging pipeline capturing sequences, rewards, and overlays.

### Tasks
- Implement `env/labeling_env.py`:
  - Maintain current conversation state, apply model actions, and render new images via the tool module.
  - Track available objects and reward progress when predicted boxes match ground truth.
- Define reward shaping:
  - +1 per correctly labeled object (IoU ≥ threshold and label text match).
  - Penalties for duplicate or incorrect boxes.
  - Optional cost per tool invocation to encourage efficiency.
- Add instrumentation (TensorBoard scalars + image summaries) showing predicted vs. true overlays.

## Phase 3 – Reinforcement Learning
### Deliverables
- PPO (or other TRL-supported algorithm) training script that builds on the supervised weights.
- Curriculum policy: start with simple scenarios (single object, high-contrast) and progress to clutter.
- Stability mitigation (entropy regularization, KL penalties, reward normalization).

### Tasks
- Integrate environment with TRL’s PPOTrainer, keeping LoRA adapters trainable while freezing base weights.
- Implement batched environment stepping with Hugging Face Accelerate.
- Establish training recipes:
  - Warm start from Phase 1 SFT checkpoint.
  - Learning rate schedules tuned for adapter layers.
  - Replay buffer or on-policy rollouts with mixed synthetic + real samples.
- Monitor metrics: average reward, number of tool calls, percent of correctly labeled objects.

## Phase 4 – Evaluation & Analysis
### Deliverables
- Evaluation harness covering:
  - Zero-shot NLVR2 dev accuracy (baseline vs. tool-augmented runs).
  - Custom counting/attribute tasks (e.g., “How many screens show a face?”).
  - Stress cases (small objects, occlusions, repetitive textures).
- Ablation report: tool feedback on/off, text-only labels, explicit chain-of-thought toggles.

### Tasks
- Extend `evals/` with scripts to run the labeling policy deterministically and log overlays.
- Define success metrics:
  - Final question accuracy.
  - Precision/recall of labeled boxes.
  - Latency (# tool calls, total tokens).
- Explore generalization:
  - New tasks that require referencing labeled objects (“Which box contains the person wearing a hat?”).
  - Domain transfer experiments (e.g., limited subset of ChartQA or TextCaps where bounding boxes help).

## Risks & Mitigations
- **Coordinate regression difficulty**: 2B models might output noisy coordinates. Mitigate with quantized grid bins, curriculum, or allowing the tool to snap to nearest anchor boxes.
- **Token budget blowup**: each labeled image increases sequence length. Use low-resolution overlays, limit number of tool calls, or compress history.
- **Vision-language drift**: repeated tool use might overshadow textual reasoning. Keep final answer loss active during RL or add supervised anchors.
- **Environment hacking**: ensure rewards only trigger when boxes genuinely match ground truth; add adversarial checks for mislabeled yet high-IoU boxes.

## Immediate Next Steps
1. Implement `tools/box_labeler.py` and notebook snippets to test rendering.
2. Prototype the prompt schema with a handful of COCO images and verify the model can parse tool tokens.
3. Generate a small supervised dataset (100–200 examples) to sanity-check the training loop.
4. Evaluate zero-shot performance on a “count televisions” subset to establish a baseline before tool training.

## Longer-Term Ideas
- Introduce additional tools (color histograms, segmentation masks) once bounding boxes are reliable.
- Experiment with multi-agent coordination (one agent proposes boxes, another validates) using the same environment scaffolding.
- Investigate human-in-the-loop corrections: log incorrect boxes and allow manual adjustments to augment the training corpus.

## File & Repo Integration
- `tools/box_labeler.py`: rendering utilities.
- `env/labeling_env.py`: RL environment.
- `training/rl_labeling.py`: PPO + LoRA setup.
- `datasets/labeling_tasks/`: scripts to generate synthetic labeling datasets.
- `docs/experiments/labeling_lab.md`: future logs and findings.

Keep NVLR2’s “television counting” failure as the motivating example while building reproducible tooling the lab can reuse for other multimodal reasoning studies.
