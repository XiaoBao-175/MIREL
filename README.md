# MIREL: Learning Flexible Evidence Use for Multi-Image Reasoning

MIREL is a multi-image reasoning model family trained with structured visual
evidence and **PG-OPD** (**Privilege-Gain On-Policy Distillation**). The release
contains Qwen3.5-4B and Qwen3.5-9B recipes, processed training data, merged
checkpoints, and the patched `verl` implementation used in our experiments.

The processed data is available at
[luo352/MIREL-data](https://huggingface.co/datasets/luo352/MIREL-data).

## Overview

MIREL first teaches a model to produce structured evidence for multi-image
questions. PG-OPD then transfers the useful part of verified evidence to a
native full-image student without exposing privilege at inference time.

For a training row, the student and native teacher receive the original prompt
and images. The privileged teacher receives the same input plus verified
evidence-only text. The privileged text contains no ground-truth answer,
answer option, `support=[...]` locator, synthesis, or final-answer block.

For each on-policy student token, PG-OPD computes the teacher gap:

```text
gain_t = log p_privileged(y_t | prefix_t)
         - log p_native(y_t | prefix_t)
```

Only positive privilege gains increase the distillation weight. The objective is
reward-free: no GRPO answer reward, verifier, or inference-time privilege is
used. Rows without verified privilege follow the native-teacher path.

## Repository structure

```text
final/
├── environment.yml       # reference Conda environment
├── environment.toml      # auditable version and experiment settings
├── README.md             # this file
├── models/
│   ├── MIREL-4B/         # merged Hugging Face checkpoint
│   └── MIREL-9B/         # merged Hugging Face checkpoint
├── data/
│   ├── SFT/              # 2,698 verified structured-CoT rows
│   ├── Qwen35-4B/        # 2,394 4B PG-OPD rows
│   └── Qwen35-9B/        # 3,000 9B PG-OPD rows
└── code/
    ├── PG-OPD/           # patched verl implementation
    ├── train/             # SFT and PG-OPD launchers
    └── eval/              # evaluation runners and parsers
```

## Environment setup

The reference environment uses Python 3.12, CUDA 12.8 PyTorch wheels, vLLM,
Ray, and Flash-Attention. The host needs an NVIDIA driver that supports CUDA
12.8.

```bash
conda env create -f environment.yml
conda activate mirel-pg-opd
git lfs install
python -m pip install --no-deps -e code/PG-OPD
```

`environment.yml` is the executable Conda specification. `environment.toml`
records the same versions and the main training/evaluation settings in a
human-readable format.

Verify the installation:

```bash
python - <<'PY'
import torch
import transformers
import vllm
import ray
import flash_attn
import verl

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("ray", ray.__version__)
print("flash-attn", flash_attn.__version__)
print("verl", verl.__file__)
PY
```

Reference versions are PyTorch `2.10.0+cu128`, Transformers `5.5.0`, vLLM
`0.18.0`, Ray `2.53.0`, and Flash-Attention `2.8.3.post1`.

## Quick start

Run the following commands from the root of the release checkout:

```bash
cd /path/to/final
conda env create -f environment.yml
conda activate mirel-pg-opd
python -m pip install --no-deps -e code/PG-OPD

# Download the processed parquet files.
hf download luo352/MIREL-data --repo-type dataset --local-dir data

# After downloading the licensed M4/Mantis image archives, put them under:
#   data/images/m4/
#   data/images/mantis/

export ENV_ROOT="$CONDA_PREFIX"
export WANDB_MODE=online
export WANDB_API_KEY=YOUR_WANDB_API_KEY

# SFT, then use its checkpoint as MODEL_PATH for PG-OPD.
export MODEL_PATH=Qwen/Qwen3.5-4B
bash code/train/run_sft_qwen35_4b.sh

export MODEL_PATH=/path/to/sft/checkpoint
bash code/train/run_pg_opd_qwen35_4b.sh
```

For Qwen3.5-9B, use the LoRA recipes:

```bash
export MODEL_PATH=Qwen/Qwen3.5-9B
bash code/train/run_sft_qwen35_9b_lora.sh

export MODEL_PATH=/path/to/merged/9b/sft/checkpoint
bash code/train/run_pg_opd_qwen35_9b_lora.sh
```

Training outputs are written below `runs/` by default. SFT and PG-OPD save
checkpoints every 100 steps and keep at most two checkpoints. The 9B PG-OPD
actor uses a new LoRA adapter; the native and privileged teachers are frozen.

## Data

Download the processed release data with:

```bash
hf download luo352/MIREL-data \
  --repo-type dataset \
  --local-dir data
```

The files are model-specific and are not interchangeable:

| File | Rows | Description |
|---|---:|---|
| `data/SFT/verified_cot.parquet` | 2,698 | Deduplicated verified structured-CoT SFT data |
| `data/Qwen35-4B/pg_opd.parquet` | 2,394 | 1,197 privilege rows + 1,197 native fallback rows |
| `data/Qwen35-9B/pg_opd.parquet` | 3,000 | 1,500 9B-verified privilege rows + 1,500 native controls |

The parquet files contain repository-relative image references. The original
image archives are not redistributed. Download the source datasets from their
official releases and place or map the images as follows:

```text
data/images/m4/...
data/images/mantis/...
```

The processed data uses these source task groups:

- M4-Instruct: `MIT-States_PropertyCoherence`, `MIT-States_StateCoherence`,
  `COMICS_Dialogue`, `DocVQA`, and `OCR-VQA`;
- Mantis-Instruct: `Mantis-dreamsim` and `Mantis-imagecode`.

Official source references:

- [Mantis project](https://github.com/TIGER-AI-Lab/Mantis)
- [Mantis-Instruct dataset](https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct)
- [M4-Instruct release reference](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)

If the downloaded images live elsewhere, run the path checker from
`code/PG-OPD` and rewrite only the image fields with:

```bash
cd code/PG-OPD
python scripts/rewrite_image_paths.py \
  --input ../../data/Qwen35-4B/pg_opd.parquet \
  --old-prefix ../../data/images \
  --new-prefix /path/to/local/images \
  --check-only
```

If the check passes, omit `--check-only` and provide `--output` to create a
local-path copy. Preserve the original image file names.

## Training

Set the model path and environment explicitly when using local checkpoints:

```bash
export ENV_ROOT="$CONDA_PREFIX"
export MODEL_PATH=Qwen/Qwen3.5-4B
export WANDB_MODE=online
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```

Run SFT for Qwen3.5-4B:

```bash
bash code/train/run_sft_qwen35_4b.sh
```

Run LoRA SFT for Qwen3.5-9B:

```bash
export MODEL_PATH=Qwen/Qwen3.5-9B
bash code/train/run_sft_qwen35_9b_lora.sh
```

Use the resulting SFT checkpoint as the initial model for PG-OPD. The 4B
recipe is full-parameter PG-OPD:

```bash
export MODEL_PATH=/path/to/MIREL-4B-SFT
bash code/train/run_pg_opd_qwen35_4b.sh
```

The 9B recipe uses a new LoRA adapter on top of the merged SFT checkpoint; the
native and privileged teachers are frozen copies of that same checkpoint:

```bash
export MODEL_PATH=/path/to/MIREL-9B-SFT-merged
bash code/train/run_pg_opd_qwen35_9b_lora.sh
```

The launchers default to one epoch, disabled thinking, WandB online logging,
checkpoint saving every 100 steps, and retention of at most two actor
checkpoints. Override paths and resource settings with the variables documented
inside each launcher.

## Evaluation

The evaluator uses an OpenAI-compatible vLLM server and the audited benchmark
parsers. The reference settings are eight-way tensor parallelism, temperature
`0`, maximum output length `512`, and disabled thinking.

For the current research checkout, point the wrapper at the surrounding
benchmark-input/orchestration layout with `MIREL_EXPERIMENT_ROOT`:

```bash
export MIREL_EXPERIMENT_ROOT=/path/to/multi-image-experiment

bash code/eval/run_mirel_eval.sh \
  --model-name MIREL-4B \
  --suite multi \
  --include-mmiu \
  --max-model-len 160000
```

For the general suite:

```bash
bash code/eval/run_mirel_eval.sh \
  --model-name MIREL-9B \
  --suite general \
  --max-model-len 32768
```

To evaluate one of the packaged checkpoints directly, use `--model` instead
of `--model-name`:

```bash
bash code/eval/run_mirel_eval.sh \
  --model "$PWD/models/MIREL-4B" \
  --tag mirel_4b \
  --suite multi \
  --max-model-len 32768
```

Prepared benchmark JSONL files and their image assets must be downloaded from
the corresponding official benchmark releases. Keep the generated prediction
JSONL files and per-benchmark summaries for auditability.

## Reproducibility checklist

1. Use the pinned environment in `environment.yml`.
2. Use the model-specific PG-OPD parquet file.
3. Verify every image path before training or evaluation.
4. Keep thinking disabled and use the same chat template.
5. Use the same benchmark split, input JSONL, parser, and answer normalization.
6. Record the model path, data path, git revision, GPU count, and WandB run.

## Citation

If you use the released implementation or data, please cite the MIREL paper
when it becomes available. The PG-OPD implementation is based on the open
source [Vision-OPD](https://github.com/VisionOPD/Vision-OPD) codebase.

## License

The PG-OPD code retains the upstream Apache-2.0 license. The training and
evaluation data remain subject to the licenses and terms of their original
M4-Instruct, Mantis-Instruct, and benchmark releases.
