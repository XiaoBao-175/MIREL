# MIREL: Learning Flexible Evidence Use for Multi-Image Reasoning

MIREL is a multi-image reasoning model family trained with structured visual
evidence and PG-OPD (Privilege-Gain On-Policy Distillation). The release
contains the key SFT and PG-OPD training code together with the processed
training files.

## Environment

```bash
conda env create -f environment.yml
conda activate mirel-pg-opd
python -m pip install --no-deps -e code/PG-OPD
```

The reference environment uses Python 3.12, PyTorch with CUDA 12.8,
Transformers 5.5.0, vLLM 0.18.0, Ray 2.53.0, and Flash-Attention 2.8.3.

## Prepare

Download the processed SFT and PG-OPD files from
[`luo352/MIREL-data`](https://huggingface.co/datasets/luo352/MIREL-data), and
prepare the licensed M4 and Mantis images required by the parquet paths.

```bash
cd /path/to/final
hf download luo352/MIREL-data --repo-type dataset --local-dir data

export ENV_ROOT="$CONDA_PREFIX"
export WANDB_MODE=online
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```

## SFT

Run one epoch of SFT on the verified structured-CoT data:

```bash
export MODEL_PATH=Qwen/Qwen3.5-4B
bash code/train/run_sft_qwen35_4b.sh
```

For Qwen3.5-9B, use the LoRA SFT launcher:

```bash
export MODEL_PATH=Qwen/Qwen3.5-9B
bash code/train/run_sft_qwen35_9b_lora.sh
```

The 9B SFT defaults to LoRA rank 32 and LoRA alpha 16.

## PG-OPD

Set `MODEL_PATH` to the corresponding SFT checkpoint. The 4B recipe trains
all parameters; the 9B recipe initializes a new LoRA adapter on top of the
merged SFT checkpoint. Both launchers use JSD with β=0.5, one epoch,
disabled thinking, and no GRPO reward.

```bash
export MODEL_PATH=/path/to/qwen35-4b-sft
bash code/train/run_pg_opd_qwen35_4b.sh
```

```bash
export MODEL_PATH=/path/to/qwen35-9b-sft-merged
bash code/train/run_pg_opd_qwen35_9b_lora.sh
```

Checkpoints are saved every 100 steps and at most two actor checkpoints are
kept by default. Training logs are sent to Weights & Biases when online mode
is enabled.

## Evaluation

Prepare the benchmark inputs and image assets, then use the evaluation entry
under `code/eval`. The evaluator supports both the multi-image benchmark suite
and the general-ability suite.

```bash
bash code/eval/run_mirel_eval.sh \
  --model /path/to/model \
  --suite multi
```

The PG-OPD implementation is released under the Apache-2.0 license.
