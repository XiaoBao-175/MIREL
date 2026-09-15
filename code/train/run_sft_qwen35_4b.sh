#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-4B}
DATA_PATH=${DATA_PATH:-$FINAL_ROOT/data/SFT/verified_cot.parquet}
RUN_ROOT=${RUN_ROOT:-$FINAL_ROOT/runs/sft_qwen35_4b_cot_2698_1epoch}
RUN_NAME=${RUN_NAME:-$(basename "$RUN_ROOT")}

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/wandb"
export WANDB_DIR="$RUN_ROOT/wandb"

exec "$PYTHON" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_port="${MASTER_PORT:-29574}" \
  -m verl.trainer.sft_trainer \
  data.train_files="$DATA_PATH" \
  data.val_files=null \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-1}" \
  data.max_length="${MAX_LENGTH:-8192}" \
  data.pad_mode=no_padding \
  data.truncation=right \
  data.use_dynamic_bsz=True \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU:-16384}" \
  data.train_max_samples=-1 \
  data.messages_key=messages \
  +data.image_key=images \
  +data.image_patch_size=16 \
  data.num_workers="${NUM_WORKERS:-4}" \
  data.ignore_input_ids_mismatch=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  model.path="$MODEL_PATH" \
  model.trust_remote_code=True \
  model.lora_rank=0 \
  model.use_remove_padding=True \
  model.enable_gradient_checkpointing=True \
  model.use_fused_kernels=False \
  engine=fsdp \
  optim=fsdp \
  optim.lr="${LR:-2e-6}" \
  optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO:-0.05}" \
  optim.weight_decay=0.01 \
  optim.betas='[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.lr_scheduler_type=cosine \
  engine.strategy=fsdp2 \
  engine.ulysses_sequence_parallel_size=1 \
  engine.fsdp_size=-1 \
  engine.reshard_after_forward=True \
  engine.forward_prefetch=False \
  engine.precompute_fa_varlen_kwargs=True \
  engine.mixed_precision.param_dtype=bf16 \
  engine.mixed_precision.reduce_dtype=fp32 \
  engine.mixed_precision.buffer_dtype=fp32 \
  engine.dtype=bfloat16 \
  trainer.project_name="${WANDB_PROJECT:-MIREL}" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_ROOT/checkpoints" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.test_freq=-1 \
  trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-2}" \
  trainer.resume_mode=disable \
  trainer.logger='["console","wandb"]' \
  trainer.n_gpus_per_node="${NPROC_PER_NODE:-8}" \
  trainer.nnodes=1
