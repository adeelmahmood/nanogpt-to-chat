#!/usr/bin/env bash
set -e

# Simplified runner for ablation experiments
# No S3 syncing, no complex dataset management, just pure training

############################################
# LOAD RUN PARAMS
############################################

PARAM_FILE="${1:-params/run_params_local.env}"
if [[ ! -f "$PARAM_FILE" ]]; then
  echo "Missing $PARAM_FILE (required)."
  exit 1
fi
source "$PARAM_FILE"

# Ensure RUN_ID is set
if [[ -z "${RUN_ID:-}" ]]; then
  echo "ERROR: RUN_ID must be set in parameter file"
  exit 1
fi

export RUN_ID="$RUN_ID"
RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR/checkpoints"

echo "Starting ablation run: $RUN_ID"
echo "Run directory: $RUN_DIR"

############################################
# SIMPLE DATASET SETUP
############################################

if [[ "$DATASET" == "fw" ]]; then
  DATASET_DIR="download/edu_fineweb10B"
elif [[ "$DATASET" == "ts" ]]; then
  DATASET_DIR="download/tinystories"
else
  echo "ERROR: Invalid dataset: $DATASET"
  exit 1
fi

# Just check if dataset exists - don't try to download
if [[ ! -d "$DATASET_DIR" ]] || [[ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: Dataset not found at $DATASET_DIR"
  echo "Please download the dataset first using the main run scripts"
  exit 1
fi

############################################
# LAUNCHER DETECTION
############################################

if command -v nvidia-smi >/dev/null 2>&1; then
  NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
else
  NPROC_PER_NODE=${NPROC_PER_NODE:-1}
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  PYTHON_CMD=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE")
else
  PYTHON_CMD=(python)
fi

############################################
# BUILD TRAINING COMMAND
############################################

# Build architecture arguments if they're set
ARCH_ARGS=()
if [[ -n "${use_rope:-}" ]]; then
    ARCH_ARGS+=(--use_rope "$use_rope")
fi
if [[ -n "${use_rmsnorm:-}" ]]; then
    ARCH_ARGS+=(--use_rmsnorm "$use_rmsnorm")
fi
if [[ -n "${use_qk_norm:-}" ]]; then
    ARCH_ARGS+=(--use_qk_norm "$use_qk_norm")
fi
if [[ -n "${use_gqa:-}" ]]; then
    ARCH_ARGS+=(--use_gqa "$use_gqa")
fi
if [[ -n "${use_kv_cache:-}" ]]; then
    ARCH_ARGS+=(--use_kv_cache "$use_kv_cache")
fi

# Resume arguments (if checkpoint exists)
RESUME_ARGS=()
if [[ -n "${resume_ckpt:-}" ]] && [[ -f "$resume_ckpt" ]]; then
    RESUME_ARGS+=(--resume_ckpt "$resume_ckpt")
fi

TRAIN_CMD=(
  "${PYTHON_CMD[@]}" train.py
  --dataset "$DATASET"
  --dataset_dir "$DATASET_DIR"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --max_steps "$TRAIN_STEPS"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --ckpt_out "$RUN_DIR/checkpoints"
  --log_dir "${log_dir:-$RUN_DIR}"
  --log_file "${log_file:-train.jsonl}"
  "${ARCH_ARGS[@]}"
  "${RESUME_ARGS[@]}"
)

############################################
# RUN TRAINING
############################################

echo
echo "Starting training..."
echo "Command: ${TRAIN_CMD[*]}"
echo

"${TRAIN_CMD[@]}"

# Verify final checkpoint
FINAL_CKPT="$RUN_DIR/checkpoints/pretrain_${DATASET}_${MODEL_DEPTH}.pt"
if [[ ! -f "$FINAL_CKPT" ]]; then
  echo "ERROR: Final checkpoint missing: $FINAL_CKPT"
  exit 1
fi

echo
echo "✅ Training complete!"
echo "📁 Run directory: $RUN_DIR"
echo "📊 Logs: $RUN_DIR/train.jsonl"
echo "💾 Checkpoint: $FINAL_CKPT"