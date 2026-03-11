#!/usr/bin/env bash
set -e

# CONSTANT 
DATASET=fw
MODEL_DEPTH=d12
BATCH_SIZE=4
TOTAL_BATCH_SIZE=32768
SAVE_EVERY=-1
EVAL_EVERY=-1

export RUNNING_IN_AUTO_MODE=1
export AUTO_MODE_TIME_LIMIT=300  # 5 mins

# Generate RUN_ID if not provided
if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="$(date +%-m-%-d)_${DATASET}_${MODEL_DEPTH}"
fi
export RUN_ID="$RUN_ID"

RUN_DIR="runs/${RUN_ID}"
CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# ---------- dataset metadata ----------
if [[ "$DATASET" == "fw" ]]; then
  DATASET_NAME="HuggingFaceFW/fineweb-edu"
  REMOTE_NAME="sample-10BT"
  DATASET_DIR="download/edu_fineweb10B"
elif [[ "$DATASET" == "ts" ]]; then
  DATASET_NAME="roneneldan/TinyStories"
  REMOTE_NAME=""
  DATASET_DIR="download/tinystories"
else
  echo "Invalid dataset: $DATASET"; exit 1
fi

mkdir -p "$DATASET_DIR" "$CKPT_DIR"

# ---------- dataset sync / download ----------
if [[ -n "${BUCKET:-}" ]]; then
  echo "Syncing dataset from S3..."
  aws s3 sync "s3://$BUCKET/${DATASET_DIR##download/}/" "$DATASET_DIR/" --only-show-errors || true
fi

if [[ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
  echo "Downloading dataset..."
  python download/download.py \
    --dataset_name "$DATASET_NAME" \
    --remote_name "$REMOTE_NAME" \
    --local_dir "$DATASET_DIR/"

  echo "Syncing dataset back to S3..."
  if [[ -n "${BUCKET:-}" ]]; then
    aws s3 sync "$DATASET_DIR/" "s3://$BUCKET/${DATASET_DIR##download/}/" --only-show-errors || true
  fi  
fi


# ---------- helper: detect launcher ----------
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


echo; echo "Running pretraining..."
PRETRAIN_CMD=("${PYTHON_CMD[@]}" train.py 
  --dataset "$DATASET" 
  --dataset_dir "$DATASET_DIR" 
  --model_depth "$MODEL_DEPTH" 
  --batch_size "$BATCH_SIZE" 
  --save_every "$SAVE_EVERY"
  --eval_every "$EVAL_EVERY"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  )
echo "Command: ${PRETRAIN_CMD[*]}"
"${PRETRAIN_CMD[@]}"
