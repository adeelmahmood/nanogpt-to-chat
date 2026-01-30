#!/usr/bin/env bash
set -e

############################################
# LOAD RUN PARAMS
############################################

PARAM_FILE="run_params_multi.env"
if [[ ! -f "$PARAM_FILE" ]]; then
  echo "Missing $PARAM_FILE (required)."
  exit 1
fi
source "$PARAM_FILE"

export RUN_ID="$RUN_ID"

RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR"

############################################
# DATASET SETUP
############################################

if [[ "$DATASET" == "fw" ]]; then
  DATASET_NAME="HuggingFaceFW/fineweb-edu"
  REMOTE_NAME="sample-10BT"
  DATASET_DIR="download/edu_fineweb10B"
elif [[ "$DATASET" == "ts" ]]; then
  DATASET_NAME="roneneldan/TinyStories"
  REMOTE_NAME=""
  DATASET_DIR="download/tinystories"
else
  echo "ERROR: Invalid dataset: $DATASET"
  exit 1
fi

# Only populate dataset if local dir is empty
if [[ -z "$(ls -A "${DATASET_DIR}" 2>/dev/null)" ]]; then

  if [[ -n "${BUCKET:-}" ]]; then
    echo "Syncing dataset from S3..."
    aws s3 sync "s3://$BUCKET/${DATASET_DIR##download/}/" "$DATASET_DIR/" --only-show-errors || true
  fi

  # If still empty, download from source
  if [[ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
    echo "Downloading dataset..."
    python download/download.py \
      --ds "$DATASET" \
      --dataset_name "$DATASET_NAME" \
      --remote_name "$REMOTE_NAME" \
      --local_dir "$DATASET_DIR/"

    echo "Syncing dataset back to S3..."
    if [[ -n "${BUCKET:-}" ]]; then
      aws s3 sync "$DATASET_DIR/" "s3://$BUCKET/${DATASET_DIR##download/}/" --only-show-errors || true
    fi
  fi
fi


############################################
# CHECKPOINT PRELOAD
############################################

if [[ -n "${BUCKET:-}" ]]; then
  echo
  echo "Retrieving existing checkpoints from S3..."
  aws s3 sync "s3://$BUCKET/${RUN_DIR}/checkpoints/" "${RUN_DIR}/checkpoints/" || true
fi

############################################
# BACKGROUND CHECKPOINT SYNC (5 min)
############################################

if [[ -n "${BUCKET:-}" ]]; then
  (
    while true; do
      aws s3 sync "${RUN_DIR}/checkpoints/" \
        "s3://$BUCKET/${RUN_DIR}/checkpoints/" \
        --exclude "*.tmp" \
        --only-show-errors || true
      sleep 300
    done
  ) &
  SYNC_PID=$!
  trap "kill $SYNC_PID 2>/dev/null || true" EXIT
fi

############################################
# RESUME LOGIC
############################################

LATEST_PRETRAIN_CKPT=$(
  ls -1 "${RUN_DIR}/checkpoints/pretrain_${DATASET}_${MODEL_DEPTH}_"*.pt 2>/dev/null \
  | grep -v "\.rank*" \
  | sort \
  | tail -n 1
)

echo 
echo "Latest pretrain checkpoint: $LATEST_PRETRAIN_CKPT"

RESUME_ARGS=()

if [[ -n "$LATEST_PRETRAIN_CKPT" ]]; then
  if [[ "${AUTO_RESUME_PRETRAIN:-Y}" =~ ^[Yy]$ ]]; then
    RESUME_ARGS=(--resume_ckpt "$LATEST_PRETRAIN_CKPT")
    echo "Resuming from $(basename "$LATEST_PRETRAIN_CKPT")"
  fi
fi

############################################
# LAUNCHER
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
# PRETRAIN
############################################

echo
echo "Starting pretraining..."

PRETRAIN_CMD=(
  "${PYTHON_CMD[@]}" train.py
  --dataset "$DATASET"
  --dataset_dir "${DATASET_DIR}"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --max_steps "$TRAIN_STEPS"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --ckpt_out "${RUN_DIR}/checkpoints"
  "${RESUME_ARGS[@]}"
)

echo "Running command: ${PRETRAIN_CMD[*]}"
"${PRETRAIN_CMD[@]}"

PRETRAIN_CKPT="${RUN_DIR}/checkpoints/pretrain_${DATASET}_${MODEL_DEPTH}.pt"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: Pretrain checkpoint missing"
  exit 1
fi

############################################
# EVAL
############################################

echo
echo "Running evaluation..."
python eval.py --model_file "$PRETRAIN_CKPT" --model_depth "$MODEL_DEPTH"

############################################
# SAMPLING
############################################

echo
echo "Running sampling..."
python sample.py --model_file "$PRETRAIN_CKPT" --model_depth "$MODEL_DEPTH"

############################################
# FINAL SYNC
############################################

if [[ -n "${BUCKET:-}" ]]; then
  echo
  echo "Final checkpoint sync..."
  aws s3 sync "${RUN_DIR}/checkpoints/" \
    "s3://$BUCKET/${RUN_DIR}/checkpoints/" \
    --exclude "*.tmp" \
    --only-show-errors || true
fi

echo
echo "✅ Run complete"
