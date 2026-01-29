#!/usr/bin/env bash
set -e

# Single-run pre -> mid -> sft pipeline (interactive prompts, robust syncs)

# check if params file exists then source tat
PARAM_FILE="run_params.env"
if [[ -f "$PARAM_FILE" ]]; then
  echo "Loading parameters from $PARAM_FILE"
  source "$PARAM_FILE"
else
  # ---------- params (interactive with sensible defaults) ----------
  read -p "Dataset (fw / ts) [fw]: " DATASET; DATASET=${DATASET:-fw}
  read -p "Model depth (d12 / d20) [d12]: " MODEL_DEPTH; MODEL_DEPTH=${MODEL_DEPTH:-d12}
  read -p "Batch size (4/8/16/32) [16]: " BATCH_SIZE; BATCH_SIZE=${BATCH_SIZE:-16}
  read -p "Total batch size [524288]: " TOTAL_BATCH_SIZE; TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-524288}
  read -p "Pretrain max steps (Enter = default) [1000]: " TRAIN_STEPS; TRAIN_STEPS=${TRAIN_STEPS:-1000}
  read -p "Eval every: " EVAL_EVERY; EVAL_EVERY=${EVAL_EVERY:-$((TRAIN_STEPS/10))}
  read -p "Save every: " SAVE_EVERY; SAVE_EVERY=${SAVE_EVERY:-$((TRAIN_STEPS*0.25))}
  read -p "Checkpoint dir [./ckps]: " CKPT_DIR; CKPT_DIR=${CKPT_DIR:-./ckps}
  read -p "S3 bucket for sync (leave empty to skip S3): " BUCKET

  read -p "Midtrain max steps [1000]: " MID_STEPS; MID_STEPS=${MID_STEPS:-1000}
  read -p "Mid eval every: " MID_EVAL_EVERY; MID_EVAL_EVERY=${MID_EVAL_EVERY:-$((MID_STEPS/10))}

  read -p "SFT max steps [1000]: " SFT_STEPS; SFT_STEPS=${SFT_STEPS:-1000}
  read -p "SFT eval every: " SFT_EVAL_EVERY; SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-$((SFT_STEPS/10))}
fi

RUN_DIR="runs/$(date +%Y%m%d_%H)_${DATASET}_${MODEL_DEPTH}"
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
    --ds "$DATASET" \
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

# ---------- resume helper ----------
find_latest_ckpt() {
  pattern="$1"
  ls -1 $pattern 2>/dev/null  | grep -v "\.rank*" | sort | tail -n 1 || true
}

# ---------- PRETRAIN ----------
LATEST_PRETRAIN_CKPT=$(find_latest_ckpt "${CKPT_DIR}/pretrain_${DATASET}_${MODEL_DEPTH}_"*.pt)

RESUME_ARGS=()

if [[ -n "$LATEST_PRETRAIN_CKPT" ]]; then
  RESUME_ARGS=(--resume_ckpt "$LATEST_PRETRAIN_CKPT")
fi


echo; echo "Running pretraining..."
PRETRAIN_CMD=("${PYTHON_CMD[@]}" train.py 
  --dataset "$DATASET" 
  --dataset_dir "$DATASET_DIR" 
  --model_depth "$MODEL_DEPTH" 
  --batch_size "$BATCH_SIZE" 
  --total_batch_size "$TOTAL_BATCH_SIZE" 
  --max_steps "$TRAIN_STEPS" 
  --eval_every "$EVAL_EVERY" 
  --save_every "$SAVE_EVERY" 
  --ckpt_out "$CKPT_DIR" 
  "${RESUME_ARGS[@]}")
echo "Command: ${PRETRAIN_CMD[*]}"
"${PRETRAIN_CMD[@]}"

# sync checkpoints
if [[ -n "${BUCKET:-}" ]]; then
  echo "Syncing pretrain checkpoints to S3..."
  aws s3 sync "$CKPT_DIR/" "s3://$BUCKET/${RUN_DIR}/checkpoints/" --exclude "*.tmp" --only-show-errors || true
fi

PRETRAIN_CKPT="${CKPT_DIR}/pretrain_${DATASET}_${MODEL_DEPTH}.pt"
if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "Pretrain checkpoint missing: $PRETRAIN_CKPT"; exit 1
fi

# ---------- MIDTRAIN ----------
echo; echo "Running midtraining..."
MIDTRAIN_CMD=("${PYTHON_CMD[@]}" mid_train.py \
  --dataset "$DATASET" \
  --model_depth "$MODEL_DEPTH" \
  --batch_size "$BATCH_SIZE" \
  --total_batch_size "$TOTAL_BATCH_SIZE" \
  --resume_ckpt "$PRETRAIN_CKPT" \
  --max_steps "$MID_STEPS" \
  --eval_every "$MID_EVAL_EVERY" \
  --save_every "$SAVE_EVERY" \
  --ckpt_out "$CKPT_DIR")
echo "Command: ${MIDTRAIN_CMD[*]}"
"${MIDTRAIN_CMD[@]}"

# sync checkpoints
if [[ -n "${BUCKET:-}" ]]; then
  echo "Syncing midtrain checkpoints to S3..."
  aws s3 sync "$CKPT_DIR/" "s3://$BUCKET/${RUN_DIR}/checkpoints/" --exclude "*.tmp" --only-show-errors || true
fi

MIDTRAIN_CKPT="${CKPT_DIR}/midtrain_${DATASET}_${MODEL_DEPTH}.pt"
if [[ ! -f "$MIDTRAIN_CKPT" ]]; then
  echo "Midtrain checkpoint missing: $MIDTRAIN_CKPT"; exit 1
fi

# ---------- SFT ----------
echo; echo "Running sft-training..."
SFTTRAIN_CMD=("${PYTHON_CMD[@]}" sft_train.py \
  --dataset "$DATASET" \
  --model_depth "$MODEL_DEPTH" \
  --batch_size "$BATCH_SIZE" \
  --total_batch_size "$TOTAL_BATCH_SIZE" \
  --resume_ckpt "$MIDTRAIN_CKPT" \
  --max_steps "$SFT_STEPS" \
  --eval_every "$SFT_EVAL_EVERY" \
  --save_every "$SAVE_EVERY" \
  --ckpt_out "$CKPT_DIR")
echo "Command: ${SFTTRAIN_CMD[*]}"
"${SFTTRAIN_CMD[@]}"


# sync checkpoints
if [[ -n "${BUCKET:-}" ]]; then
  echo "Syncing sfttrain checkpoints to S3..."
  aws s3 sync "$CKPT_DIR/" "s3://$BUCKET/${RUN_DIR}/checkpoints/" --exclude "*.tmp" --only-show-errors || true
fi

SFT_TRAIN_CKPT="${CKPT_DIR}/sfttrain_${DATASET}_${MODEL_DEPTH}.pt"
if [[ ! -f "$SFT_TRAIN_CKPT" ]]; then
  echo "SFT checkpoint missing: $SFT_TRAIN_CKPT"; exit 1
fi

# ---------- EVAL & SAMPLE ----------
echo; echo "Running evaluation..."
python eval.py --model_file "$SFT_TRAIN_CKPT" --model_depth "$MODEL_DEPTH"

echo; echo "Running sampling..."
python sample.py --model_file "$SFT_TRAIN_CKPT" --model_depth "$MODEL_DEPTH"

echo; echo "✅ All stages complete."
