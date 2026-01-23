#!/usr/bin/env bash
set -e

PARAM_FILE="run_params.env"

############################################
# LOAD OR COLLECT PARAMETERS
############################################

if [[ -f "$PARAM_FILE" ]]; then
  echo "Loading run parameters from $PARAM_FILE"
  source "$PARAM_FILE"
else
  echo "=============================="
  echo " PRE-TRAINING SETUP "
  echo "=============================="
  echo


  read -p "Dataset (fw / ts / tsk) [fw]: " DATASET
  DATASET=${DATASET:-fw}

  read -p "Model depth (d12 / d20) [d20]: " MODEL_DEPTH
  MODEL_DEPTH=${MODEL_DEPTH:-d20}

  read -p "Batch size (4 / 8 / 16 / 32) [16]: " BATCH_SIZE
  BATCH_SIZE=${BATCH_SIZE:-16}

  read -p "Total batch size [524288]: " TOTAL_BATCH_SIZE
  TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-524288}

  read -p "Max steps (Enter = dataset default (fw: 20000, ts: 1000, tsk: 500)): " TRAIN_STEPS
  TRAIN_STEPS=${TRAIN_STEPS:-1000}

  read -p "Eval Every [$(($TRAIN_STEPS/10))]: " EVAL_EVERY
  EVAL_EVERY=${EVAL_EVERY:-$((TRAIN_STEPS/10))}

  read -p "Save Every [$(($TRAIN_STEPS/25))]: " SAVE_EVERY
  SAVE_EVERY=${SAVE_EVERY:-$((TRAIN_STEPS/25))}

  read -p "Checkpoint directory [./ckps]: " CKPT_DIR
  CKPT_DIR=${CKPT_DIR:-./ckps}

  echo
  echo "=============================="
  echo " MID-TRAINING SETUP "
  echo "=============================="
  echo

  read -p "Max steps [1000]: " MID_STEPS
  MID_STEPS=${MID_STEPS:-1000}

  read -p "Eval Every [$(($MID_STEPS/10))]: " MID_EVAL_EVERY
  MID_EVAL_EVERY=${MID_EVAL_EVERY:-$((MID_STEPS/10))}

  echo
  echo "=============================="
  echo " SFT-TRAINING SETUP "
  echo "=============================="
  echo

  read -p "Max steps [1000]: " SFT_STEPS
  SFT_STEPS=${SFT_STEPS:-1000}

  read -p "Eval Every [$(($SFT_STEPS/10))]: " SFT_EVAL_EVERY
  SFT_EVAL_EVERY=${SFT_EVAL_EVERY:-$((SFT_STEPS/10))}

  ############################################
  # SAVE PARAMETERS
  ############################################

  cat > "$PARAM_FILE" <<EOF
DATASET=$DATASET
MODEL_DEPTH=$MODEL_DEPTH
BATCH_SIZE=$BATCH_SIZE
TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE
TRAIN_STEPS=$TRAIN_STEPS
MID_STEPS=$MID_STEPS
SFT_STEPS=$SFT_STEPS
EVAL_EVERY=$EVAL_EVERY
SAVE_EVERY=$SAVE_EVERY
MID_EVAL_EVERY=$MID_EVAL_EVERY
SFT_EVAL_EVERY=$SFT_EVAL_EVERY
CKPT_DIR=$CKPT_DIR  
EOF

  echo
  echo "Saved run parameters to $PARAM_FILE"
fi

############################################
# DATASET DOWNLOAD
############################################

if [[ "$DATASET" == "fw" || "$DATASET" == "ts" ]]; then
  echo
  echo "Downloading dataset: $DATASET"
  python download/download.py --ds "$DATASET"
else
  echo
  echo "Using local Tiny Shakespeare dataset"
fi

############################################
# PRETRAIN
############################################



# PRETRAIN RESUME LOGIC

# Find latest in-progress checkpoint (pretrain_*_NNNNN.pt)
LATEST_PRETRAIN_CKPT=$(
  ls -1 "${CKPT_DIR}/pretrain_${DATASET}_${MODEL_DEPTH}_"*.pt 2>/dev/null \
  | sort \
  | tail -n 1
)

RESUME_ARGS=()

if [[ -n "$LATEST_PRETRAIN_CKPT" ]]; then
  if [[ -t 0 ]]; then
    read -p "Resume pretraining from checkpoint $(basename "$LATEST_PRETRAIN_CKPT")? [Y/n]: " RESUME_CONFIRM
    RESUME_CONFIRM=${RESUME_CONFIRM:-Y}
  else
    RESUME_CONFIRM="Y"
  fi

  if [[ "$RESUME_CONFIRM" == "Y" || "$RESUME_CONFIRM" == "y" ]]; then
    RESUME_ARGS=(--resume_ckpt "$LATEST_PRETRAIN_CKPT")
  fi
fi


# Launcher command

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  PYTHON_CMD=(
    torchrun
    --standalone
    --nproc_per_node="$NPROC_PER_NODE"
  )
else
  PYTHON_CMD=(python)
fi

echo
echo "Starting pretraining..."

TRAIN_CMD=(
  "${PYTHON_CMD[@]}" train.py
  --dataset "$DATASET"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --max_steps "$TRAIN_STEPS"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --ckpt_out "$CKPT_DIR"
  "${RESUME_ARGS[@]}"
)

"${TRAIN_CMD[@]}"

# PRETRAIN CHECKPOINT

PRETRAIN_CKPT="${CKPT_DIR}/pretrain_${DATASET}_${MODEL_DEPTH}.pt"

if [[ ! -f "$PRETRAIN_CKPT" ]]; then
  echo "ERROR: Expected checkpoint not found:"
  echo "  $PRETRAIN_CKPT"
  exit 1
fi

############################################
# MIDTRAIN
############################################

echo
echo "Starting midtraining..."

MID_CMD=(
  "${PYTHON_CMD[@]}" mid_train.py
  --dataset "$DATASET"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --resume_ckpt "$PRETRAIN_CKPT"
  --max_steps "$MID_STEPS"
  --eval_every "$MID_EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --ckpt_out "$CKPT_DIR"
)

"${MID_CMD[@]}"

echo
echo "✅ Completed"


# MIDTRAIN CHECKPOINT

MIDTRAIN_CKPT="${CKPT_DIR}/midtrain_${DATASET}_${MODEL_DEPTH}.pt"

if [[ ! -f "$MIDTRAIN_CKPT" ]]; then
  echo "ERROR: Expected checkpoint not found:"
  echo "  $MIDTRAIN_CKPT"
  exit 1
fi

############################################
# SFT-TRAIN
############################################

echo
echo "Starting sft-training..."

SFT_CMD=(
  "${PYTHON_CMD[@]}" sft_train.py
  --dataset "$DATASET"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --resume_ckpt "$MIDTRAIN_CKPT"
  --max_steps "$SFT_STEPS"
  --eval_every "$SFT_EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --ckpt_out "$CKPT_DIR"
)

"${SFT_CMD[@]}"

echo
echo "✅ Completed"
