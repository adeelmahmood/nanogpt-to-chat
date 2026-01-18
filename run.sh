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
  echo " TRAINING PIPELINE SETUP "
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

  read -p "Pretraining max steps (Enter = dataset default (fw: 20000, ts: 1000, tsk: 500)): " TRAIN_STEPS
  TRAIN_STEPS=${TRAIN_STEPS:-1000}

  read -p "Midtraining max steps [1000]: " MID_STEPS
  MID_STEPS=${MID_STEPS:-1000}

  read -p "Validation Frequency % [10]: " VAL_FREQ
  VAL_FREQ=${VAL_FREQ:-10}

  read -p "Checkpoint directory [./ckps]: " CKPT_DIR
  CKPT_DIR=${CKPT_DIR:-./ckps}

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
VAL_FREQ=$VAL_FREQ
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

echo
echo "Starting pretraining..."

TRAIN_CMD=(
  python train.py
  --dataset "$DATASET"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --max_steps "$TRAIN_STEPS"
  --val_freq "$VAL_FREQ"
  --ckpt_out "$CKPT_DIR"
)

"${TRAIN_CMD[@]}"

############################################
# PRETRAIN CHECKPOINT
############################################

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
  python mid_train.py
  --dataset "$DATASET"
  --model_depth "$MODEL_DEPTH"
  --batch_size "$BATCH_SIZE"
  --total_batch_size "$TOTAL_BATCH_SIZE"
  --resume_ckpt "$PRETRAIN_CKPT"
  --max_steps "$MID_STEPS"
  --val_freq "$VAL_FREQ"    
  --ckpt_out "$CKPT_DIR"
)

"${MID_CMD[@]}"

echo
echo "✅ Completed"
