#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG (override via env or flags)
########################################

# Whether to use and mount NVMe instance store
USE_INSTANCE_STORE="${USE_INSTANCE_STORE:-false}"

# Base directory for all caches, venv, repos
BASE_DIR="${BASE_DIR:-$PWD/workspace}"

# Instance store settings (only used if enabled)
INSTANCE_DEVICE="/dev/nvme1n1"
INSTANCE_MOUNT_POINT="/mnt/instancestore"

PYTHON_VERSION="3.11"

########################################
# Helpers
########################################

log() {
  echo "== $1 =="
}

need_sudo() {
  if [[ "$USE_INSTANCE_STORE" == "true" ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

########################################
# System setup
########################################

log "System update & deps"
need_sudo dnf update -y
need_sudo dnf install -y \
  git \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-devel \
  python${PYTHON_VERSION}-pip \
  nvme-cli || true

########################################
# Instance store (optional)
########################################

if [[ "$USE_INSTANCE_STORE" == "true" ]]; then
  log "Using instance store NVMe"

  BASE_DIR="$INSTANCE_MOUNT_POINT"

  need_sudo mkfs.ext4 -F "$INSTANCE_DEVICE" || true
  need_sudo mkdir -p "$INSTANCE_MOUNT_POINT"
  mountpoint -q "$INSTANCE_MOUNT_POINT" || need_sudo mount "$INSTANCE_DEVICE" "$INSTANCE_MOUNT_POINT"
  need_sudo chown -R "$USER:$USER" "$INSTANCE_MOUNT_POINT"
else
  log "Using local directory: $BASE_DIR"
  mkdir -p "$BASE_DIR"
fi

########################################
# Directory layout
########################################

log "Create directories"
mkdir -p \
  "$BASE_DIR"/{pip_cache,torch_extensions} \
  "$BASE_DIR"/hf_cache/{datasets,hub,transformers} \
  "$BASE_DIR"/wandb/{cache,config,run}

########################################
# Python venv
########################################

VENV_DIR="$BASE_DIR/venv"

log "Create Python venv"
rm -rf "$VENV_DIR"
python${PYTHON_VERSION} -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

########################################
# Environment variables
########################################

log "Environment variables"

if [[ "$USE_INSTANCE_STORE" == "true" ]]; then
  # System-wide (EC2-style)
  need_sudo tee /etc/profile.d/ml.sh > /dev/null <<EOF
export PIP_CACHE_DIR=$BASE_DIR/pip_cache

export HF_HOME=$BASE_DIR/hf_cache
export HF_DATASETS_CACHE=$BASE_DIR/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=$BASE_DIR/hf_cache/hub
export TRANSFORMERS_CACHE=$BASE_DIR/hf_cache/transformers

export TORCH_EXTENSIONS_DIR=$BASE_DIR/torch_extensions

export WANDB_DIR=$BASE_DIR/wandb/run
export WANDB_CACHE_DIR=$BASE_DIR/wandb/cache
export WANDB_CONFIG_DIR=$BASE_DIR/wandb/config

export NETRC=$BASE_DIR/.netrc
EOF

  source /etc/profile.d/ml.sh
else
  # Local / Docker friendly
  export PIP_CACHE_DIR="$BASE_DIR/pip_cache"

  export HF_HOME="$BASE_DIR/hf_cache"
  export HF_DATASETS_CACHE="$BASE_DIR/hf_cache/datasets"
  export HUGGINGFACE_HUB_CACHE="$BASE_DIR/hf_cache/hub"
  export TRANSFORMERS_CACHE="$BASE_DIR/hf_cache/transformers"

  export TORCH_EXTENSIONS_DIR="$BASE_DIR/torch_extensions"

  export WANDB_DIR="$BASE_DIR/wandb/run"
  export WANDB_CACHE_DIR="$BASE_DIR/wandb/cache"
  export WANDB_CONFIG_DIR="$BASE_DIR/wandb/config"

  export NETRC="$BASE_DIR/.netrc"
fi

########################################
# netrc
########################################

log "netrc setup"
touch "$BASE_DIR/.netrc"
chmod 600 "$BASE_DIR/.netrc"

########################################
# Cleanup (safe locally)
########################################

log "Cleanup user caches"
rm -rf ~/.cache/{huggingface,wandb,pip} ~/.local/share/wandb || true

########################################
# Clone repo
########################################

cd "$BASE_DIR"
git clone https://github.com/adeelmahmood/nanogpt-to-chat.git || true

########################################
# Validation
########################################

log "Validation"
python - <<EOF
import sys, torch, os
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
print("BASE_DIR:", "$BASE_DIR")
print("HF_HOME:", os.environ.get("HF_HOME"))
print("WANDB_DIR:", os.environ.get("WANDB_DIR"))
print("NETRC:", os.environ.get("NETRC"))
EOF

########################################
# Install requirements
########################################

cd "$BASE_DIR/nanogpt-to-chat"
pip install --upgrade pip
pip install -r requirements.txt
