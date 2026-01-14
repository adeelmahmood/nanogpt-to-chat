#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################
MOUNT_POINT="/mnt/instancestore"
VENV_DIR="$MOUNT_POINT/venv"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

########################################
# 1. System update & base packages
########################################
sudo dnf update -y

sudo dnf install -y \
  git \
  nvme-cli \
  python3.10 \
  python3.10-devel \
  python3.10-pip

########################################
# 2. Find FIRST unmounted instance-store NVMe disk
#    (skip root NVMe which is already mounted)
########################################
NVME_DISK=$(lsblk -dn -o NAME,SIZE,MOUNTPOINT \
  | awk '$2 ~ /G/ && $2+0 > 100 && $3 == "" {print $1; exit}')

if [ -z "$NVME_DISK" ]; then
  echo "❌ No unmounted instance-store NVMe disk found"
  exit 1
fi

NVME_DEV="/dev/$NVME_DISK"
echo "Using instance-store disk: $NVME_DEV"

########################################
# 3. Format disk if needed
########################################
if ! sudo blkid "$NVME_DEV" >/dev/null 2>&1; then
  sudo mkfs.ext4 -F "$NVME_DEV"
fi

########################################
# 4. Mount disk
########################################
sudo mkdir -p "$MOUNT_POINT"

if ! mountpoint -q "$MOUNT_POINT"; then
  sudo mount "$NVME_DEV" "$MOUNT_POINT"
fi

sudo chown -R "$USER:$USER" "$MOUNT_POINT"

########################################
# 5. Directory layout
########################################
mkdir -p \
  "$MOUNT_POINT/pip_cache" \
  "$MOUNT_POINT/hf_cache/datasets" \
  "$MOUNT_POINT/hf_cache/hub" \
  "$MOUNT_POINT/hf_cache/transformers" \
  "$MOUNT_POINT/torch_extensions"

########################################
# 6. Create Python 3.10 venv on instance store
########################################
if [ ! -d "$VENV_DIR" ]; then
  python3.10 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

########################################
# 7. Python packages
########################################
pip install --upgrade pip wheel

pip install torch torchvision torchaudio \
  --index-url "$TORCH_INDEX_URL"

pip install \
  tiktoken \
  datasets \
  transformers \
  importlib-metadata

########################################
# 8. Persist environment variables
########################################
BASHRC="$HOME/.bashrc"

grep -q "Instance store ML setup" "$BASHRC" || {
  cat >> "$BASHRC" <<EOF

# ===== Instance store ML setup =====
export PIP_CACHE_DIR=$MOUNT_POINT/pip_cache

# Hugging Face caches
export HF_HOME=$MOUNT_POINT/hf_cache
export HF_DATASETS_CACHE=$MOUNT_POINT/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=$MOUNT_POINT/hf_cache/hub
export TRANSFORMERS_CACHE=$MOUNT_POINT/hf_cache/transformers

# Torch extensions
export TORCH_EXTENSIONS_DIR=$MOUNT_POINT/torch_extensions

# Auto-activate venv
source $VENV_DIR/bin/activate
# ===================================
EOF
}

########################################
# 9. Cleanup old root-disk HF cache (safe)
########################################
rm -rf ~/.cache/huggingface || true

########################################
# 10. Validation
########################################
python - <<'EOF'
import sys, torch, os
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("HF_HOME:", os.environ.get("HF_HOME"))
EOF

echo "✅ Instance-store ML environment fully ready"
