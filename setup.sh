#!/usr/bin/env bash
set -euo pipefail

### CONFIG
MOUNT_POINT="/mnt/instancestore"
VENV_DIR="$MOUNT_POINT/venv"
REPO_NAME="nanogpt-to-chat"
REPO_URL="https://github.com/adeelmahmood/nanogpt-to-chat.git"

echo "== System update =="
sudo dnf update -y
sudo dnf install -y git nvme-cli python3.11 python3.11-devel python3.11-pip mdadm

########################################
# Instance-store detection & mounting
########################################

echo "== Detecting instance-store NVMe devices =="

mapfile -t INSTANCE_DEVS < <(
  lsblk -ndo NAME,MODEL | awk '$2 ~ /Instance Storage/ {print "/dev/"$1}'
)

if [ ${#INSTANCE_DEVS[@]} -eq 0 ]; then
  echo "ERROR: No instance-store NVMe devices found."
  lsblk -o NAME,SIZE,MODEL,MOUNTPOINT
  exit 1
fi

echo "Found instance-store devices:"
printf "  %s\n" "${INSTANCE_DEVS[@]}"

sudo mkdir -p "$MOUNT_POINT"

if mountpoint -q "$MOUNT_POINT"; then
  echo "Instance-store already mounted at $MOUNT_POINT"

else
  if [ ${#INSTANCE_DEVS[@]} -eq 1 ]; then
    DEV="${INSTANCE_DEVS[0]}"
    echo "Using single instance-store device $DEV"

    if ! blkid "$DEV" >/dev/null 2>&1; then
      sudo mkfs.ext4 -F "$DEV"
    fi

    sudo mount "$DEV" "$MOUNT_POINT"

  else
    echo "Using RAID0 over instance-store devices"

    if [ ! -e /dev/md0 ]; then
      sudo mdadm --create /dev/md0 \
        --level=0 \
        --raid-devices=${#INSTANCE_DEVS[@]} \
        "${INSTANCE_DEVS[@]}"
    fi

    if ! blkid /dev/md0 >/dev/null 2>&1; then
      sudo mkfs.ext4 -F /dev/md0
    fi

    sudo mount /dev/md0 "$MOUNT_POINT"
  fi
fi

sudo chown -R "$USER:$USER" "$MOUNT_POINT"

echo "== Disk sanity check =="
lsblk -o NAME,SIZE,MODEL,MOUNTPOINT
df -h "$MOUNT_POINT"

########################################
# Directory layout
########################################

echo "== Create directories =="
mkdir -p \
  "$MOUNT_POINT"/{pip_cache,torch_extensions,download} \
  "$MOUNT_POINT"/hf_cache/{datasets,hub,transformers} \
  "$MOUNT_POINT"/wandb/{cache,config,run}

########################################
# Python environment
########################################

echo "== Create Python 3.11 venv =="
rm -rf "$VENV_DIR"
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

########################################
# Environment variables
########################################

echo "== Environment variables =="
mkdir -p ~/.config
cat > ~/.config/ml_env.sh <<EOF
export PIP_CACHE_DIR=$MOUNT_POINT/pip_cache

export HF_HOME=$MOUNT_POINT/hf_cache
export HF_DATASETS_CACHE=$MOUNT_POINT/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=$MOUNT_POINT/hf_cache/hub
export TRANSFORMERS_CACHE=$MOUNT_POINT/hf_cache/transformers

export TORCH_EXTENSIONS_DIR=$MOUNT_POINT/torch_extensions

export WANDB_DIR=$MOUNT_POINT/wandb/run
export WANDB_CACHE_DIR=$MOUNT_POINT/wandb/cache
export WANDB_CONFIG_DIR=$MOUNT_POINT/wandb/config

export NETRC=$MOUNT_POINT/.netrc
export SEND_TO_WANDB=1
EOF

source ~/.config/ml_env.sh

########################################
# netrc & cleanup
########################################

echo "== netrc setup =="
touch "$MOUNT_POINT/.netrc"
chmod 600 "$MOUNT_POINT/.netrc"
rm -f ~/.netrc || true

echo "== Cleanup root caches =="
rm -rf ~/.cache/{huggingface,wandb,pip} ~/.local/share/wandb || true

########################################
# Repo setup
########################################

cd "$MOUNT_POINT"

if [ ! -d "$REPO_NAME" ]; then
  git clone "$REPO_URL"
fi

cd "$MOUNT_POINT/$REPO_NAME"

# Force downloads to instance-store (single safety symlink)
rm -rf download
ln -s "$MOUNT_POINT/download" download

########################################
# Python deps
########################################

pip install --upgrade pip
pip install -r requirements.txt

########################################
# Entry helper
########################################

cat > ~/enter_ml.sh <<EOF
#!/usr/bin/env bash
set -e
source ~/.config/ml_env.sh
source $VENV_DIR/bin/activate
cd $MOUNT_POINT/$REPO_NAME
wandb login
EOF

chmod +x ~/enter_ml.sh

########################################
# Validation
########################################

echo "== Validation =="
python - <<EOF
import sys, os
print("Python:", sys.version)
print("HF_HOME:", os.environ.get("HF_HOME"))
print("WANDB_DIR:", os.environ.get("WANDB_DIR"))
print("NETRC:", os.environ.get("NETRC"))
EOF

echo "== Setup complete =="
echo "Next: source ~/enter_ml.sh"
