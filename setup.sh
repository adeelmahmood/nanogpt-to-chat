#!/usr/bin/env bash
set -euo pipefail

### CONFIG
MOUNT_POINT="/mnt/instancestore"
VENV_DIR="$MOUNT_POINT/venv"

echo "== System update =="
sudo dnf update -y
sudo dnf install -y git nvme-cli python3.11 python3.11-devel python3.11-pip

echo "== Mount instance store (nvme1n1) =="
sudo mkfs.ext4 -F /dev/nvme1n1 || true
sudo mkdir -p $MOUNT_POINT
mountpoint -q $MOUNT_POINT || sudo mount /dev/nvme1n1 $MOUNT_POINT
sudo chown -R $USER:$USER $MOUNT_POINT

echo "== Create directories =="
mkdir -p \
  $MOUNT_POINT/{pip_cache,torch_extensions} \
  $MOUNT_POINT/hf_cache/{datasets,hub,transformers} \
  $MOUNT_POINT/wandb/{cache,config,run}

echo "== Create Python 3.11 venv =="
rm -rf "$VENV_DIR"
python3.11 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

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
EOF

# activate python env in current shell
source ~/.config/ml_env.sh
source $VENV_DIR/bin/activate

echo "== netrc setup =="
touch $MOUNT_POINT/.netrc
chmod 600 $MOUNT_POINT/.netrc
rm -f ~/.netrc

echo "== Cleanup root caches =="
rm -rf ~/.cache/{huggingface,wandb,pip} ~/.local/share/wandb || true

cd /mnt/instancestore
cd $MOUNT_POINT
if [ ! -d nanogpt-to-chat ]; then
  git clone https://github.com/adeelmahmood/nanogpt-to-chat.git
fi

# == Install repo requirements ==
cd $MOUNT_POINT/nanogpt-to-chat
pip install --upgrade pip
pip install -r requirements.txt

echo "== Validation =="
python - <<EOF
import sys, torch, os
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
print("HF_HOME:", os.environ.get("HF_HOME"))
print("WANDB_DIR:", os.environ.get("WANDB_DIR"))
print("NETRC:", os.environ.get("NETRC"))
EOF


cat > ~/enter_ml.sh <<EOF
#!/usr/bin/env bash
source ~/.config/ml_env.sh
source $VENV_DIR/bin/activate
cd $MOUNT_POINT/nanogpt-to-chat
EOF

chmod +x ~/enter_ml.sh

echo "== Setup complete =="
echo "Next: source ~/enter_ml.sh"
