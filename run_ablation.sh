#!/usr/bin/env bash
set -e

# Unified Ablation Runner - Directly calls train.py
# Usage: ./run_ablation.sh <experiment_name> [model_depth]
# Example: ./run_ablation.sh positioning d12
# Example: ./run_ablation.sh positioning d20

# Source the experiments config
source ./ablation_experiments.sh

EXPERIMENT_NAME="${1:-}"
MODEL_DEPTH="${2:-d12}"  # Default to d12 if not specified
EXPERIMENT_DATE="$(date +%m-%d)"

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "❌ Error: No experiment specified"
    echo
    echo "Usage: $0 <experiment_name> [model_depth]"
    echo "  model_depth: d12 (default), d20, d2"
    echo
    list_experiments
    exit 1
fi

echo "🚀 Starting ${EXPERIMENT_NAME} ablation study..."
echo "Date: ${EXPERIMENT_DATE}"
echo "Model: ${MODEL_DEPTH}"
echo

# Create ablation-specific results directory (include model depth)
RESULTS_DIR="ablations/results/${EXPERIMENT_DATE}/${EXPERIMENT_NAME}_${MODEL_DEPTH}"
mkdir -p "$RESULTS_DIR"

# Dataset setup - check if dataset exists locally
DATASET="fw"
if [[ "$DATASET" == "fw" ]]; then
    DATASET_DIR="download/edu_fineweb10B"
elif [[ "$DATASET" == "ts" ]]; then
    DATASET_DIR="download/tinystories"
else
    echo "ERROR: Invalid dataset: $DATASET"
    exit 1
fi

if [[ ! -d "$DATASET_DIR" ]] || [[ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "Please download the dataset first using the main run scripts"
    exit 1
fi

# Launcher detection
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

# Get variants for this experiment (compatible with older bash)
VARIANT_SPECS=()
while IFS= read -r line; do
    VARIANT_SPECS+=("$line")
done < <(get_variants "$EXPERIMENT_NAME")
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Set training parameters based on model size
if [[ "$MODEL_DEPTH" == "d20" ]]; then
    BATCH_SIZE=8
    DEVICE_BATCH_SIZE=16384
    MAX_STEPS=1000
else  # d12
    BATCH_SIZE=8
    DEVICE_BATCH_SIZE=32768
    MAX_STEPS=2000
fi

TOTAL_BATCH_SIZE=$((DEVICE_BATCH_SIZE * NPROC_PER_NODE))
echo "Using ${MODEL_DEPTH} training regime: batch_size=$BATCH_SIZE, total_batch_size=$TOTAL_BATCH_SIZE, max_steps=$MAX_STEPS"

echo "Running ${#VARIANT_SPECS[@]} variants:"
for variant_spec in "${VARIANT_SPECS[@]}"; do
    variant_name="${variant_spec%%:*}"
    echo "  - ${variant_name}"
done
echo

# Helper function to parse and add variant parameters
add_variant_params() {
    local params="$1"
    
    # Parse comma-separated key=value pairs
    if [[ -n "$params" ]]; then
        IFS=',' read -ra PARAM_PAIRS <<< "$params"
        for pair in "${PARAM_PAIRS[@]}"; do
            if [[ "$pair" == *"="* ]]; then
                key="${pair%%=*}"
                value="${pair#*=}"
                
                # Handle max_steps specially - don't add to TRAIN_CMD, just set VARIANT_MAX_STEPS
                if [[ "$key" == "max_steps" ]]; then
                    VARIANT_MAX_STEPS="$value"
                else
                    TRAIN_CMD+=(--"$key" "$value")
                fi
            fi
        done
    fi
}

# Run each variant
for variant_spec in "${VARIANT_SPECS[@]}"; do
    variant_name="${variant_spec%%:*}"
    variant_params="${variant_spec#*:}"
    
    echo "============================================"
    echo "🔄 Running variant: ${variant_name}"
    if [[ -n "$variant_params" ]]; then
        echo "   Parameters: ${variant_params}"
    else
        echo "   Parameters: (using defaults)"
    fi
    echo "============================================"
    
    # Set up run-specific directory for this variant
    VARIANT_RUN_ID="${EXPERIMENT_DATE}/${EXPERIMENT_NAME}_${MODEL_DEPTH}_${variant_name}"
    VARIANT_RUN_DIR="runs/${VARIANT_RUN_ID}"
    mkdir -p "${VARIANT_RUN_DIR}"
    
    # Initialize variant-specific max_steps (default to global MAX_STEPS)
    VARIANT_MAX_STEPS="$MAX_STEPS"
    
    # Build training command directly
    TRAIN_CMD=(
        "${PYTHON_CMD[@]}" train.py
        --dataset "$DATASET"
        --dataset_dir "$DATASET_DIR"
        --model_depth "$MODEL_DEPTH"
        --batch_size "$BATCH_SIZE"
        --total_batch_size "$TOTAL_BATCH_SIZE"
        --eval_every -1
        --save_every -1
        --ckpt_out "${VARIANT_RUN_DIR}/checkpoints"
        --log_dir "$VARIANT_RUN_DIR"
        --log_file "train.jsonl"
    )
    
    # Add variant-specific parameters (this may update VARIANT_MAX_STEPS)
    add_variant_params "$variant_params"
    
    # Add the final max_steps to command (after potential override)
    TRAIN_CMD+=(--max_steps "$VARIANT_MAX_STEPS")
    
    # Display and run the command
    echo "Starting training for ${variant_name}..."
    echo "  Run ID: ${VARIANT_RUN_ID}"
    echo "  Max steps: ${VARIANT_MAX_STEPS}$([ "$VARIANT_MAX_STEPS" != "$MAX_STEPS" ] && echo " (overridden from $MAX_STEPS)" || echo "")"
    echo "  Logs will be saved to: ${VARIANT_RUN_DIR}/train.jsonl"
    echo "  Command: ${TRAIN_CMD[*]}"
    echo
    
    "${TRAIN_CMD[@]}" || {
        echo "❌ Failed to run ${variant_name}"
        continue
    }
    
    echo "✅ Completed ${variant_name}"
    echo
done

echo "============================================"
echo "🎯 All variants completed! Generating plots..."
echo "============================================"

# Generate ablation plots - look for logs in run directories
TITLE="$(echo "${EXPERIMENT_NAME}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}') Ablation Study (${MODEL_DEPTH} - ${EXPERIMENT_DATE})"
python ablations.py --filter-pattern "${EXPERIMENT_DATE}/${EXPERIMENT_NAME}_${MODEL_DEPTH}_*" \
                   --output-dir "$RESULTS_DIR" \
                   --title "$TITLE" \
                   --runs-dir "runs"

echo "✅ Ablation study complete!"
echo "📊 Results saved to: ${RESULTS_DIR}"
echo "📈 Plots generated in: ${RESULTS_DIR}/ablation_plots.png"

echo
CAPITALIZED_NAME="$(echo "${EXPERIMENT_NAME}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
echo "🎉 ${CAPITALIZED_NAME} ablation study finished successfully!"