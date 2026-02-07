#!/usr/bin/env bash
set -e

# Unified Ablation Runner
# Usage: ./run_ablation.sh <experiment_name>
# Example: ./run_ablation.sh positioning

# Source the experiments config
source ./ablation_experiments.sh

EXPERIMENT_NAME="${1:-}"
EXPERIMENT_DATE="$(date +%m-%d)"

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo "❌ Error: No experiment specified"
    echo
    echo "Usage: $0 <experiment_name>"
    echo
    list_experiments
    exit 1
fi

echo "🚀 Starting ${EXPERIMENT_NAME} ablation study..."
echo "Date: ${EXPERIMENT_DATE}"
echo

# Create ablation-specific results directory
RESULTS_DIR="ablations/results/${EXPERIMENT_DATE}_${EXPERIMENT_NAME}"
mkdir -p "$RESULTS_DIR"

# Get variants for this experiment (compatible with older bash)
VARIANT_SPECS=()
while IFS= read -r line; do
    VARIANT_SPECS+=("$line")
done < <(get_variants "$EXPERIMENT_NAME")
if [[ $? -ne 0 ]]; then
    exit 1
fi

echo "Running ${#VARIANT_SPECS[@]} variants:"
for variant_spec in "${VARIANT_SPECS[@]}"; do
    variant_name="${variant_spec%%:*}"
    echo "  - ${variant_name}"
done
echo

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
    
    # Create temporary parameter file
    TEMP_PARAMS="ablations/params/temp_${EXPERIMENT_NAME}_${variant_name}.env"
    cat > "$TEMP_PARAMS" << EOF
# Generated ablation parameters for ${EXPERIMENT_NAME} - ${variant_name}
DATASET=ts
MODEL_DEPTH=d12
BATCH_SIZE=8
TOTAL_BATCH_SIZE=32768
TRAIN_STEPS=3000
EVAL_EVERY=-1
SAVE_EVERY=-1

# Ablation-specific parameters
${variant_params}

# Base parameters (RUN_ID and logging will be added dynamically)
EOF
    
    # Set up run-specific directory for this variant
    VARIANT_RUN_ID="${EXPERIMENT_DATE}_${EXPERIMENT_NAME}_${variant_name}"
    VARIANT_RUN_DIR="runs/${VARIANT_RUN_ID}"
    mkdir -p "${VARIANT_RUN_DIR}"
    
    # Update the temp params with proper RUN_ID and logging
    echo "RUN_ID=\"${VARIANT_RUN_ID}\"" >> "$TEMP_PARAMS"
    echo "log_dir=\"${VARIANT_RUN_DIR}\"" >> "$TEMP_PARAMS"
    echo "log_file=\"train.jsonl\"" >> "$TEMP_PARAMS"
    
    # Run the experiment
    echo "Starting training for ${variant_name}..."
    echo "  Run ID: ${VARIANT_RUN_ID}"
    echo "  Logs will be saved to: ${VARIANT_RUN_DIR}/train.jsonl"
    ./run_ablation_simple.sh "$TEMP_PARAMS" || {
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
TITLE="$(echo "${EXPERIMENT_NAME}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}') Ablation Study (${EXPERIMENT_DATE})"
python ablations.py --filter-pattern "${EXPERIMENT_DATE}_${EXPERIMENT_NAME}_*" \
                   --output-dir "$RESULTS_DIR" \
                   --title "$TITLE" \
                   --runs-dir "runs"

echo "✅ Ablation study complete!"
echo "📊 Results saved to: ${RESULTS_DIR}"
echo "📈 Plots generated in: ${RESULTS_DIR}/ablation_plots.png"

# Clean up temp files
rm -f ablations/params/temp_${EXPERIMENT_NAME}_*.env

echo
CAPITALIZED_NAME="$(echo "${EXPERIMENT_NAME}" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')"
echo "🎉 ${CAPITALIZED_NAME} ablation study finished successfully!"