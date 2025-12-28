#!/usr/bin/env bash
set -e

echo "Starting all ablation experiments..."
echo "==================================="

EXPS=(
  baseline
  rope
  rope_rms
  rope_rms_gqa
)

for EXP in "${EXPS[@]}"; do
  echo ""
  echo "-----------------------------------"
  echo "Running experiment: $EXP"
  echo "-----------------------------------"

  python model_fw.py --exp "$EXP"

  echo "Finished experiment: $EXP"
done

echo ""
echo "All experiments completed successfully."
