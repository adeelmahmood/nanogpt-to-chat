#!/usr/bin/env bash
set -e

# Multi-Ablation Runner - Simple orchestrator
# Usage: ./run_ablations.sh run=positioning:d12 run=positioning:d20 run=gqa_kv_heads_small:d20
# Usage: ./run_ablations.sh run=positioning run=attention (defaults to d12)

# Check if any arguments provided
if [[ $# -eq 0 ]]; then
    echo "❌ Error: No runs specified"
    echo ""
    echo "Usage: $0 run=<experiment>[:<model>] ..."
    echo ""
    echo "Examples:"
    echo "  $0 run=positioning:d12 run=positioning:d20"
    echo "  $0 run=positioning run=attention          # defaults to d12"
    echo "  $0 run=gqa_kv_heads_small:d20 run=stress:d12"
    echo ""
    source ./ablation_experiments.sh
    list_experiments
    exit 1
fi

# Parse run specifications
RUNS=()
for arg in "$@"; do
    if [[ "$arg" == run=* ]]; then
        run_spec="${arg#run=}"
        
        # Split experiment:model (default to d12 if no model specified)
        if [[ "$run_spec" == *":"* ]]; then
            experiment="${run_spec%%:*}"
            model="${run_spec#*:}"
        else
            experiment="$run_spec"
            model="d12"  # default model
        fi
        
        # Validate model depth
        if [[ ! "$model" =~ ^d(12|20|2)$ ]]; then
            echo "❌ Invalid model depth: $model (must be d12, d20, or d2)"
            exit 1
        fi
        
        RUNS+=("${experiment}:${model}")
    else
        echo "❌ Invalid argument: $arg (must start with 'run=')"
        exit 1
    fi
done

TOTAL_RUNS=${#RUNS[@]}

echo "🚀 Starting multi-ablation run"
echo "📊 Total runs: $TOTAL_RUNS"
for run in "${RUNS[@]}"; do
    experiment="${run%%:*}"
    model="${run#*:}"
    echo "  - $experiment ($model)"
done
echo

# Track progress
COMPLETED=0
FAILED=0

# Run each experiment
for i in "${!RUNS[@]}"; do
    run_spec="${RUNS[$i]}"
    experiment="${run_spec%%:*}"
    model="${run_spec#*:}"
    
    current_run=$((i + 1))
    
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "🎯 PROGRESS [$current_run/$TOTAL_RUNS] - $experiment ($model)"
    echo "✅ Completed: $COMPLETED | ❌ Failed: $FAILED | ⏳ Remaining: $((TOTAL_RUNS - current_run + 1))"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo
    
    # Run the ablation (each handles its own logging/folders)
    if ./run_ablation.sh "$experiment" "$model"; then
        ((COMPLETED++))
        echo
        echo "✅ SUCCESS: $experiment ($model)"
    else
        ((FAILED++))
        echo
        echo "❌ FAILED: $experiment ($model)"
    fi
    echo
done

# Final summary
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🎉 MULTI-ABLATION COMPLETE!"
echo "📊 Total: $TOTAL_RUNS | ✅ Completed: $COMPLETED | ❌ Failed: $FAILED"
echo "════════════════════════════════════════════════════════════════════════════════"

if [[ $FAILED -eq 0 ]]; then
    echo "🎉 All ablations completed successfully!"
    exit 0
else
    echo "⚠️  $FAILED ablations failed."
    exit 1
fi