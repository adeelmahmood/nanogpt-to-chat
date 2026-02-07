#!/bin/bash
# Ablation Experiments Configuration
# Define all your experiments here (compatible with older bash versions)

# Simple function-based approach instead of associative arrays

# Helper function to get variants for an experiment
get_variants() {
    local experiment_name="$1"
    case "$experiment_name" in
        "positioning")
            echo "rope:use_rope=true"
            echo "abs:use_rope=false"
            ;;
        "attention")
            echo "gqa:use_gqa=true" 
            echo "mqa:use_gqa=false"
            ;;
        "normalization")
            echo "rms:use_rmsnorm=true"
            echo "layer:use_rmsnorm=false"
            ;;
        "qk_norm")
            echo "qk_norm:use_qk_norm=true"
            echo "no_qk_norm:use_qk_norm=false"
            ;;
        "kv_cache")
            echo "kv_cache:use_kv_cache=true"
            echo "no_kv_cache:use_kv_cache=false"
            ;;
        *)
            echo "ERROR: Unknown experiment '$experiment_name'" >&2
            echo "Available experiments: positioning, attention, normalization, qk_norm, kv_cache" >&2
            return 1
            ;;
    esac
}

# List all available experiments
list_experiments() {
    echo "Available experiments:"
    echo "  positioning   - RoPE vs Absolute positioning"
    echo "  attention     - GQA vs MQA attention mechanisms"
    echo "  normalization - RMSNorm vs LayerNorm"
    echo "  qk_norm       - With vs Without QK normalization"
    echo "  kv_cache      - With vs Without KV caching"
}