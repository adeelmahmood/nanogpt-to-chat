#!/bin/bash
# Ablation Experiments Configuration
# Define all your experiments here (compatible with older bash versions)

# Simple function-based approach instead of associative arrays

# Helper function to get variants for an experiment
get_variants() {
    local experiment_name="$1"
    case "$experiment_name" in
        "positioning")
            echo "none:pos_emb_type=none"
            echo "abs:pos_emb_type=absolute"
            echo "rope:pos_emb_type=rope"
            ;;
        "positioning_short_seq_len")
            echo "abs:pos_emb_type=absolute,block_size=128"
            echo "rope:pos_emb_type=rope,block_size=128"
            ;;
        "attention")
            echo "mha:attn_type=mha" 
            echo "gqa:attn_type=gqa"
            echo "mqa:attn_type=mqa"
            ;;
        "gqa_kv_heads_small")
            echo "gqa:attn_type=gqa,num_kv_heads=2"
            echo "gqa:attn_type=gqa,num_kv_heads=3"
            ;;
        "gqa_kv_heads_big")
            echo "gqa:attn_type=gqa,num_kv_heads=2"
            echo "gqa:attn_type=gqa,num_kv_heads=5"
            ;;
        "normalization")
            echo "rms:use_rmsnorm=true"
            echo "layer:use_rmsnorm=false"
            ;;
        "qk_norm")
            echo "with_qk:use_qk_norm=true"
            echo "no_qk:use_qk_norm=false"
            ;;
        "logits_cap")
            echo "no_cap:logit_softcap=0.0"
            echo "softcap_15:logit_softcap=15.0"
            echo "softcap_30:logit_softcap=30.0"
            ;;
        "model_size")
            echo "small:n_layer=6,n_emb=384"
            echo "medium:n_layer=12,n_emb=768"
            echo "large:n_layer=20,n_emb=1280"
            ;;
        "stress")
            echo "all_good:pos_emb_type=rope,use_rmsnorm=true,use_qk_norm=true,attn_type=mha,use_kv_cache=true"
            echo "all_bad:pos_emb_type=absolute,use_rmsnorm=false,use_qk_norm=false,attn_type=mqa,use_kv_cache=false"
            ;;
        *)
            echo "ERROR: Unknown experiment '$experiment_name'" >&2
            echo "Available experiments: positioning, positioning_short_seq_len, attention, gqa_kv_heads_small, gqa_kv_heads_big, normalization, qk_norm, logits_cap, model_size, stress" >&2
            return 1
            ;;
    esac
}

# List all available experiments
list_experiments() {
    echo "Available experiments:"
    echo "  positioning   - RoPE vs Absolute positioning"
    echo "  positioning_short_seq_len - RoPE vs Absolute with short block size (128)"
    echo "  attention     - MHA vs GQA vs MQA attention mechanisms"
    echo "  gqa_kv_heads_small - GQA with 2 vs 3 KV heads (small model)"
    echo "  gqa_kv_heads_big   - GQA with 2 vs 5 KV heads (big model)"
    echo "  normalization - RMSNorm vs LayerNorm"
    echo "  qk_norm       - With vs Without QK normalization"
    echo "  logits_cap    - Different logit softcap values"
    echo "  model_size    - Small vs Medium vs Large model dimensions"
    echo "  stress        - All good vs All bad configurations"