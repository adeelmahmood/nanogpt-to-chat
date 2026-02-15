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
        "positioning_mqa")
            echo "none:pos_emb_type=none,attn_type=mqa"
            echo "abs:pos_emb_type=absolute,attn_type=mqa"
            echo "rope:pos_emb_type=rope,attn_type=mqa"
            ;;
        "positioning_short_seq_len")
            echo "abs:pos_emb_type=absolute,block_size=128"
            echo "rope:pos_emb_type=rope,block_size=128"
            ;;
        "attention")
            echo "mha:attn_type=mha" 
            echo "gqa:attn_type=gqa,n_kv_head=2"
            echo "mqa:attn_type=mqa"
            ;;
        "gqa_kv_heads_small")
            echo "gqa_2:attn_type=gqa,n_kv_head=2"
            echo "gqa_3:attn_type=gqa,n_kv_head=3"
            ;;
        "gqa_kv_heads_big")
            echo "gqa_2:attn_type=gqa,n_kv_head=2"
            echo "gqa_5:attn_type=gqa,n_kv_head=5"
            ;;
        "stability")
            echo "stable:use_rmsnorm=true,use_qk_norm=true,logit_softcap=15.0"
            echo "unstable:use_rmsnorm=false,use_qk_norm=false,logit_softcap=0.0"
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
        "lr_alpha_sweep")
            echo "alpha_0.40:lr_alpha=0.40"
            echo "alpha_0.45:lr_alpha=0.45"
            echo "alpha_1.00:lr_alpha=1.00"
            ;;
        "resid_lambda_alpha_sweep")
            echo "resid_lambda_alpha_0.5:resid_lambda_alpha=0.5,max_tokens=10000"
            echo "resid_lambda_alpha_1.0:resid_lambda_alpha=1.0,max_tokens=10000"
            echo "resid_lambda_alpha_1.5:resid_lambda_alpha=1.5,max_tokens=10000"
            ;;
        "matrix_lr_alpha_sweep")
            echo "matrix_alpha_0.5:matrix_lr_alpha=0.5,max_tokens=10000"
            echo "matrix_alpha_1.0:matrix_lr_alpha=1.0,max_tokens=10000"
            echo "matrix_alpha_1.5:matrix_lr_alpha=1.5,max_tokens=10000"
            ;;
        "model_size")
            echo "d6:block_size=512,n_layer=6,n_head=3,n_kv_head=3,n_emb=384,max_steps=2000"
            echo "d12:block_size=1024,n_layer=12,n_head=6,n_kv_head=6,n_emb=768,max_steps=4000"
            echo "d20:block_size=2048,n_layer=20,n_head=10,n_kv_head=10,n_emb=1280,max_steps=6000"
            ;;
        "embed_alpha_scale_check")
            echo "embed_alpha_0.25:lr_alpha=0.55,matrix_lr_alpha=0.16,embed_lr_alpha=0.25,max_tokens=150000000"
            echo "embed_alpha_1.0:lr_alpha=0.55,matrix_lr_alpha=0.16,embed_lr_alpha=1.0,max_tokens=150000000"
            echo "embed_alpha_2.0:lr_alpha=0.55,matrix_lr_alpha=0.16,embed_lr_alpha=2.0,max_tokens=150000000"
            ;;
        "resid_lambda_scale_check")
            echo "resid_lambda_alpha_0.5:lr_alpha=0.55,matrix_lr_alpha=0.16,resid_lambda_alpha=0.5,max_tokens=150000000"
            echo "resid_lambda_alpha_1.0:lr_alpha=0.55,matrix_lr_alpha=0.16,resid_lambda_alpha=1.0,max_tokens=150000000"
            echo "resid_lambda_alpha_1.5:lr_alpha=0.55,matrix_lr_alpha=0.16,resid_lambda_alpha=1.5,max_tokens=150000000"
            ;;
        "long_run_global_matrix_confirm")
            echo "best_config:lr_alpha=0.55,matrix_lr_alpha=0.16,max_tokens=1000000000"
            echo "runnerup_config:lr_alpha=0.45,matrix_lr_alpha=0.16,max_tokens=1000000000"
            ;;
        "endtoend")
            echo "improved:pos_emb_type=rope,use_rmsnorm=true,use_qk_norm=true,attn_type=mha,use_kv_cache=true"
            echo "current:pos_emb_type=rope,use_rmsnorm=true,use_qk_norm=true,attn_type=mha,use_kv_cache=true"
            echo "baseline:pos_emb_type=absolute,use_rmsnorm=false,use_qk_norm=false,attn_type=mqa,use_kv_cache=false"
            ;;
        *)
            echo "ERROR: Unknown experiment '$experiment_name'" >&2
            return 1
            ;;
    esac
}

# List all available experiments
list_experiments() {
    echo "Available experiments:"
    echo "  positioning   - RoPE vs Absolute positioning"
    echo "  positioning_mqa - RoPE vs Absolute positioning with MQA attention"
    echo "  positioning_short_seq_len - RoPE vs Absolute with short block size (128)"
    echo "  attention     - MHA vs GQA vs MQA attention mechanisms"
    echo "  gqa_kv_heads_small - GQA with 2 vs 3 KV heads (small model)"
    echo "  gqa_kv_heads_big   - GQA with 2 vs 5 KV heads (big model)"
    echo "  normalization - RMSNorm vs LayerNorm"
    echo "  qk_norm       - With vs Without QK normalization"
    echo "  logits_cap    - Different logit softcap values"
    echo "  lr_alpha_sweep - Different learning rate alpha values"
    echo "  matrix_lr_alpha_sweep - Different matrix learning rate alpha values"
    echo "  resid_lambda_alpha_sweep - Different residual lambda alpha values"
    echo "  model_size    - d6 vs d12 vs d20 model dimensions"
    echo "  stress        - All good vs All bad configurations"
}