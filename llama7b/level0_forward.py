"""
Level 0: Llama forward entry (see llama7b_model.md §3).
llama_forward(input_ids, sequence_length, weights) -> logits.
Supports batch and variable sequence_length[b].
"""
from __future__ import annotations

from llama7b import config
from llama7b.level1_embedding import embedding_tiled
from llama7b.level1_transformer import transformer_block_tiled
from llama7b.level2_rmsnorm import rmsnorm_tiled
from llama7b.level1_lm_head import lm_head_tiled

L = config.N_LAYERS


def llama_forward(input_ids, sequence_length, weights):
    """
    Llama 7B forward pass.

    Args:
        input_ids: [B, max_S] int32 token ids
        sequence_length: [B] int32, sequence_length[b] = valid length for batch b
        weights: dict with keys embedding, layer_{i}_attn_norm, layer_{i}_q/k/v/o_proj,
                 layer_{i}_mlp_norm, layer_{i}_gate/up/down_proj, final_norm, lm_head

    Returns:
        logits: [B, max_S, V]
    """
    B = input_ids.shape[0]
    max_S = input_ids.shape[1]
    # Embedding
    hidden = embedding_tiled(
        input_ids,
        weights["embedding"],
        sequence_length,
    )
    # Transformer layers
    for layer_id in range(L):
        hidden = transformer_block_tiled(
            layer_id,
            hidden,
            sequence_length,
            attn_norm_weight=weights[f"layer_{layer_id}_attn_norm"],
            attn_q_proj=weights[f"layer_{layer_id}_q_proj"],
            attn_k_proj=weights[f"layer_{layer_id}_k_proj"],
            attn_v_proj=weights[f"layer_{layer_id}_v_proj"],
            attn_o_proj=weights[f"layer_{layer_id}_o_proj"],
            mlp_norm_weight=weights[f"layer_{layer_id}_mlp_norm"],
            gate_proj=weights[f"layer_{layer_id}_gate_proj"],
            up_proj=weights[f"layer_{layer_id}_up_proj"],
            down_proj=weights[f"layer_{layer_id}_down_proj"],
        )
    # Final norm and lm_head
    hidden = rmsnorm_tiled(hidden, weights["final_norm"], hidden, sequence_length)
    logits = lm_head_tiled(hidden, weights["lm_head"], sequence_length)
    return logits
