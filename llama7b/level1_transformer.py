"""
Level 1: Transformer block and add_tiled (see llama7b_model.md §4.2).
transformer_block_tiled = rmsnorm -> attention -> add -> rmsnorm -> mlp -> add.
"""
from __future__ import annotations

from llama7b import config
from llama7b.level2_rmsnorm import rmsnorm_tiled
from llama7b.level2_attention import attention_tiled
from llama7b.level2_mlp import mlp_tiled

L = config.N_LAYERS


def transformer_block_tiled(layer_id, hidden, sequence_length,
                            attn_norm_weight, attn_q_proj, attn_k_proj, attn_v_proj, attn_o_proj,
                            mlp_norm_weight, gate_proj, up_proj, down_proj):
    """
    One transformer layer: residual + attention(rmsnorm(hidden)) + residual + mlp(rmsnorm(hidden)).
    hidden [B, S, H]; weights per layer. Returns new hidden [B, S, H].
    """
    residual = hidden
    # Attention path: rmsnorm -> attention -> add residual
    hidden = rmsnorm_tiled(hidden, attn_norm_weight, hidden, sequence_length)  # in-place
    hidden = attention_tiled(hidden, attn_q_proj, attn_k_proj, attn_v_proj, attn_o_proj, hidden, sequence_length)
    hidden = add_tiled(hidden, residual)
    residual = hidden
    # MLP path: rmsnorm -> mlp -> add residual
    hidden = rmsnorm_tiled(hidden, mlp_norm_weight, hidden, sequence_length)
    hidden = mlp_tiled(hidden, gate_proj, up_proj, down_proj, hidden, sequence_length)
    hidden = add_tiled(hidden, residual)
    return hidden


def add_tiled(a, b):
    """
    Element-wise add a + b over full tensor. Tiled with incore_scope; each block uses prim.add_tile.
    a, b same shape [B, S, H]; returns a + b (same shape).
    """
    from llama7b._scope import incore_scope
    from llama7b.primitives import LlamaPrimitives
    from types import SimpleNamespace
    B, S, H = a.shape[0], a.shape[1], a.shape[2]
    out = SimpleNamespace(shape=(B, S, H))
    prim = LlamaPrimitives()
    for b_start in range(0, B, config.TB):
        for s_start in range(0, S, config.TS):
            for h_start in range(0, H, config.TILE_COLS):
                with incore_scope():
                    _apply_add_tile(prim, a, b, out, b_start, s_start, h_start)
    return out


def _apply_add_tile(prim, a, b, out, b_start, s_start, h_start):
    """One block add. Compiler lowers to prim.add_tile with views."""
    pass
