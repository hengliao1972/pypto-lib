"""
Level 2: Attention with tiling and incore_scope (see llama7b_model.md §4.3, §5.1).
Q,K,V projections; Q@K^T; softmax; attn@V; O projection. Per tile block.
"""
from __future__ import annotations

from llama7b._scope import incore_scope
from llama7b import config
from llama7b.primitives import LlamaPrimitives

TB = config.TB
TS = config.TS
TH = config.TH
TR = config.TILE_ROWS
TC = config.TILE_COLS


def attention_tiled(hidden, q_proj, k_proj, v_proj, o_proj, out, sequence_length):
    """
    Tiled attention: Q=hidden@q_proj, K=hidden@k_proj, V=hidden@v_proj;
    scores=Q@K^T; attn=softmax(scores); out=attn@V@o_proj.
    hidden [B, S, H], out [B, S, H]. Uses incore_scope for each tile block of Q,K,V and matmul/softmax.
    """
    B, max_S, H = hidden.shape[0], hidden.shape[1], hidden.shape[2]
    prim = LlamaPrimitives()
    for b_start in range(0, B, TB):
        for s_start in range(0, max_S, TS):
            with incore_scope():
                _apply_attention_block(prim, hidden, q_proj, k_proj, v_proj, o_proj, out,
                                      b_start, s_start, B, max_S, H)
    return out


def _apply_attention_block(prim, hidden, q_proj, k_proj, v_proj, o_proj, out,
                           b_start, s_start, B, max_S, H):
    """
    One block: compute Q,K,V tiles; scores=Q@K^T; softmax_tile(scores); attn@V; O_proj.
    Compiler lowers to matmul_tile, softmax_tile, matmul_tile, linear_tile with views.
    """
    pass
