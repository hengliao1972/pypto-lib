"""
Level 1: LM head with tiling (see llama7b_model.md §4.5).
hidden [B, S, H] @ lm_head_weight [V, H].T -> logits [B, S, V].
"""
from __future__ import annotations

from llama7b._scope import incore_scope
from llama7b import config
from llama7b.primitives import LlamaPrimitives

TB = config.TB
TS = config.TS
H = config.HIDDEN_SIZE
V = config.VOCAB_SIZE


def lm_head_tiled(hidden, lm_head_weight, sequence_length):
    """
    Tiled lm_head: logits = hidden @ lm_head_weight.T.
    hidden [B, max_S, H], lm_head_weight [V, H], returns logits [B, max_S, V].
    """
    B, max_S, _ = hidden.shape[0], hidden.shape[1], hidden.shape[2]
    from types import SimpleNamespace
    logits = SimpleNamespace(shape=(B, max_S, V))
    prim = LlamaPrimitives()
    for b_start in range(0, B, TB):
        for s_start in range(0, max_S, TS):
            for v_start in range(0, V, config.TILE_COLS):
                h_start = 0
                with incore_scope():
                    _apply_lm_head_block(prim, hidden, lm_head_weight, logits,
                                        b_start, s_start, h_start, v_start)
    return logits


def _apply_lm_head_block(prim, hidden, lm_head_weight, logits, b_start, s_start, h_start, v_start):
    """One block: linear_tile(hidden_block, lm_head_weight_block.T). Compiler lowers to matmul_tile with views."""
    pass
