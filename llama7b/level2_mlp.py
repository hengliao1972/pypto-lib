"""
Level 2: MLP with tiling and incore_scope (see llama7b_model.md §4.4).
Gate/Up projection -> SiLU(gate)*up -> Down projection. All within tile blocks.
"""
from __future__ import annotations

from llama7b._scope import incore_scope
from llama7b import config
from llama7b.primitives import LlamaPrimitives

TB = config.TB
TS = config.TS
TH = config.TH
M = config.MLP_INTERMEDIATE_SIZE
TR = config.TILE_ROWS
TC = config.TILE_COLS


def mlp_tiled(hidden, gate_proj, up_proj, down_proj, out, sequence_length):
    """
    Tiled MLP: gate = gate_proj(hidden), up = up_proj(hidden), hidden = down_proj(SiLU(gate) * up).
    hidden [B, S, H], gate_proj [M, H], up_proj [M, H], down_proj [H, M], out [B, S, H].
    """
    B, max_S, H = hidden.shape[0], hidden.shape[1], hidden.shape[2]
    prim = LlamaPrimitives()
    for b_start in range(0, B, TB):
        for s_start in range(0, max_S, TS):
            for h_start in range(0, H, TC):
                for m_start in range(0, M, TC):
                    with incore_scope():
                        _apply_mlp_block(prim, hidden, gate_proj, up_proj, down_proj, out,
                                        b_start, s_start, h_start, m_start)
    return out


def _apply_mlp_block(prim, hidden, gate_proj, up_proj, down_proj, out,
                     b_start, s_start, h_start, m_start):
    """
    One block: linear_tile (gate), linear_tile (up), silu_tile (gate), mul, linear_tile (down).
    Compiler lowers to primitive calls with tensor views.
    """
    pass
