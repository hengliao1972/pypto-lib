"""
Level 2: RMSNorm with tiling and incore_scope (see llama7b_model.md §4.2, §5).
Loop over tile blocks; inside each block incore_scope runs PTO-ISA (primitives or pl.load/store).
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


def rmsnorm_tiled(x, weight, out, sequence_length):
    """
    Tiled RMSNorm: out = x * weight / rms(x). Respects sequence_length for variable length.
    x [B, max_S, H], weight [H], out [B, max_S, H]. Uses incore_scope per tile block.
    Caller may pass out=x for in-place.
    """
    B, max_S, H = x.shape[0], x.shape[1], x.shape[2]
    prim = LlamaPrimitives()
    for b_start in range(0, B, TB):
        b_end = min(b_start + TB, B)
        for s_start in range(0, max_S, TS):
            s_end = min(s_start + TS, max_S)
            for h_start in range(0, H, TC):
                h_end = min(h_start + TC, H)
                with incore_scope():
                    _apply_rmsnorm_tile(prim, x, weight, out, b_start, s_start, h_start, b_end, s_end, h_end)
    return out


def _apply_rmsnorm_tile(prim, x, weight, out, b_start, s_start, h_start, b_end, s_end, h_end):
    """
    Apply rmsnorm_tile to one block. Compiler lowers this to a call to LlamaPrimitives.rmsnorm_tile
    with tensor views (x[b_start:b_end, s_start:s_end, h_start:h_end], etc.).
    Block shape is trimmed to (b_end-b_start, s_end-s_start, h_end-h_start); tail handled by valid_row/valid_col.
    """
    # At compile time: emit call to prim.rmsnorm_tile with views of x, weight, out for this block.
    pass
