"""
Level 1: Embedding with tiling (see llama7b_model.md §4.1).
input_ids [B, max_S] -> hidden [B, max_S, H] via embedding_weight [V, H].
Tiling (TB, TS); incore_scope per block for lookup + store.
"""
from __future__ import annotations

from llama7b._scope import incore_scope
from llama7b import config

TB = config.TB
TS = config.TS
V = config.VOCAB_SIZE
H = config.HIDDEN_SIZE


def embedding_tiled(input_ids, embedding_weight, sequence_length):
    """
    Tiled embedding lookup. For each (b_start, s_start) block, incore_scope loads
    input_ids[b,s] indices, gathers rows from embedding_weight, stores to hidden.
    input_ids [B, max_S], embedding_weight [V, H], returns hidden [B, max_S, H].
    """
    B, max_S = input_ids.shape[0], input_ids.shape[1]
    # Output: allocated by runtime at call time; placeholder here for structure
    from types import SimpleNamespace
    hidden = SimpleNamespace(shape=(B, max_S, H))
    for b_start in range(0, B, TB):
        b_end = min(b_start + TB, B)
        for s_start in range(0, max_S, TS):
            s_end = min(s_start + TS, max_S)
            with incore_scope():
                _apply_embedding_block(input_ids, embedding_weight, hidden,
                                       b_start, s_start, b_end, s_end, sequence_length)
    return hidden


def _apply_embedding_block(input_ids, embedding_weight, hidden,
                           b_start, s_start, b_end, s_end, sequence_length):
    """
    One block embedding lookup. Compiler lowers to PTO-ISA; this body is the contract.

    Views (this block):
      - ids_tile   = input_ids[b_start:b_end, s_start:s_end]   # [nB, nS], int32
      - out_tile   = hidden[b_start:b_end, s_start:s_end, :]   # [nB, nS, H]
    Semantics:
      1. Load ids_tile (or scalar indices) from input_ids.
      2. Gather: for each (b, s) in block, row = embedding_weight[ids_tile[b,s], :]; write to out_tile[b, s, :].
      3. Store out_tile to hidden.
    PTO-ISA: pto.tload (ids), gather from embedding_weight by indices, pto.tstore (hidden).
    Tail: positions with s_start + s >= sequence_length[b] are padding; compiler/runtime masks or skips.
    """
    pass
