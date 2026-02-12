"""
PTO-ISA level primitives (pypto.language → PTO-ISA).

Uses pypto.language (pl): pl.load / pl.store → pto.tload / pto.tstore;
block ops (pl.add, pl.matmul, pl.row_max, ...) → pto.taddc, pto.tmul, etc.
See llama7b_model.md §5 and pypto docs 12-pto_codegen.md.

Tile shapes from config (TILE_ROWS x TILE_COLS). Used inside incore_scope by level2.
Requires pypto installed (e.g. pip install -e ../pypto).
"""
from __future__ import annotations

from llama7b._pl import pl
from llama7b import config

TR = config.TILE_ROWS
TC = config.TILE_COLS

# Build the program only when pl is available (pypto installed)
if pl is not None:

    @pl.program
    class LlamaPrimitives:
        """Bottom-layer kernels: PTO-ISA (load/store + block ops) only."""

        @pl.function
        def add_tile(
            self,
            a: pl.Tensor[[TR, TC], pl.FP32],
            b: pl.Tensor[[TR, TC], pl.FP32],
            out: pl.Tensor[[TR, TC], pl.FP32],
        ) -> pl.Tensor[[TR, TC], pl.FP32]:
            """Element-wise add: out = a + b (one tile)."""
            ta = pl.load(a, [0, 0], [TR, TC])
            tb = pl.load(b, [0, 0], [TR, TC])
            tc = pl.add(ta, tb)
            return pl.store(tc, [0, 0], [TR, TC], out)

        @pl.function
        def linear_tile(
            self,
            hidden: pl.Tensor[[TR, TC], pl.FP32],
            weight: pl.Tensor[[TC, TC], pl.FP32],
            out: pl.Tensor[[TR, TC], pl.FP32],
        ) -> pl.Tensor[[TR, TC], pl.FP32]:
            """One tile matmul: out = hidden @ weight."""
            th = pl.load(hidden, [0, 0], [TR, TC])
            tw = pl.load(weight, [0, 0], [TC, TC])
            to = pl.matmul(th, tw)
            return pl.store(to, [0, 0], [TR, TC], out)

        @pl.function
        def rmsnorm_tile(
            self,
            x: pl.Tensor[[TR, TC], pl.FP32],
            weight: pl.Tensor[[1, TC], pl.FP32],
            out: pl.Tensor[[TR, TC], pl.FP32],
        ) -> pl.Tensor[[TR, TC], pl.FP32]:
            """RMSNorm per tile: out = (x / rms(x)) * weight."""
            tx = pl.load(x, [0, 0], [TR, TC])
            tmp = pl.block.create_tile([TR, 1], pl.FP32, 1)
            tx_sq = pl.mul(tx, tx)
            row_var = pl.row_sum(tx_sq, tmp)
            rms = pl.sqrt(row_var)
            tx_norm = pl.row_expand_div(tx, rms)
            w_tile = pl.load(weight, [0, 0], [1, TC])
            out_tile = pl.col_expand_mul(tx_norm, w_tile)
            return pl.store(out_tile, [0, 0], [TR, TC], out)

        @pl.function
        def softmax_tile(
            self,
            scores: pl.Tensor[[TR, TC], pl.FP32],
            out: pl.Tensor[[TR, TC], pl.FP32],
        ) -> pl.Tensor[[TR, TC], pl.FP32]:
            """Row-wise softmax on one tile."""
            ts = pl.load(scores, [0, 0], [TR, TC])
            tmp = pl.block.create_tile([TR, 1], pl.FP32, 1)
            tmax = pl.row_max(ts, tmp)
            ts = pl.row_expand_sub(ts, tmax)
            texp = pl.exp(ts)
            tsum = pl.row_sum(texp, tmp)
            tsoft = pl.row_expand_div(texp, tsum)
            return pl.store(tsoft, [0, 0], [TR, TC], out)

        @pl.function
        def matmul_tile(
            self,
            lhs: pl.Tensor[[TR, TC], pl.FP32],
            rhs: pl.Tensor[[TC, TC], pl.FP32],
            out: pl.Tensor[[TR, TC], pl.FP32],
        ) -> pl.Tensor[[TR, TC], pl.FP32]:
            """Tile matmul: out = lhs @ rhs."""
            tl = pl.load(lhs, [0, 0], [TR, TC])
            tr = pl.load(rhs, [0, 0], [TC, TC])
            to = pl.matmul(tl, tr)
            return pl.store(to, [0, 0], [TR, TC], out)

        @pl.function
        def silu_tile(
            self,
            x: pl.Tensor[[TR, TC], pl.FP32],
            out: pl.Tensor[[TR, TC], pl.FP32],
        ) -> pl.Tensor[[TR, TC], pl.FP32]:
            """SiLU(x) = x * sigmoid(x) on one tile."""
            tx = pl.load(x, [0, 0], [TR, TC])
            exp_neg = pl.exp(pl.neg(tx))
            sigmoid = pl.recip(pl.add(exp_neg, 1.0))
            out_tile = pl.mul(tx, sigmoid)
            return pl.store(out_tile, [0, 0], [TR, TC], out)

else:
    class LlamaPrimitives:
        """Install pypto (pip install -e pypto_all/pypto) for PTO-ISA primitives."""

        pass
