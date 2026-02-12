# Llama 7B (PyPTO-Lib)

Top-down Llama 7B model and composite functions for PyPTO-Lib. Supports **batch** and **variable sequence length** via `sequence_length[batch]`.

- **Design**: [../llama7b_model.md](../llama7b_model.md) — hierarchy, tiling, incore_scope insertion, PTO-ISA bottom layer.
- **Config**: [config.py](config.py) — model and tiling constants.
- **Level 0**: [level0_forward.py](level0_forward.py) — `llama_forward(input_ids, sequence_length, weights)`.
- **Level 1**: [level1_embedding.py](level1_embedding.py), [level1_transformer.py](level1_transformer.py), [level1_lm_head.py](level1_lm_head.py) — tiled composites.
- **Level 2**: [level2_rmsnorm.py](level2_rmsnorm.py), [level2_mlp.py](level2_mlp.py), [level2_attention.py](level2_attention.py) — tiled + incore_scope.
- **Primitives**: [primitives.py](primitives.py) — `LlamaPrimitives` @pl.program with add_tile, linear_tile, rmsnorm_tile, softmax_tile, matmul_tile, silu_tile (PTO-ISA only).

Requires **pypto** on `PYTHONPATH` (e.g. `pypto_all/pypto/python`) to import `pypto.language` for primitives.
