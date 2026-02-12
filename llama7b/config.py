# Llama 7B model and tiling config (see llama7b_model.md)
# Tiling sizes are chosen so that at incore_scope level, tiles fit AI Cube/Vector.

# Model (Llama 7B)
HIDDEN_SIZE = 4096
N_HEADS = 32
HEAD_DIM = 128  # HIDDEN_SIZE // N_HEADS
N_LAYERS = 32
VOCAB_SIZE = 32000
MLP_INTERMEDIATE_SIZE = 11008

# Tiling: Level 1 (composite) -> smaller blocks
TB = 1   # batch tiling (e.g. 1 or 2)
TS = 128 # sequence tiling
TH = 128 # hidden chunk for linear layers

# Tile size for incore_scope (must fit AI Cube/Vector; adjust per backend)
TILE_ROWS = 32
TILE_COLS = 128

# sequence_length: 1D array of length batch_size; sequence_length[b] = valid length for batch b
