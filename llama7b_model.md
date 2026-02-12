# Llama 7B Model: Top-Down Hierarchy and Composite Functions (PyPTO-Lib)

本文档在 **pypto-lib** 下按 **pypto** 目录定义的 PTO-ISA 接口与 API，自顶向下构建 Llama 7B 模型及其所需的 **composite functions**。层次中通过 **tiling** 缩小中间数据规模，在数据能放入 **AI Cube / AI Vector** 的粒度插入 **incore_scope**；最底层使用 **PTO-ISA 指令** 以及 **tensor↔tile** 的 casting（在 pypto 中对应 `pl.load` / `pl.store`）完成张量运算。模型支持 **batch**，且每个 batch 的序列长度由 **sequence_length[batch]** 数组给出（变长序列）。

---

## 1. 符号与配置

### 1.1 模型配置 (Llama 7B)

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `B` | batch size | 配置决定 |
| `S` | 当前 batch 的有效最大序列长度 | `max(sequence_length[0:B])` |
| `H` | hidden size | 4096 |
| `n_heads` | 注意力头数 | 32 |
| `head_dim` | 每头维度 | 128 (= H / n_heads) |
| `L` | transformer 层数 | 32 |
| `V` | vocab size | 32000 |
| `M` | MLP intermediate size | 11008 |

### 1.2 变长序列

- **sequence_length**: 一维数组 `sequence_length[b]` 表示第 `b` 个 batch 的有效序列长度（`0 <= b < B`）。
- 计算时对第 `b` 个样本只使用 `input_ids[b, 0 : sequence_length[b]]` 及对应位置的状态；超出部分可用 mask 或按 tile 用 `valid_row`/`valid_col` 处理。
- 文档中形状以 `[B, S, ...]` 表示逻辑维度；实现时可按 `sequence_length[b]` 做 tiling 或 mask。

### 1.3 Tiling 与 incore 粒度

- **Tiling**：在每一层对张量按块切分（如 `TB` batch 块、`TS` 序列块、`TH`/`TD` 隐藏/头维度块），使下一层中间张量尺寸变小。
- **Incore 插入条件**：当某层输出的**单块**形状（或单次调用的 tile 形状）能放入 AI Cube / AI Vector 的 tile 容量（如 16×16、32×32、64×64 等，由后端决定）时，在该层**插入 incore_scope**；scope 内用 PTO-ISA（见第 5 节）。
- **最底层**：必须用 **PTO-ISA 指令** 和 **tensor→tile（load） / tile→tensor（store）** 完成运算，对应 pypto 的 `pl.load`、`pl.store` 及 `pl.*` block 运算。

---

## 2. 自顶向下层次总览

```
Level 0:  llama_forward(input_ids, sequence_length, ...)     [B, S] → [B, S, V]
    ↓ tiling over batch & seq
Level 1:  embedding_tiled, transformer_block_tiled, lm_head_tiled
    ↓ tiling over hidden / heads / seq
Level 2:  attention_tiled, mlp_tiled, rmsnorm_tiled, ...
    ↓ tile size fits AI Cube/Vector → insert incore_scope
Level 3:  incore_scope { PTO-ISA + load/store (tensor↔tile) }
```

---

## 3. Level 0：模型入口（支持 batch 与 sequence_length）

### 3.1 顶层函数

```python
def llama_forward(
    input_ids: Tensor,           # [B, max_S], int32
    sequence_length: Tensor,     # [B], int32, sequence_length[b] = 有效长度
    embedding_weight: Tensor,    # [V, H]
    # ... 其他权重 ...
) -> Tensor:                     # [B, max_S, V] 或按 sequence_length 的 logits
    B = input_ids.shape[0]
    max_S = input_ids.shape[1]
    # 有效 S 由 sequence_length 决定，后续各层按 b 使用 sequence_length[b]
    hidden = embedding_tiled(input_ids, embedding_weight, sequence_length)  # [B, max_S, H]
    for layer_id in range(L):
        hidden = transformer_block_tiled(layer_id, hidden, sequence_length, ...)
    hidden = rmsnorm_tiled(hidden, final_norm_weight, sequence_length)
    logits = lm_head_tiled(hidden, lm_head_weight, sequence_length)  # [B, max_S, V]
    return logits
```

- 不在此层做细粒度 tiling；只按 **batch 维** 或 **batch×seq** 分块调用下一层 composite，以便中间张量在 Level 1 缩小。

### 3.2 Tiling 策略（Level 0 → 1）

- 对 `embedding_tiled`、`transformer_block_tiled`、`lm_head_tiled` 的调用可按 **batch 块** `TB` 或 **(TB, TS)** 划分：
  - 例如对 `hidden [B, S, H]` 按 `(b_start, b_end)` 与 `(s_start, s_end)` 切出 `hidden[b_start:b_end, s_start:s_end, :]`，再传入下一层。
- 这样 Level 1 每次处理的张量形状为 `[TB, TS, H]` 等，中间数据量下降。

---

## 4. Level 1：Composite 函数与 tiling

### 4.1 embedding_tiled

- **输入**: `input_ids [B, max_S]`, `embedding_weight [V, H]`, `sequence_length [B]`
- **输出**: `hidden [B, max_S, H]`
- **Tiling**: 按 `(TB, TS)` 切 batch 与 seq；每次处理 `input_ids[b_chunk, s_chunk]`，查表得 `[TB, TS, H]`，写回 `hidden` 对应块。
- **Incore 插入**: 当 `TB*TS` 或单块 `[TB, TS, H]` 在维度上可被切成 AI Cube/Vector 可容纳的 tile（如 32×128）时，在**块内循环**插入 `incore_scope`，scope 内做 embedding lookup 的 PTO-ISA（load 对应 tile、gather、store）。

```python
def embedding_tiled(input_ids, embedding_weight, sequence_length):
    B, max_S = input_ids.shape[0], input_ids.shape[1]
    H = embedding_weight.shape[1]
    hidden: Tensor  # [B, max_S, H], output
    for b_start in range(0, B, TB):
        for s_start in range(0, max_S, TS):
            b_end, s_end = min(b_start + TB, B), min(s_start + TS, max_S)
            with incore_scope():
                # 块内: 读 input_ids[b_start:b_end, s_start:s_end], 查表, 写 hidden[...]
                # 使用 pl.load / PTO-ISA / pl.store，tile 形状选为 fit AI Cube/Vector
                ...
    return hidden
```

### 4.2 transformer_block_tiled

- **输入**: `hidden [B, S, H]`, `sequence_length [B]`, 本层权重
- **输出**: `hidden [B, S, H]`（inplace 或新 tensor）
- **内部**: 先 `attention_tiled`，再 `mlp_tiled`，每部分都可按 (TB, TS, TH) 等进一步切块并在合适处插入 incore_scope。

```python
def transformer_block_tiled(layer_id, hidden, sequence_length, ...):
    residual = hidden
    hidden = rmsnorm_tiled(hidden, attn_norm_weight, sequence_length)
    hidden = attention_tiled(hidden, sequence_length, q_proj, k_proj, v_proj, o_proj, ...)
    hidden = add_tiled(hidden, residual)
    residual = hidden
    hidden = rmsnorm_tiled(hidden, mlp_norm_weight, sequence_length)
    hidden = mlp_tiled(hidden, gate_proj, up_proj, down_proj)
    hidden = add_tiled(hidden, residual)
    return hidden
```

### 4.3 attention_tiled（Level 2 入口）

- **Q, K, V**: 从 `hidden` 经线性层得到；形状逻辑上为 `[B, n_heads, S, head_dim]`。
- **Tiling**: 按 `(TB, TS_q, TS_kv, n_heads_chunk)` 等切分，使单块 Q/K/V 的 tile 可放入 incore。
- **Incore 插入**: 在计算 `Q@K^T`、softmax、`Attn@V` 的**每块**上，当块形状适合 AI Cube/Vector 时，用 incore_scope 包住该块计算；内部用 PTO-ISA：matmul、row_max/row_sum、exp、row_expand_sub、row_expand_div、再 matmul、等。

### 4.4 mlp_tiled

- **结构**: Gate/Up 投影 → SiLU(gate)*up → Down 投影。
- **Tiling**: 按 `(TB, TS, TH)` 切 hidden 与 intermediate 维；块内做线性 + 激活 + 线性。
- **Incore 插入**: 当线性层输出的块为 (TB, TS, chunk_H) 或 (TB, TS, M_chunk) 且单 tile 可放入硬件时，在该块计算处加 incore_scope，内部用 load/matmul/add/store 等 PTO-ISA。

### 4.5 lm_head_tiled

- **输入**: `hidden [B, S, H]`
- **输出**: `logits [B, S, V]`
- **Tiling**: 按 (TB, TS) 与 V 的块 `V_chunk` 切分；块内 `hidden @ lm_head_weight.T`，得到 `[TB, TS, V_chunk]`。
- **Incore**: 当 `[TB, TS]` 与 `V_chunk` 的 matmul 块适合 tile 时，在该块内插入 incore_scope。

---

## 5. 最底层：PTO-ISA 与 tensor↔tile casting

在 **incore_scope** 内部，所有张量运算必须用 **pypto** 定义的 PTO-ISA 接口完成，即：

- **Tensor → Tile（读）**: 使用 **`pl.load(tensor, offsets, shapes)`**，对应 PTO-ISA 的 `pto.subview` + `pto.tload`（或后端等价指令）；语义上即“cast_tensor_to_tile + 数据加载”。
- **Tile → Tensor（写）**: 使用 **`pl.store(tile, offsets, shapes, output_tensor)`**，对应 `pto.subview` + `pto.tstore`；即 tile 写回 tensor 的某块区域。
- **Tile 上运算**: 使用 pypto language 的 block 运算，它们会 lower 到 PTO-ISA，例如：

| 运算类型 | PyPTO API (pl.*) | PTO-ISA / 说明 |
|----------|------------------|----------------|
| 加载/写回 | `pl.load`, `pl.store` | tload, tstore（tensor↔tile） |
| 元素二元 | `pl.add`, `pl.mul`, `pl.sub`, `pl.div` | tadd, tmul, tsub, tdiv |
| 元素一元 | `pl.exp`, `pl.sqrt`, `pl.recip` | 对应 unary |
| 规约 | `pl.row_sum`, `pl.row_max`, `pl.row_min` | rowsum, rowmax, rowmin |
| 扩展 | `pl.row_expand_sub`, `pl.row_expand_div` | 用于 softmax (x - max) / sum |
| 矩阵乘 | `pl.matmul` | matmul 类指令 |
| 创建 tile | `pl.create_tile`, `pl.full` | alloc_tile / 常量 tile |

### 5.1 示例：incore 内一块 Attention 的 softmax（PTO-ISA 层）

对一块 `scores [tile_rows, tile_cols]`（例如 32×128），在 incore_scope 内用 tile 运算完成 softmax：

```python
with incore_scope():
    # scores 已通过 load 得到为 tile_scores
    tile_max: Tile = pl.row_max(tile_scores, tmp_tile)           # [tile_rows, 1]
    tile_scores = pl.row_expand_sub(tile_scores, tile_max)       # scores - max
    tile_exp: Tile = pl.exp(tile_scores)
    tile_sum: Tile = pl.row_sum(tile_exp, tmp_tile)              # [tile_rows, 1]
    tile_softmax: Tile = pl.row_expand_div(tile_exp, tile_sum)
    # 再 pl.store(tile_softmax, ...) 或继续与 V 做 matmul
```

- 此处 `tmp_tile` 为 row_max/row_sum 等规约所需的临时 tile（由 create_tile 或传入）。

### 5.2 示例：incore 内一块线性层（load + matmul + store）

```python
# 块: hidden [TB, TS, TH] @ weight [TH, TD] -> out [TB, TS, TD]
# 进一步切成 tile 级: 例如 (tr, tc) 为 tile 行列
with incore_scope():
    tile_h = pl.load(hidden, [ro, co], [tile_rows, tile_cols])
    tile_w = pl.load(weight, [wo, co], [tile_cols, tile_out_cols])
    tile_out = pl.matmul(tile_h, tile_w)   # 或 matmul_acc 若累加
    pl.store(tile_out, [out_ro, out_co], [tile_rows, tile_out_cols], out_tensor)
```

- 实际实现中需按 pypto 的 `pl.load`/`pl.store` 的 offsets/shapes 与循环配合，处理 `sequence_length[b]` 的边界（tail、mask 或 valid_row/valid_col）。

### 5.3 sequence_length 在最底层的使用

- **Masking**: 对第 `b` 个 batch，在 seq 维上只处理 `0..sequence_length[b]`；在 tile 内可用 **predicate** 或 **valid_row/valid_col**（PTO-ISA 支持的有效范围）屏蔽超出部分。
- **Tiling 边界**: 对 `s_start, s_end`，若 `s_end > sequence_length[b]`，则该块为 tail block，按 pypto-lib 的 tail/padding 策略处理（mask 或小块）。

---

## 6. 文件与模块建议（pypto-lib 子目录）

在 **pypto-lib** 下可组织为：

```
pypto-lib/
  llama7b_model.md          # 本文档
  llama7b/
    __init__.py
    config.py               # B, H, n_heads, L, V, M, TB, TS, tile_* 等常量
    level0_forward.py       # llama_forward 入口，sequence_length 传入
    level1_embedding.py     # embedding_tiled
    level1_transformer.py  # transformer_block_tiled, attention_tiled, mlp_tiled
    level1_lm_head.py      # lm_head_tiled
    level2_attention.py    # 带 tiling 的 attention，内部 incore_scope + PTO-ISA
    level2_mlp.py           # 带 tiling 的 mlp，内部 incore_scope + PTO-ISA
    level2_rmsnorm.py      # rmsnorm_tiled，incore_scope 内 row_sum/平方/scale
    primitives.py           # 最底层封装：pl.load/store + block 运算，供 level2 调用
```

- **primitives.py** 中只使用 **pl.load**、**pl.store** 以及 **pl.*** 的 block 运算（对应 PTO-ISA）；不在此层以上再引入新的“tile 级原语”，与 pypto-lib.md 一致。
- **level2_*.py** 中在循环内根据块形状判断并插入 **incore_scope**；scope 内调用 primitives 或直接写 pl.load/store/pl.matmul/pl.row_max 等。

---

## 7. 小结

| 层次 | 内容 | Tiling | Incore |
|------|------|--------|--------|
| Level 0 | llama_forward(input_ids, sequence_length, ...) | 按 B/S 分块调用 Level 1 | 否 |
| Level 1 | embedding_tiled, transformer_block_tiled, lm_head_tiled | (TB, TS) 等，缩小中间张量 | 在块内可插 |
| Level 2 | attention_tiled, mlp_tiled, rmsnorm_tiled | (TB, TS, TH) 等，块 fit tile | **是**，插入 incore_scope |
| 最底层 | pl.load / pl.store + pl.matmul, pl.row_max, pl.exp, ... | tile 级 (tr, tc) | scope 内全为 PTO-ISA + tensor↔tile |

- **Batch**：所有 composite 接受 `sequence_length [B]`，在 tiling 与 mask/valid 处使用。
- **最底层**：仅使用 PTO-ISA 接口与 tensor→tile（load）、tile→tensor（store）完成运算，符合 pypto 目录中的定义。
