# Week 1 executable examples and figures

Run each block from `LLM/` with `uv run python 'week 1/codeblocks/<file>.py'`; the adjacent `.txt` file is captured stdout.

## Code blocks

| File | Demonstrates | topics.json id |
|---|---|---|
| `01_attention_from_scratch.py` | NumPy scaled dot-product attention and PyTorch SDPA agreement | `w1-scaled-dot-product` |
| `02_dot_product_variance.py` | Empirical Var(q·k) growth with d_k and scaled variance | `w1-scaled-dot-product` |
| `03_multihead_shape_check.py` | `[B,T,d_model]` → `[B,h,T,d_k]` split and exact MHA module agreement | `w1-multi-head-attention` |
| `04_encoder_parameter_count.py` | Formula against PyTorch encoder-layer count; includes biases and two affine norms | `w1-encoder-parameter-count` |
| `05_additive_attention.py` | Bahdanau additive scores, normalized alignments, weighted context | `w1-bahdanau-attention` |

## Figures

| File | What it shows | Factual caption suggestion |
|---|---|---|
| `assets/01_scaled_attention_heatmap.png` | Scaled dot-product attention on five toy tokens | Each query row sums to one after softmax; cells show the normalized weight assigned to each key token. |
| `assets/02_scaling_saturation.png` | Dot-product standard deviation, mean maximum probability and derivative magnitude | Unscaled dot-product spread grows with d_k while dividing by √d_k keeps its standard deviation near one and delays softmax saturation. |
| `assets/03_seq2seq_vs_alignment.png` | Reused fixed context versus Bahdanau alignment matrix | The left panel reuses a uniform context distribution at each output step; the right panel computes a separate learned alignment per decoder step. |
| `assets/04_multihead_patterns_shapes.png` | Four random head patterns and width flow | Four independently projected 64-wide heads concatenate to 512 features in the base-dimension schematic. |
| `assets/05_encoder_parameter_breakdown.png` | Base encoder parameter groups | With d_model=512 and d_ff=2048, FFN parameters dominate MHA, while norm parameters are comparatively small. |
