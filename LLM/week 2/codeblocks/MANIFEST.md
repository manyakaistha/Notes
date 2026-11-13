# Week 2 executable examples and figures

Run each block from `LLM/` with `uv run python 'week 2/codeblocks/<file>.py'`; the adjacent `.txt` file is captured stdout.

## Code blocks

| File | Demonstrates | topics.json id |
|---|---|---|
| `01_causal_mask.py` | Upper-triangular `-inf` mask and exact zero future weights | `w2-masked-self-attention` |
| `02_cross_attention_shapes.py` | Decoder Q against encoder K/V with distinct sequence lengths | `w2-cross-attention` |
| `03_sinusoidal_pe.py` | Sin/cos positional encoding implementation compared to formula | `w2-positional-encoding` |
| `04_norm_axes.py` | LayerNorm feature axis and BatchNorm batch/sequence-dependent statistics | `w2-residual-layernorm` |
| `05_label_smoothing.py` | Smoothed categorical target and manual/PyTorch cross-entropy equality | `w2-training-tricks` |
| `06_warmup_schedule.py` | Inverse-square-root learning-rate schedule at requested steps | `w2-training-tricks` |
| `07_decoder_parameter_count.py` | Decoder formula compared with PyTorch self/cross-attention, FFN, and 3 norms | `w2-decoder-output-and-params` |
| `08_teacher_forcing.py` | Gold-prefix training input versus autoregressive prediction input | `w2-teacher-forcing` |

## Figures

| File | What it shows | Factual caption suggestion |
|---|---|---|
| `assets/01_sinusoidal_pe_heatmap.png` | Sinusoidal values over 100 positions and 64 dimensions | Paired sine and cosine channels show fast position variation at low dimension indices and progressively slower variation at higher indices. |
| `assets/02_pe_relative_offset.png` | Rotation identity and encoding dot product by offset | A 2×2 rotation per frequency maps PE(pos) to PE(pos+7) up to floating-point error; the encoding dot product varies with relative offset. |
| `assets/03_norm_axes.png` | LayerNorm and BatchNorm axes on a `[B,T,d]=[2,3,4]` tensor | LayerNorm statistics are computed per token across features, whereas BatchNorm statistics pool observations by feature across batch and sequence locations. |
| `assets/04_warmup_schedule.png` | Warm-up and inverse-square-root decay for d_model 512 and 256 | Both curves peak at step 4000; d_model=512 has lower rates than d_model=256 by the inverse square-root width factor. |
| `assets/05_causal_mask_softmax.png` | Causal blocking matrix and attention probabilities | All entries above the causal diagonal receive zero probability after pre-softmax masking. |
| `assets/06_label_smoothing.png` | ε=0.1 target distribution and loss against confidence | The correct class receives 0.92 target mass for V=5; the remaining 0.08 is distributed over the other classes. |
