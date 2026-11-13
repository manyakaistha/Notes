# Second-pass audit — independent review and comparison

**Scope.** `LLM/week 1/week-1-notes.md`, `LLM/week 1/week-1-exam-focused-notes.md`, `LLM/week 2/week-2-notes.md`, and `LLM/week 2/week-2-exam-focused-notes.md`. The first review has not been opened at this point. The only quantitative note content treated as claims is data counts, dimensions, formula outputs, ratios, probabilities, parameter totals, learning-rate values, and explicit examples; exam IDs, years in IDs, Markdown heading numbers, code seeds and repeated citation numerals are not independent claims. Duplicate numerical statements are additionally checked via exact code/output comparison or repeated self-checks.

## 1. Numeric-claim extraction and re-derivation

The digit-bearing-line extraction was run with:

```sh
cd LLM && uv run python /tmp/luna-audit/extract_numeric.py > /tmp/luna-audit/numeric-claims.tsv
```

The raw extraction has 402 lines including section headers, record IDs, dates, code, and outputs. The ledger below normalizes those into quantitative claims, re-derives 37 numerical assertions, and records all 20 self-check answers separately. Routed counts were recomputed from `sources/exam-papers/routing.json`, not transcribed from the note tables.

### Routing counts

| File / line | Claim | Independent derivation | Result |
|---|---|---|---:|
| `week-1-exam-focused-notes.md:21,25–31` | W1 records and topic counts | Filter `routing.json` to `week == 1`; count records and each record’s `topic_ids` | 73 records; 30 self-attention/QKV; 17 MHA; 14 encoder parameters; 14 scaled dot product; 8 FFN/encoder stack; 4 Bahdanau; 2 seq2seq |
| `week-2-exam-focused-notes.md:27,31–38` | W2 topic counts | Filter `routing.json` to `week == 2`; count records and each topic ID | 65 records; 21 residual/LN; 15 causal attention; 11 PE; 8 teacher forcing; 5 cross-attention; 4 decoder output/params; 4 full Transformer; 1 training tricks |

All 17 displayed totals/counts match. Exact source recomputation:

```sh
cd LLM && uv run python - <<'PY'
import json
from collections import Counter
r=json.load(open('sources/exam-papers/routing.json'))['records']
for week in (1,2):
    rows=[x for x in r if str(x['week'])==str(week)]
    print(week, len(rows), Counter(t for x in rows for t in x['topic_ids']))
PY
```

### Formula, shape, and worked-number ledger

The independent script `/tmp/luna-audit/rederive.py` passes 37 assertions; it recomputes the routed counts and core formulas/examples. The claim ledger below records the quantitative statements and derivations, while the separate output ledger records every numeric stdout block regenerated from the 13 embedded examples.

| # | Note line(s) | Claim re-derived | Independent result |
|---:|---|---|---|
| 1 | W1 `:71`; exam `:80–81,89–90` | Two equal scores on values `[1,0]`, `[0,2]` | Weights `[0.5,0.5]`; output `[0.5,1]` |
| 2 | W1 exam `:83–84` | `[3,7,12] @ [12,4]`, then self-attention | Q `[3,7,4]`; QKᵀ `[3,7,7]` |
| 3 | W1 `:73`; exam `:43,86–87` | Independent zero-mean unit-variance coordinates | `Var(q·k)=d_k`; dividing by √d_k gives variance 1 under stated independence assumptions |
| 4 | W1 `:94`; exam `:92–93` | Base width and head split | `512/8=64`; `8×64=512`; `W^O=[512,512]` |
| 5 | W1 `:65,69,108`; exam `:41,46` | Row-oriented Q/K/V, scores, output, FFN dimensions | Matrix products conform: `[d_model,d_k]`, `[d_model,d_v]`; FFN `[d_model,d_ff]→[d_ff,d_model]` |
| 6 | W1 `:122,124` | Encoder weights only, `d=512,f=2048` | `4d²+2df=3,145,728` |
| 7 | W1 `:122,124` | Add FFN biases only | `+f+d=2,560`; subtotal `3,148,288` |
| 8 | W1 `:124` | Add four attention projection biases, before norms | `+4d=2,048`; subtotal `3,150,336` |
| 9 | W1 `:124,259–277` | Add two affine LayerNorms and compare PyTorch | `+4d=2,048`; full layer `3,152,384`, matching `nn.TransformerEncoderLayer` with `bias=True`, `norm_first=False` |
| 10 | W1 `:124` | Component counts under the stated biased convention | MHA `1,050,624`; FFN `2,099,712`; norms `2,048`; sum `3,152,384` |
| 11 | W1 `:124` | Six independent encoder layers | `6×3,152,384=18,914,304` |
| 12 | W1 exam `:98–99` | Standard MHA weight matrices at `d=256` | `4×256²=262,144` |
| 13 | W1 exam `:101–102` | FFN at `d=4,f=8` | Weights `4×8+8×4=64`; biases `8+4=12`; total `76` |
| 14 | W1 `:323`; exam `:104–105` | Dense score-entry count for `(B,h,T)=(4,2,12)` and doubled T | `1,152`; at `T=24`, `4,608`, exactly 4× |
| 15 | W1 `:324` | Per-head `[512,64]` projection | `512×64=32,768` |
| 16 | W1 `:325`; exam `:95–96` | Row-vector W^O toy map | `[1,2,3,4] @ [[1,0],[0,1],[1,0],[0,1]]=[4,6]` |
| 17 | W2 `:60`; exam `:83–84` | Softmax of visible scores `[2,1]` | `[0.73105858,0.26894142]`, rounding to `[0.7311,0.2689]` |
| 18 | W2 `:70`; exam `:86–87` | Cross-attention for `B=2,T=3,S=5,d=16,h=4` | `d_k=d_v=4`; Q `[2,3,4]`; K/V `[2,5,4]`; weights `[2,4,3,5]`; output `[2,3,16]` |
| 19 | W2 `:76` | FFN for `d=512,f=2048` | `2df+f+d=2,099,712` |
| 20 | W2 `:76`; exam `:101–102` | Vocabulary projection, `d=512,V=37,000` | Weights `18,944,000`; optional bias `37,000`; total with bias `18,981,000` |
| 21 | W2 `:78–82`; exam `:98–99` | Biased decoder layer components | Attention `2,101,248`; FFN `2,099,712`; three affine norms `3,072`; total `4,204,032` |
| 22 | W2 `:82`; exam `:98–99` | Six decoder layers | `6×4,204,032=25,224,192`, excluding embeddings/output as stated |
| 23 | W2 `:90–92`; exam `:56–57,89–90` | PE at p=0, d=4 and squared norm | `[0,1,0,1]`; norm² `2`; pairwise identity gives general norm² `d/2` |
| 24 | W2 `:94,104`; exam `:92–93` | Fixed-offset PE transform | Rotation matrix with angle `kω`; checked numerically at `k=7`, output equals the p+k encoding to double precision |
| 25 | W2 `:118`; exam `:95–96` | LayerNorm on `[1,2,3]` | Mean `2`; population variance `2/3`; normalized values `[-1.22474487,0,1.22474487]`; `2d=1,024` affine params at d=512 |
| 26 | W2 `:120`; exam `:72` | BatchNorm normalization set containing one value | Centered value `0`; after affine scale/shift output `β`; implementations may reject a one-value training set |
| 27 | W2 `:128,132`; exam `:63,104–105` | Uniform-mixture label smoothing `V=5, ε=.1` | `[.02,.02,.92,.02,.02]`, sum `1`; alternate non-gold-only convention gives gold `.9`, others `.025` |
| 28 | W2 `:136–142`; exam `:107–108` | Warmup, `d=512,w=4000` | Step 1000 `.0001746928107`; peak `.0006987712430`; step 8000 `.0004941058844`; first is 1/4 peak and last is peak/√2. Width 256 peak is √2 times width 512. |
| 29 | W2 `:138` | Original training hyperparameters | Adam `(β1,β2,ε)=(.9,.98,10⁻⁹)`; base dropout `.1` |
| 30 | W1 `:114`; W2 `:144` | Original stack depth | Six encoder and six decoder layers; confirmed against the paper and transcript |
| 31 | W2 `:100,104`; code `03_sinusoidal_pe.py` | Figure PE dimensions/offset | 100 positions × 64 channels; offset example `k=7`; code output separately reproduced exactly |
| 32 | W1 `:81,85`; code `02_dot_product_variance.py` | Five-token attention rows and random widths | Widths `4,16,64,256`; seeded variance estimates in stored `.txt` reproduce exactly |
| 33 | W1 `:100,132`; code `03_multihead_shape_check.py` | Four illustrative patterns vs base 8×64 schematic | The caption explicitly distinguishes 4 displayed patterns from the separate 8×64=512 base-width sketch |
| 34 | W2 `:124`; code `04_norm_axes.py` | Normalization example shapes and axes | `[B,T,d]=[2,3,4]` in caption; script verifies LN per token and BN over batch/time |
| 35 | W2 `:142`; code `06_warmup_schedule.py` | Schedule plot widths | widths 256 and 512; same-step ratio √2, as above |
| 36 | W2 `:144` | Full model data flow | Encoder depth 6; decoder depth 6; decoder has three sublayers per layer |
| 37 | W1 `:128`; exam `:49,57` | Sequence-length dependence | Dense scores `[B,h,T,T]`, `BhT²`; doubling T quadruples entries, shared projection weights unchanged |

### The 20 self-checks — independently solved

| File / question | Recomputed answer | Result |
|---|---|---|
| W1 exam 1 | Fixed context `c` must carry source information for every target step; attention recomputes `c_t` from encoder states, while recurrent decoding remains sequential. | Correct |
| W1 exam 2 | Equal scores give weights `[1/2,1/2]`; `c=[1,2]`. | Correct |
| W1 exam 3 | Q `[3,7,4]`; QKᵀ `[3,7,7]`. | Correct |
| W1 exam 4 | Independent products sum to variance `d_k`; division by √d_k yields 1. | Correct under stated assumptions |
| W1 exam 5 | Scores `[0,0]`, weights `[1/2,1/2]`, output `[1/2,1]`. | Correct |
| W1 exam 6 | Head width `64`; concatenated width `512`; W^O `[512,512]`. | Correct |
| W1 exam 7 | Row multiplication gives `[4,6]`. | Correct |
| W1 exam 8 | `4×256²=262,144`. | Correct |
| W1 exam 9 | FFN weights `64`, biases `12`, total `76`. | Correct |
| W1 exam 10 | `BhT²=1,152`; doubling T gives `4,608`, fourfold; projection parameters unchanged. | Correct |
| W2 exam 1 | Shifted training input `[BOS,A,B]`; prediction 3 receives gold `B`; free running may use mistaken `C`. | Correct |
| W2 exam 2 | After mask, softmax `[.73105858,.26894142,0,0]`. | Correct |
| W2 exam 3 | Per-head width `4`; Q `[2,3,4]`; K/V `[2,5,4]`; weights `[2,4,3,5]`; projected output `[2,3,16]`. | Correct |
| W2 exam 4 | PE at zero `[0,1,0,1]`; squared norm `2`; general `d/2`. | Correct |
| W2 exam 5 | Offset map `[[cos(kω),sin(kω)],[-sin(kω),cos(kω)]]` applied to `[sin(pω),cos(pω)]ᵀ`. | Correct |
| W2 exam 6 | Mean `2`, variance `2/3`, output `[-1.22474487,0,1.22474487]`. | Correct |
| W2 exam 7 | Attention `2,101,248`; FFN `2,099,712`; norms `3,072`; total `4,204,032`. | Correct |
| W2 exam 8 | Projection `18,944,000`; optional bias `37,000`; total `18,981,000`. | Correct |
| W2 exam 9 | `[.02,.02,.92,.02,.02]`; sum 1. | Correct |
| W2 exam 10 | At steps 1000/4000/8000: `.0001746928/.0006987712/.0004941059`; ratios 1/4 and 1/√2. | Correct |

## 2. Formula and convention audit

Compared to the primary [Vaswani et al. 2017 NeurIPS paper](https://papers.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), plus Ba et al. for LayerNorm and the original Google BERT implementation for BERT normalization:

| Topic | Source definition/check | Notes’ treatment | Result |
|---|---|---|---|
| Scaled dot-product attention | `softmax(QKᵀ/√d_k)V` | Same; score mask added before softmax where used | Match |
| Multi-head and W^O | `Concat(heads)W^O`, `W^O ∈ R^(h d_v × d_model)` | Same row-vector orientation and right multiplication | Match |
| FFN | `max(0,xW1+b1)W2+b2`; `W1:d_model×d_ff`, `W2:d_ff×d_model` | Same row-vector convention | Match |
| Sinusoidal PE | `sin(pos/10000^(2i/d_model))`, `cos(...)` | `ω_i=10000^(-2i/d)` and sin/cos pair; `d` is model width in full note | Equivalent |
| LayerNorm | Per-example hidden-feature mean/variance and affine `γ,β` | Population variance over d features, per token; `2d` affine params | Match |
| Residual + normalization | Vaswani output is `LayerNorm(x+Sublayer(x))`; decoder has 3 sublayers | Explicit post-LN; pre-LN written separately; dropout noted before addition | Match |
| Label smoothing | Paper reports `ε_ls=.1` and better accuracy/BLEU, worse perplexity; target-distribution convention is not defined there | Uniform-mixture convention explicitly named; alternate convention explicitly distinguished | Correctly scoped |
| Warm-up / Adam | `d_model^-1/2 min(step^-1/2, step·warmup^-3/2)`, warmup 4000; Adam `.9,.98,10^-9`; dropout `.1` | Same values and formula | Match |
| Encoder parameter count | Matrix weights only: `4d²+2df`; biases and norm parameters implementation-dependent | Weights-only, FFN-bias-only, attention-bias, and two affine LN terms separated; PyTorch count explicitly post-LN/bias=True | Match; assumptions stated |
| Decoder parameter count | 2 attention modules + FFN + 3 norms under declared PyTorch configuration | `2(4d²+4d)+(2df+f+d)+6d` | Match |
| Tied embeddings/output | Paper shares embedding and pre-softmax matrix; tied weights counted once | Full note flags tying and excludes output from layer totals | Match |

**Orientation check.** With row-oriented `X:[T,d]`, Q/K/V projections are on the right, scores are `QKᵀ`, values aggregate as `AV`, head outputs concatenate by feature, and `W^O` multiplies on the right. For column-stored token inputs, transpose the input and projection matrices before using the row formula; the resulting output remains the same row-oriented `AV`. The FFN’s stated row shapes require `xW1`, then `W2`. Both note variants state these consistently.

## 3. Code and output identity / execution

Exact command:

```sh
cd LLM && uv run python /tmp/luna-audit/verify_artifacts.py
```

All 13 embedded Python blocks and 13 text output blocks matched their corresponding `codeblocks/NN_*.py/.txt` byte for byte. Each `.py` was then run with `uv run python` from `LLM/`; its captured stdout matched the `.txt` bytes exactly. No mismatch or nondeterministic output.

| Week | Script / matching text output | Result |
|---|---|---|
| 1 | `01_attention_from_scratch.py/.txt` | exact code/text match; rerun stdout exact |
| 1 | `02_dot_product_variance.py/.txt` | exact code/text match; rerun stdout exact |
| 1 | `03_multihead_shape_check.py/.txt` | exact code/text match; rerun stdout exact |
| 1 | `04_encoder_parameter_count.py/.txt` | exact code/text match; rerun stdout exact |
| 1 | `05_additive_attention.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `01_causal_mask.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `02_cross_attention_shapes.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `03_sinusoidal_pe.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `04_norm_axes.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `05_label_smoothing.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `06_warmup_schedule.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `07_decoder_parameter_count.py/.txt` | exact code/text match; rerun stdout exact |
| 2 | `08_teacher_forcing.py/.txt` | exact code/text match; rerun stdout exact |

### Executed output numeric ledger

The exact `.txt` contents were also read and their numeric stdout claims regenerated by the 13 script runs. These values are not inferred from the table above; they are byte-compared to the stored outputs.

| Artifact | Numeric output re-derived |
|---|---|
| W1 `01_attention_from_scratch` | Shape `(1,2,3,4)`; row-sum range `1.000000..1.000000`; max NumPy/SDPA difference `0.00000011`. |
| W1 `02_dot_product_variance` | `d_k=4: var 4.04, scaled 1.011`; `16: 15.87, .992`; `64: 63.83, .997`; `256: 253.99, .992`; targets equal each d. |
| W1 `03_multihead_shape_check` | Input `[2,5,16]`; heads `[2,4,5,4]`; concat `[2,5,16]`; max difference `0.00000009`. |
| W1 `04_encoder_parameter_count` | MHA `1,050,624`; FFN `2,099,712`; norms `2,048`; total and PyTorch count both `3,152,384`; equality `True`. |
| W1 `05_additive_attention` | Scores `[.699,.539,-.081,-.522]`; alpha `[.384,.327,.176,.113]`; sum `1.000000`; context `[.351,-.993,-.053]`. |
| W2 `01_causal_mask` | 4×4 strict upper triangle blocked; weight rows `[1,0,0,0]`, `[.3100,.6900,0,0]`, `[.2344,.1286,.6370,0]`, `[.2352,.1743,.1168,.4737]`; future-weight sum `0.0`. |
| W2 `02_cross_attention_shapes` | Q `[2,3,16]`; K/V `[2,5,16]`; weights `[2,4,3,5]`; output `[2,3,16]`. |
| W2 `03_sinusoidal_pe` | Shape `(10,16)`; max implementation difference `0.00000011`; range `[-.989992,1.000000]`. |
| W2 `04_norm_axes` | Input `[2,2,3]`; LN per-token means all `0`; BN feature-0 values `[[-1.3416,-.4472],[.4472,1.3416]]`; single-token LN `[-1.2247,0,1.2247]`; training BN on `[1,3,1]` rejects one value/channel. |
| W2 `05_label_smoothing` | Target `[.02500000037,.02500000037,.9249999523,.02500000037]`; manual and PyTorch losses both `.66827178`; equality `True`. |
| W2 `06_warmup_schedule` | For d=512 at steps 1/1000/4000/8000: `.0000001747/.0001746928/.0006987712/.0004941059`; d=256: `.0000002471/.0002470529/.0009882118/.0006987712`. |
| W2 `07_decoder_parameter_count` | Attention `2,101,248`; FFN `2,099,712`; norms `3,072`; total and PyTorch count both `4,204,032`; equality `True`. |
| W2 `08_teacher_forcing` | Gold `[1,2,3,4]`; teacher input `[0,1,2,3]`; free-running `[0,1,7,6]`. |

## 4. Ebook wording/correction audit

Every sentence starting with “The ebook…” and every `Ebook correction` marker in the four scoped notes was inventoried with:

```sh
rg -n -i 'The ebook|Ebook correction|ebook correction' 'LLM/week 1/week-1-notes.md' 'LLM/week 1/week-1-exam-focused-notes.md' 'LLM/week 2/week-2-notes.md' 'LLM/week 2/week-2-exam-focused-notes.md'
```

Each was compared with the corresponding Week 1 or Week 2 transcript. Transcript excerpts and note excerpts are paired below.

| Note line and claim | Transcript evidence (exact excerpt) | Assessment |
|---|---|---|
| W1 `:48`: “The ebook’s definition is narrower than the architecture family.” | W1 transcript `:20–22`: “A Sequence-to-Sequence (seq2seq) model consists of two RNNs”; it describes a fixed context encoder/decoder. | Accurate qualification of that historical RNN architecture, not a false claim that all seq2seq systems use RNNs. |
| W1 `:77`: row/column attention conventions | W1 transcript `:195`: “The two formulations are transposes of each other — conceptually identical”; `:197`: “[possible-error p.8: The displayed outputs in the two conventions are identical after transposing the input matrices, rather than transposes of each other.]” Note `:77` says input transposition yields the same row-oriented `AV`. | Transcript’s marked correction is mathematically right; note’s orientation is right. |
| W1 `:96`: W^O orientation | W1 transcript `:218`: `W^O Concat(...)`; `:220`: “[possible-error p.9: With the row-oriented output defined in Eq. (7), the output projection must multiply on the right, or the head outputs must be transposed.]” | Correction right; paper’s row-vector form postmultiplies by W^O. |
| W1 `:116`: FFN orientation | W1 transcript `:253`: `max(0,W_1 z+b_1)W_2+b_2`; `:255` gives `W_1∈R^(512×2048)`; `:257`: “[possible-error p.10: The stated matrix shapes require zW_1, rather than W_1z, in the displayed FFN formula.]” | Correction right for the stated row-vector convention. |
| W1 `:124`; W1 exam `:107`: “1M + 2M ≈ 3M” mnemonic and “ebook’s corrected matrix orientations” | W1 transcript `:591`: “~3 million parameters per encoder layer”; `:603`: MHA `1,048,576≈1.05×10^6`; `:606`: FFN with biases `≈2.1×10^6`; `:608`: total `≈3.15M`. Orientation phrase points to the p.8, p.9, p.10 and p.12 correction annotations at W1 transcript `:197,220,257,342`. | Correct to call the parameter mnemonic rounded; corrected orientations are supported by the four annotations above. Exact note count uses a separately declared all-bias/affine-norm PyTorch convention. |
| W1 `:126`: head-count cost | W1 transcript `:340`: “The only extra cost is the projection W^O”; `:342`: “[possible-error p.12: A single-head attention layer can also include an output projection, so W^O is not inherently an extra multi-head cost.]” | Correction right; standard fixed-width projection weights remain `4d²`. |
| W1 `:324`: typo 32,178 | W1 transcript `:594`: `64×512=32,768 (per head; the slide rounds to 32,178 — a typo)`. | Note recomputes `512×64=32,768`; correction confirmed. |
| W1 `:331`: two-head example | W1 transcript `:528`: “[possible-error p.17: The displayed value matrix should be (V²)ᵀ=[[2,3],[0,4]]; the stated output also does not match the shown softmax weights.]” Note `:331` identifies the transpose and output inconsistency. | Recomputed scores `[[9.1924,8.4853],[8.4853,11.3137]]`; weights `[[.66976,.33024],[.05581,.94419]]`; corrected value multiplication gives `[[1.33952,3.33024],[.11161,3.94419]]`. Note is correct. |
| W2 `:82`: “ebook’s ~4M per layer is a rounded estimate” | W2 transcript `:169–177`: “Per-layer total: ≈4M”; Exam Tip `:176–177` gives encoder ≈3M, decoder ≈4M. | Correct characterization; note adds explicit exact configuration count `4,204,032`. |
| W2 `:120`; exam `:72`: BatchNorm correction | W2 transcript `:274`: “[possible-error p.28: With a single batch value, the displayed training-time normalization gives zero normalized activation and output β, not the identity.]” The PDF p.28 itself says “With m=1, σ_i=0 and BN becomes a no-op.” | The ebook error is real; the note’s singleton-set correction is correct. Its caveat that sequence BN may include time and batch-size-one need not mean one value is also correct. |
| W2 `:120`; exam `:76`: BERT normalization | W2 transcript `:309`: “(used by GPT-2, BERT-large, and most modern LLMs)”; `:313`: “[possible-error p.29: Original BERT-large uses post-layer normalization, not Pre-LN.]” The [original Google BERT source](https://github.com/google-research/bert/blob/master/modeling.py) applies `layer_norm(attention_output + layer_input)` and `layer_norm(layer_output + attention_output)`. | Claimed ebook error is real; post-residual LayerNorm correction is right. |

## 5. Page references and hidden conventions

Exact page check command:

```sh
cd LLM && uv run python /tmp/luna-audit/check_pages.py
```

All in-note PDF page references are present in the corresponding transcript page markers: W1 markers PDF `4–20` map to printed `3–19`; W2 markers PDF `21–33` map to printed `20–32`. Every inline range in the four notes resolves; no out-of-range or shifted PDF/printed reference was found. Independently, the source PDF pages 28 and 29 were rendered to `/tmp/luna-audit/ebook-p-028.png` and `ebook-p-029.png` and visually checked against the transcript corrections.

Convention audit found no unstated parameter-count assumptions in the claims reviewed: W1 distinguishes weights-only, FFN biases, attention projection biases, and two affine LayerNorms; W2 explicitly says “biased” and gives three affine norms; optional output bias is separate; tied weights are called out; pre-LN and post-LN wiring is explicitly distinguished; dropout omission/placement is stated. The output/tied-embedding convention is not included in layer totals and is explicitly named separately.

**Findings:** none. Severity counts: critical 0, high 0, medium 0, low 0. Clean categories and their exact checks are recorded above: 37 numeric assertions via `rederive.py`; 20 individually recomputed self-checks; 13 code and 13 output block byte comparisons plus 13 `uv run python` executions; formula comparison against the linked primary sources; all PDF page refs via `check_pages.py`; and each ebook correction paired with its transcript line.

## 6. Comparison with first review

The first review reports zero findings for the four overlapping note files. This second pass found no substantive disagreement on those four files: all numeric, formula, code, correction, and page-reference checks above passed. The first review additionally claims coverage of 15 concept notes, `ebook-discrepancies.md`, and every linked question ID; those artifacts were outside this second-pass scope, so those broader clean-scope claims are neither confirmed nor disputed here. The prior review’s zero-finding result is independently corroborated for the four target notes.
