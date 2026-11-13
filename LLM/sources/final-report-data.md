# Final gate report data

Generated from current LLM JSON, Markdown, scripts, and Git records.

## (a) Topics by week

### Week 1

| Topic ID | Topic | Ebook PDF pages | Routed occurrences |
|---|---|---:|---:|
| `w1-seq2seq-bottleneck` | Seq2seq and the RNN bottleneck | 4, 5 | 2 |
| `w1-bahdanau-attention` | Bahdanau (additive) attention | 5, 6 | 4 |
| `w1-self-attention-qkv` | Self-attention and Q/K/V projections | 6, 8 | 30 |
| `w1-scaled-dot-product` | Scaled dot-product attention and the sqrt(d_k) factor | 8, 9 | 14 |
| `w1-multi-head-attention` | Multi-head attention and W^O | 9, 18 | 17 |
| `w1-ffn-encoder-stack` | Position-wise FFN and the encoder layer/stack | 10, 11 | 8 |
| `w1-encoder-parameter-count` | Parameter counting for the encoder | 10, 20 | 14 |
### Week 2

| Topic ID | Topic | Ebook PDF pages | Routed occurrences |
|---|---|---:|---:|
| `w2-teacher-forcing` | Teacher forcing | 22, 22 | 8 |
| `w2-masked-self-attention` | Masked (causal) self-attention | 22, 24 | 15 |
| `w2-cross-attention` | Cross-attention (encoder-decoder attention) | 24, 24 | 5 |
| `w2-decoder-output-and-params` | Decoder output layer and decoder parameter counts | 24, 25 | 4 |
| `w2-positional-encoding` | Sinusoidal positional encoding | 25, 27 | 11 |
| `w2-residual-layernorm` | Residual connections, LayerNorm vs BatchNorm | 27, 29 | 21 |
| `w2-training-tricks` | Training tricks: label smoothing, warm-up schedule, dropout | 29, 30 | 1 |
| `w2-full-transformer` | The full Transformer data flow | 29, 29 | 4 |
### Week 3

| Topic ID | Topic | Ebook PDF pages | Routed occurrences |
|---|---|---:|---:|
| `w3-language-modelling` | Language-modelling objective, chain rule and perplexity | 35, 35 | 10 |
| `w3-model-parameterisations` | Ways to parameterise a language model | 35, 36 | 0 |
| `w3-causal-lm` | Causal language modelling with a decoder-only stack | 36, 37 | 8 |
| `w3-gpt1-architecture` | GPT-1 architecture and pre-train/fine-tune paradigm | 37, 38 | 4 |
| `w3-gpt1-parameter-count` | GPT-1 parameter count (117M) | 38, 39 | 7 |
| `w3-pretraining-objective` | Pre-training objective and teacher forcing | 39, 39 | 5 |
| `w3-finetuning` | Fine-tuning GPT and the fine-tuning loss | 39, 40 | 4 |
| `w3-task-input-formats` | Task-specific input transformations | 40, 40 | 1 |
### Week 4

| Topic ID | Topic | Ebook PDF pages | Routed occurrences |
|---|---|---:|---:|
| `w4-exhaustive-greedy` | Exhaustive and greedy search | 46, 47 | 15 |
| `w4-beam-search` | Beam search and length normalisation | 47, 48 | 13 |
| `w4-topk-temperature` | Top-k sampling and temperature | 48, 50 | 22 |
| `w4-topp-nucleus` | Top-p (nucleus) sampling and repetition issues | 47, 50 | 15 |
| `w4-bert-mlm` | BERT masked language modelling (80/10/10) | 51, 52 | 3 |
| `w4-bert-nsp-inputs` | NSP, [CLS]/[SEP] and the three input embeddings | 52, 53 | 6 |
| `w4-bert-architecture-params` | BERT sizes and parameter counts | 53, 54 | 2 |
| `w4-bert-finetuning` | BERT fine-tuning heads and GPT vs BERT | 54, 56 | 11 |

## (b) Routing and answer status by week

| Week | Routed | Distinct stems | MCQ | MSQ | SA | MCQ derived/ambiguous | MSQ derived/ambiguous | SA derived/ambiguous | Total derived/ambiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73 | 66 | 19 | 11 | 43 | 16/3 | 9/2 | 38/5 | 63/10 |
| 2 | 65 | 62 | 26 | 8 | 31 | 23/3 | 7/1 | 25/6 | 55/10 |
| 3 | 36 | 34 | 15 | 12 | 9 | 13/2 | 7/5 | 9/0 | 29/7 |
| 4 | 67 | 59 | 28 | 20 | 19 | 27/1 | 17/3 | 18/1 | 62/5 |

SA combines numerical and short-answer records. Answer-status counts are routed questions with a matching record in question-solutions.json; archived questions are excluded.

## (c) Archive count by likely_later_week

| Likely later week | Count |
|---|---:|
| 11 | 46 |
| 12 | 44 |
| 5 | 71 |
| 7 | 42 |
| 8 | 36 |
| other/unknown | 8 |

## (d) Ambiguous answer IDs and notes

### Week 1 (10)

- `da5004_2024T1_ET_AN--q151` — The pictured shared passage supplies no vocabulary size V, although token embeddings are requested. With bias-free attention, FFN biases and two affine norms per layer, the three encoder layers have 2,366,208 parameters and embeddings add 256V; other bias assumptions change even the constant. Fixed sinusoids add zero. The printed 2566 cannot establish a missing vocabulary size.
- `da5004_2024T1_ET_FN--q206` — The pictured shared passage supplies no vocabulary size V, although token embeddings are requested. With bias-free attention, FFN biases and two affine norms per layer, the three encoder layers have 2,366,208 parameters and embeddings add 256V; other bias assumptions change even the constant. Fixed sinusoids add zero. The printed 2566 cannot establish a missing vocabulary size.
- `da5004_2024T1_Q1_NA--q189` — The shared passage fixes vocabulary 100 and two heads but does not give model width. The preceding subquestion introduces 1600 embedding parameters only as a supposition: if carried forward, d=1600/100=16 and 3d²=768 weights; otherwise d remains free. Projection bias is also unspecified.
- `da5004_2025T1_Q1_NA--q112` — Implied course/exam convention: 3,072 Q/K/V matrix weights, or 3,168 including Q/K/V biases, if d=32. Carrying the preceding conditional source table to the common model gives d=3200/100=32; 3d²=3072 weights, plus 3d=96 biases if present. The course key implies shared context and weights only, but the earlier supposition is not expressly repeated and biases are not fixed.
- `da5004_2025T1_Q1_NA--q113` — Implied course/exam convention: 256 matrix weights if d=32; otherwise d²/4. If the preceding subquestion’s embedding condition applies, d=3200/100=32 and one of four heads has W_Q of size 32×8=256. Without carrying the conditional value forward the common passage leaves d unknown. The matrix itself has no separate bias; the course key implies carrying the width forward.
- `da5004_2025T2_Q1_NA--q147` — The original shared image has X=[[1,0],[0,2],[1,0],[0,2]] and W¹=[[1,1,1,1],[2,2,2,2]], with column-oriented tokens. Hence Q=K=V=[[2,4],[4,8]], and the displayed softmax(QᵀK/√2)Vᵀ is approximately [[4,8],[4,8]]. No option matches. The Markdown transcription is different.
- `da5004_2025T2_Q1_NA--q148` — From the pictured shared X and W², Q=K=V=[[2,0],[3,4]]. Row-softmax(QᵀK/√2)Vᵀ gives the stated matrix; no pictured option agrees. The Markdown transcription differs from the shared image.
- `da5004_2025T2_Q1_NA--q149` — Transpose the two head outputs into the image’s feature-by-token layout, concatenate and left-multiply diag(0.5,0.5,1,1). The resulting four-by-two matrix matches no pictured option; the shared-image matrices differ from the transcription.
- `da5004_2025T3_ET_AN--q17` — Implied course/exam convention: B and C when “self-attention” means the score and value products; C alone when projection FLOPs are included. The two attention matrix products scale as L²d and quadruple, while the positionwise FFN scales as L and doubles. Including the Q/K/V/output projections adds a linear-in-L term, making the exact total factor between 2 and 4. The usual textbook attention-complexity convention implies B and C; the old note correctly identifies the scope choice.
- `da5004_2025T3_ET_FN--q25` — Implied course/exam convention: B and C when “self-attention” means the score and value products; C alone when projection FLOPs are included. The two attention matrix products scale as L²d and quadruple, while the positionwise FFN scales as L and doubles. Including the Q/K/V/output projections adds a linear-in-L term, making the exact total factor between 2 and 4. The usual textbook attention-complexity convention implies B and C; the old note correctly identifies the scope choice.

### Week 2 (10)

- `da5004_2024T1_Q1_NA--q192` — Autoregressive factorization is compatible with teacher forcing, so the two named approaches are not mutually defined. A switch to generated-prefix training is possible (C), whereas an unconditional faster-convergence claim (B) depends on setup and convergence metric; the item does not specify these.
- `da5004_2024T3_Q1_NA--q276` — Multiple heads can capture different relationships (B); Transformer variants can use batch normalization (D). Both are defensible literal claims, although layer normalization is standard, and only one choice is allowed.
- `da5004_2024T3_Q1_NA--q277` — Implied course/exam convention: 230,656 with bias-free attention and affine LayerNorm; 232,192 with attention biases. The shared passage explicitly makes encoder and decoder configurations equal, so N=2 applies to both. Six attention blocks contribute 6×4×64²=98,304, four bias-free FFNs 4×2×64×256=131,072, and ten affine norms 1,280, total 230,656. Attention biases add 6×4×64=1,536. The printed answer favors bias-free attention; the old note incorrectly suggests independent stack depths are unspecified.
- `da5004_2024T3_Q1_NA--q278` — Implied course/exam convention: 48,000 weights; 48,750 including an output bias. A 64-to-750 output projection contains 64×750=48,000 weights; an output bias contributes 750. Whether tied output weights are counted as independent parameters is another convention. The paper key implies a weights-only count.
- `da5004_2024T3_Q1_NA--q279` — The shared encoder–decoder has source vocabulary 500, target vocabulary 750, and width 64, yet asks for “the embedding layer” without identifying source, target, or both. Their untied counts are 500×64, 750×64, and their sum; sharing would need further specification.
- `da5004_2025T1_Q1_NA--q111` — Implied course/exam convention: 0 for fixed sinusoidal positions; 1,024 for a learned table. The source embedding condition gives d=3200/100=32 and the context has 32 positions. A learned position table contains 32×32=1,024 parameters; fixed sinusoids contain zero. The original Transformer used sinusoids, while the paper key implies a learned table; “vanilla” alone does not settle this conflict.
- `da5004_2025T3_ET_FN--q2` — For one observation per normalized channel during training, the mean is that observation, not necessarily 0, and variance is 0; some implementations reject the case. Evaluation with tracked running statistics uses those statistics (D), but other spatial dimensions and evaluation modes are possible. The prompt specifies neither mode nor reduction axes.
- `da5004_2025T3_Q1_NA--q190` — Implied course/exam convention: 524,800 unmasked or finite-softmax positive entries; raw nonzero logits undetermined. The inclusive causal triangle contains 1024×1025/2=524,800 allowed positions. With finite logits, post-softmax allowed weights are positive and masked weights zero; pre-softmax raw allowed scores can be zero and additive-masked scores are −∞. The course convention counts available causal weights, but “score matrix” does not specify the stage.
- `da5004_2026T2_ET_AN--q3` — Sinusoidal dot products sum cosines of relative offsets and do not strictly increase with separation (A false). One-hot vectors have equal pairwise distances, but their active indices can be decoded and subtracted, so one-hot positions are not mathematically incapable of representing relative distance (D false). A single-choice question has two false statements under literal wording.
- `da5004_2026T2_ET_AN--q6` — Implied course/exam convention: 3 when ε=0; 1+2/√(1+ε) otherwise. The third pictured row is [7,5], with mean 6 and population variance 1. LayerNorm gives 2(7−6)/√(1+ε)+1. Elementary hand calculations conventionally omit ε and obtain 3; implementations use a small positive ε and the problem does not state it.

### Week 3 (7)

- `da5004_2024T1_ET_AN--q149` — Implied course/exam convention: A, B, C, D with a fixed complete-sequence window; A–E with sliding and reset position IDs. The shared prompt gives T=2048 and 712 prompt tokens, leaving 1336 slots under a complete-sequence cap, so 256, 16, 1024 and 512 fit. A sliding context can generate 2048; the supplied prompt does not specify inference policy. The exam key implies the fixed-window reading.
- `da5004_2024T1_ET_FN--q202` — Implied course/exam convention: A, B with a fixed complete-sequence window; A–D with sliding and reset position IDs. The shared prompt gives T=1024 and 712 prompt tokens, leaving 312 slots under a complete-sequence cap. Only 256 and 16 fit. Sliding can generate all listed lengths; the exam key implies the fixed-window reading.
- `da5004_2024T1_Q1_NA--q193` — Implied course/exam convention: A if BOS counts within a 1024-token full sequence; A, B if 1024 newly predicted tokens are counted; A–D with sliding. A capped full sequence leaves 1023 positions after BOS; a final prediction can instead be emitted from a full input context, allowing 1024 new tokens. Sliding or resetting positional IDs permits still longer text. The exam’s fixed-window treatment is the closest course convention; its wording does not settle output counting.
- `da5004_2024T1_Q1_NA--q194` — Implied course/exam convention: A, B, D for the original GPT-1 supervised experiments; A, D for ordinary supervised fine-tuning without the optional auxiliary CLM loss. GPT-1 pretraining starts from random weights; supervised fine-tuning reuses them and typically uses labels. Radford et al. included an auxiliary language-model objective in their experiments, while fine-tuning in general need not. The paper-specific and generic conventions differ, and the course paper implies the GPT-1 experiment.
- `da5004_2025T1_Q1_NA--q108` — Implied course/exam convention: A, B, D for the original GPT-1 supervised experiments; A, D for ordinary supervised fine-tuning without the optional auxiliary CLM loss. GPT-1 pretraining starts from random weights; supervised fine-tuning reuses them and typically uses labels. Radford et al. included an auxiliary language-model objective in their experiments, while fine-tuning in general need not. The paper-specific and generic conventions differ, and the course paper implies the GPT-1 experiment.
- `da5004_2026T1_Q1_NA--q18` — Implied course/exam convention: D: 147,456 weights; 147,648 including Q/K/V biases. With width 768 and 12 heads, each head width is 64. The three 768×64 projection matrices contain 147,456 weights; three bias vectors add 192. D and the course-style matrix count imply bias-free projections, whereas the track’s GPT parameter convention includes biases.
- `da5004_2026T2_Q1_NA--q11` — The pictured partial tree gives A=.5×.3×.4=.06, C=.5×.3×.6=.09, D=.4×.4×.5=.08, and E=.4×.4×.4=.064. B starts with an unpictured apples branch whose mass may be as high as .1; its continuation probabilities are not given. Both C and B can win.

### Week 4 (5)

- `da5004_2024T1_Q1_NA--q195` — Top-k, top-p, beam and exhaustive search can all generate translations. If appropriate means the conventional mode-seeking objective, B and D are the course key; if it means usable, A–D qualify. No criterion for appropriateness is given, so a unique set cannot be established.
- `da5004_2024T3_ET_AN--q178` — With a fixed model and deterministic tie-breaking both beam search with width four and greedy decoding return the same completion. The item is single-choice but has two correct choices.
- `da5004_2025T1_Q1_NA--q110` — Top-k, top-p, beam and exhaustive search can all generate translations. If appropriate means the conventional mode-seeking objective, B and D are the course key; if it means usable, A–D qualify. No criterion for appropriateness is given, so a unique set cannot be established.
- `da5004_2025T2_ET_FN--q141` — Deterministic greedy is consistent with repeated stories, and top-p sampling can produce variation; a fixed model and unspecified stochastic seed, sample size, probability distribution and priors do not determine the “most likely” mappings. Top-k can repeat by chance and p=0.5 can even yield a one-token nucleus despite T=1.5.
- `da5004_2026T2_Q1_NA--q15` — Implied course/exam convention: 400 unbatched prefix evaluations; 4 batched depth-wise forward passes. General exhaustive decoding evaluates 1+7+7²+7³=400 prefixes to assign probabilities to four tokens. If “decoder run” means batched forward pass, four depth-wise calls suffice; if the supplied table is reusable and prefix-invariant, the computation needs no additional model calls. The usual course search-tree convention implies 400 but the unit is not defined.

## (e) Evidence and overrides

- Ebook discrepancies logged: 21 rows.
- Review files: review-luna-notes-w12.md, review-luna-notes-w34.md, review-luna-routing.md, review-luna-second-pass-w12.md, review-terra-ambiguous.md, review-terra-figures-banks.md.
- Review findings by reviewer/severity (counts use explicit review summaries; routing confidence findings are shown separately):
  - luna / explicit-zero-finding-reviews: 2
  - luna / high-confidence-findings: 1
  - luna / medium-confidence-findings: 1
  - terra / error: 6
  - terra / unrated-ambiguity-cases-reviewed: 62
  - terra / warning: 6
- Routing overrides/corrections applied: 2 records from `routing-overrides.json`.
- Solutions overrides/corrections applied: 62 records from `solutions-overrides.json`.
- Corrections overrides/corrections applied: 30 records from `question-corrections.json`.

## (f) Commits on branch since scaffold

```text
7b62e7a docs(llm): second-pass audit of week 1-2 notes
508ddb5 fix(llm): concept definition lengths and retag consistency
6c4075c fix(llm): apply review overrides and rebuild banks
e48b444 fix(llm): review overrides and conventions
2bd8b3f docs(llm): audit of ambiguous answers
9496e2b docs(llm): routing review
c63c550 docs(llm): review of figures and banks
9d6e958 docs(llm): review of week 1-2 notes
56d28b3 docs(llm): review of week 3-4 notes
20e47f8 feat(llm): exam question banks, index and study guide
04fbf19 feat(llm): link and notes quality checkers
d406507 feat(llm): verified and merged question solutions
ca2a320 fix(llm): adjudicate numeric answer mismatches
ebb0fda fix(llm): expand Terra numeric verification
2d58d70 docs(llm): knowledge map and track conventions
1cd4777 feat(llm): derived answers W1 chunk A (Terra)
5bb01e5 feat(llm): derived answers W2 chunk A (Terra)
8473335 docs(llm): week 4 notes and exam-focused notes
76c32f7 feat(llm): derived answers W2 chunk B (Terra)
a7442c7 feat(llm): derived answers weeks 3-4 (Terra)
de2cfb7 feat(llm): derived answers W1 chunk B (Terra)
0712ab2 docs(llm): week 3 notes and exam-focused notes
e7806ba docs(llm): week 2 notes and exam-focused notes
0ab51a6 feat(llm): numeric answers with code verification (Luna)
5a5772a docs(llm): week 1 notes and exam-focused notes
373d794 feat(llm): merge and validate final routing
15ff529 feat(llm): week 4 figures and executed code blocks
5241e46 feat(llm): final routing part 1
8d59388 feat(llm): week 3 figures and executed code blocks
ea69852 feat(llm): final routing part 2
e18c31d feat(llm): final routing part 3
2d71d9b feat(llm): final routing part 4
3d9ca21 feat(llm): week 2 figures and executed code blocks
6f0c17f feat(llm): week 1 figures and executed code blocks
cd847aa feat(llm): first-pass question pre-router
d52a20d docs(llm): week 4 concept notes
2147e60 docs(llm): week 3 concept notes
84b3d30 docs(llm): week 2 concept notes
8e578fb docs(llm): week 1 concept notes
57d8741 docs(llm): gate A topics, transcript check, discrepancy log
26fe5d5 docs(llm): ebook transcript week 4
37f49d2 docs(llm): ebook transcript week 1
0cc9baa docs(llm): ebook transcript week 2
f051291 docs(llm): ebook transcript week 3
c6fe544 feat(llm): exam paper question inventory
```

## (g) Exact rebuild commands

```bash
cd LLM
uv run python scripts/build_inventory.py
uv run python scripts/merge_routing.py
uv run python scripts/merge_solutions.py
uv run python scripts/build_exam_question_banks.py
uv run python scripts/verify_exam_banks.py
uv run python scripts/check_links.py
uv run python scripts/check_notes_quality.py
for n in 1 2 3 4; do uv run python generate_week${n}_plots.py; done
uv run python scripts/run_gates.py
```
