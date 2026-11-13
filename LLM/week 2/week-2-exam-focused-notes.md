---
title: "Week 2: Transformer Decoder Exam-Focused Notes"
week: 2
status: complete
date: 2026-11-13
tags: [llm/week-02, llm/exam, llm/transformer]
concepts:
  - "[[teacher-forcing]]"
  - "[[masked-self-attention]]"
  - "[[cross-attention]]"
  - "[[sinusoidal-positional-encoding]]"
  - "[[layer-normalization]]"
  - "[[residual-connections]]"
  - "[[label-smoothing]]"
  - "[[learning-rate-warmup]]"
prev: "[[week-1-exam-focused-notes|Week 1 Exam Guide]]"
next: "[[week-3-exam-focused-notes|Week 3 Exam Guide]]"
map: "[[Knowledge-Map]]"
---

# Week 2: Transformer Decoder Exam-Focused Notes

**Study:** [[week-2-notes|Full Week 2 notes]] · [[week-2-exam-questions|Routed questions]] · [[Knowledge-Map|Knowledge Map]]

## 1. What to study more, and why

Counts below are numbers of routing records with each topic ID, not independent exams or a probability forecast. A record assigned to several topics contributes once to each topic. Routing is an editorial classification; it is not an official syllabus weighting.

| Priority by frequency | Canonical topic | Routed records |
|---:|---|---:|
| 1 | Residual connections, LayerNorm vs BatchNorm | 21 |
| 2 | Masked (causal) self-attention | 15 |
| 3 | Sinusoidal positional encoding | 11 |
| 4 | Teacher forcing | 8 |
| 5 | Cross-attention (encoder-decoder attention) | 5 |
| 6 | Decoder output layer and decoder parameter counts | 4 |
| 7 | The full Transformer data flow | 4 |
| 8 | Training tricks: label smoothing, warm-up schedule, dropout | 1 |

## 2. How the actual papers ask it

- **Position arithmetic:** [[week-2-exam-questions#da5004_2024T1_Q1_NA--q184|2024 T1 Q1 Q184]] asks for the sum of a positional vector from a shared setup. Write the pair indices and evaluate the given $d$ and position; check whether positions start at zero. [[week-2-exam-questions#da5004_2026T1_Q1_NA--q12|2026 T1 Q1 Q12]] has an ambiguous index in the prompt, so distinguish coordinate index from pair index.
- **Cross-attention dimensions:** [[week-2-exam-questions#da5004_2024T1_Q1_NA--q190|2024 T1 Q1 Q190]] gives source length $T_1$ and decoder length $T_2$: each head's attention matrix is `[T2, T1]`, because queries determine rows and keys columns.
- **Gold versus generated prefix:** [[week-2-exam-questions#da5004_2025T3_Q1_NA--q175|2025 T3 Q1 Q175]] asks for the input at the third prediction after an incorrect second prediction. Teacher forcing supplies the gold previous token, “suis,” not “content.”
- **Mask a score row:** [[week-2-exam-questions#da5004_2025T3_Q1_NA--q176|2025 T3 Q1 Q176]] gives row `[1, 2, 0, 2]` for token 3; the masked row is `[1, 2, 0, -∞]`, before softmax.
- **Accounting and normalization:** [[week-2-exam-questions#da5004_2024T3_Q1_NA--q277|2024 T3 Q1 Q277]] specifies a two-layer model and excludes embedding/output and FFN biases; follow those exclusions instead of substituting a library total. [[week-2-exam-questions#da5004_2024T3_Q1_NA--q281|2024 T3 Q1 Q281]] asks for a BatchNorm row sum: compute the specified feature-wise statistics first; row sums are not necessarily zero.
- **Architecture choices:** [[week-2-exam-questions#da5004_2025T2_ET_FN--q142|2025 T2 ET FN Q142]] asks about valid model design choices. A training or architectural choice can vary; do not treat one original-model setting as the only possible Transformer.

## 3. Formula and shape sheet

| Quantity | Expression or shape |
|---|---|
| Sequence likelihood | $p(y\mid x)=\prod_t p(y_t\mid y_{<t},x)$ |
| Causal score mask | $M_{ij}=0$ for $j\le i$, $-\infty$ for $j>i$; softmax over keys after addition |
| Cross-attention | $Q$ from decoder `[B,T,d_k]`, $K,V$ from encoder `[B,S,d_k]`, `[B,S,d_v]`; per-head weights `[B,T,S]` |
| Sinusoidal PE | $PE(p,2i)=\sin(p\omega_i)$, $PE(p,2i+1)=\cos(p\omega_i)$, $\omega_i=10000^{-2i/d}$ |
| Offset property | Each sine–cosine pair rotates linearly by $k\omega_i$ at offset $k$; $\lVert PE(p)\rVert^2=d/2$ |
| LayerNorm | $\mu=\operatorname{mean}_i x_i$, $\sigma^2=\operatorname{mean}_i(x_i-\mu)^2$, $y_i=\gamma_i(x_i-\mu)/\sqrt{\sigma^2+\epsilon}+\beta_i$ |
| Post-LN | $\operatorname{LN}(x+F(x))$; original decoder sublayers: masked self-attention, cross-attention, FFN |
| FFN and parameters | $\operatorname{ReLU}(xW_1+b_1)W_2+b_2$; biased count $2df+f+d$ |
| Biased decoder layer | $2(4d^2+4d)+2df+f+d+6d$; assumes two full-width attention modules and three affine norms |
| Output | `[B,T,d]` × `[d,V]` → `[B,T,V]`; matrix count $dV$, output bias if present $V$ |
| Smoothing | $q_i=(1-\varepsilon)\mathbf1[i=y]+\varepsilon/V$, loss $-\sum_iq_i\log p_i$ |
| Warmup | $d^{-1/2}\min(s^{-1/2},sw^{-3/2})$, peak at $s=w$ |

## 4. Traps and decision rules

**The Dangerous Misconception:** A mask removes the diagonal. **Why this is false:** With shifted-right targets the diagonal is the current input, which is already a previous target; only future input positions are blocked.

**The Dangerous Misconception:** The source must be causally masked in ordinary cross-attention. **Why this is false:** The full source is already known; exclude padded source keys and apply other restrictions only if the task requires them.

**The Dangerous Misconception:** BatchNorm at batch size one always returns its input. **Why this is false:** Statistics may include sequence positions; with just one observation the normalized value is zero, and some implementations raise an error. Evaluation using running statistics is a different case. The ebook's PDF p. 28 claim is incorrect.

**The Dangerous Misconception:** Sinusoidal PE guarantees successful extrapolation or monotonically increasing distance. **Why this is false:** It is computable at new positions and has a linear fixed-offset map, while distance oscillates and model performance remains empirical.

**The Dangerous Misconception:** BERT-large was a pre-LN model. **Why this is false:** Original BERT uses post-residual LayerNorm; the ebook's PDF p. 29 example is incorrect. Count affine parameters and biases only when the question includes them, and count tied weights once.

## 5. Closed-book self-check: exactly ten worked questions

1. **Teacher forcing.** Gold target is `[A,B,EOS]`; at step 2 the model incorrectly predicts `C`. What are the decoder input sequence during training and the input for prediction 3?
   **Answer:** Shift right to `[BOS,A,B]`; prediction 3 receives `B`, the gold previous token. At free-running inference it could instead receive `C`.

2. **Mask and normalize.** Scores are `[2,1,5,3]`; only the first two keys are visible. Find the weights.
   **Answer:** Mask to `[2,1,-∞,-∞]`. Divide `[e²,e,0,0]` by $e²+e$ to get approximately `[0.7311,0.2689,0,0]`.

3. **Cross-attention shape.** For `[B,T,S,d,h]=[2,3,5,16,4]` with equal head widths, give per-head $Q,K,V$ and all-head weight shapes.
   **Answer:** $d_k=d_v=16/4=4$; $Q$ is `[2,3,4]`, $K,V$ are `[2,5,4]`, and weights are `[2,4,3,5]`; output after concatenation and projection is `[2,3,16]`.

4. **PE at zero.** Compute a four-dimensional sinusoidal encoding at position zero and its squared norm.
   **Answer:** $[\sin0,\cos0,\sin0,\cos0]=[0,1,0,1]$, with squared norm 2. In general it is $d/2$.

5. **Relative offset.** Show the linear map of a pair $[\sin(p\omega),\cos(p\omega)]^\top$ for offset $k$.
   **Answer:** Angle addition yields $\left[\begin{smallmatrix}\cos(k\omega)&\sin(k\omega)\\-\sin(k\omega)&\cos(k\omega)\end{smallmatrix}\right]$ times the original pair. The entries depend on offset and frequency, not on $p$.

6. **LayerNorm.** Normalize `[1,2,3]` with $\gamma=1$, $\beta=0$, neglecting $\epsilon$.
   **Answer:** Mean $=2$, variance $=(1+0+1)/3=2/3$; output is $[-\sqrt{3/2},0,\sqrt{3/2}]\approx[-1.2247,0,1.2247]$.

7. **Layer parameters.** For $d=512,f=2048$, two biased attention modules and three affine norms, find the decoder-layer total.
   **Answer:** Attention: $2(4\cdot512^2+4\cdot512)=2,101,248$. FFN: $2\cdot512\cdot2048+2048+512=2,099,712$. Norms: $6\cdot512=3,072$. Total $4,204,032$.

8. **Output projection.** For $d=512,V=37,000$, count untied output weights with and without bias.
   **Answer:** Matrix $512\cdot37,000=18,944,000$; a vocabulary bias adds 37,000, yielding 18,981,000. Weight tying changes the number of distinct parameters.

9. **Label smoothing.** For $V=5$, gold class 2 and $\varepsilon=0.1$ using the uniform-mixture convention, find $q$.
   **Answer:** Each class gets $0.1/5=0.02$ and the gold gets an extra $0.9$: `[0.02,0.02,0.92,0.02,0.02]`. This sums to 1.

10. **Warmup.** For $d=512,w=4000$, compare steps 1000, 4000 and 8000.
    **Answer:** The first lies on the linear branch and is one quarter of the peak. The peak is $512^{-1/2}4000^{-1/2}\approx0.0006987712$; step 1000 is $0.0001746928$. Step 8000 is the peak divided by $\sqrt2$, approximately $0.0004941059$.

## 6. Last-pass readiness

Trace source and shifted target separately; identify queries and keys before multiplying dimensions; mask before softmax; state normalization axes and the smoothing convention; and enumerate shared weights, biases and layer counts before arithmetic. See [[week-2-notes|full notes]] for derivations, exact executed examples, figures and paper references, and [[week-3-exam-focused-notes|Week 3 exam guide]] for the next step.
