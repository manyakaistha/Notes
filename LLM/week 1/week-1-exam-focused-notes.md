---
title: "Week 1: Attention and Encoder Exam Guide"
week: 1
status: complete
date: 2026-11-13
tags: [llm/week-01, llm/exam, llm/attention]
prev: "[[Knowledge-Map|Knowledge Map]]"
next: "[[week-2-notes|Week 2 Notes]]"
notes: "[[week-1-notes|Week 1 Notes]]"
map: "[[Knowledge-Map]]"
---

# Week 1: Attention and Encoder Exam Guide

**Study first:** [[week-1-notes|Week 1 master note]] · **Authentic questions:** [[week-1-exam-questions|Week 1 question bank]]

**Navigation:** [[Knowledge-Map|Knowledge Map]] · [[week-2-notes|Week 2 Notes]] →

## Priority from routed questions

Counts below are occurrences of topic IDs among the 73 records with `week == 1` in `routing.json`. A question can count for more than one topic. This is source coverage, not a prediction of future exam frequency.

| Priority | Canonical topic | Routed records |
|---:|---|---:|
| 1 | Self-attention and Q/K/V projections | 30 |
| 2 | Multi-head attention and W^O | 17 |
| 3 | Parameter counting for the encoder | 14 |
| 4 | Scaled dot-product attention and the sqrt(d_k) factor | 14 |
| 5 | Position-wise FFN and the encoder layer/stack | 8 |
| 6 | Bahdanau (additive) attention | 4 |
| 7 | Seq2seq and the RNN bottleneck | 2 |

## Formula sheet

Let $d=d_{\rm model}$, $f=d_{\rm ff}$, and $h$ be the number of heads. Use row-oriented matrices throughout.

| Item | Formula and dimensions |
|---|---|
| Fixed-context seq2seq | $p(y\mid x)=\prod_t p(y_t\mid y_{<t},c)$; $c$ must carry source information needed by all output steps. |
| Additive attention | $e_{ti}=v_a^\top\tanh(W_as_{t-1}+U_ah_i)$; $\alpha_{ti}=e^{e_{ti}}/\sum_j e^{e_{tj}}$; $c_t=\sum_i\alpha_{ti}h_i$. |
| Projections | $Q=XW^Q$, $K=XW^K$, $V=XW^V$. For `X [B,T,d]`, $Q,K$ are `[B,T,d_k]`, $V$ is `[B,T,d_v]`. |
| Attention | $A=\operatorname{softmax}_{\rm keys}(QK^\top/\sqrt{d_k}+M)$; $Z=AV$. Shapes: $A$ `[B,T_q,T_k]`, $Z$ `[B,T_q,d_v]`. |
| Scaling | Under independent zero-mean unit-variance coordinates, $\operatorname{Var}(q\cdot k)=d_k$, so the scaled variance is 1. |
| Heads | $H_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)$; $\operatorname{MHA}(X)=\operatorname{Concat}(H_i)W^O$. |
| Standard head widths | $d_k=d_v=d/h$; concat `[B,T,d]`; $W^O$ `[d,d]`. |
| FFN | $\operatorname{ReLU}(xW_1+b_1)W_2+b_2$; $W_1$ `[d,f]`, $W_2$ `[f,d]`. |
| Original encoder layer | $U=\operatorname{LN}_1(X+\operatorname{MHA}(X))$, $Y=\operatorname{LN}_2(U+\operatorname{FFN}(U))$; dropout omitted. |
| Parameters | Weights only $4d^2+2df$; FFN biases add $f+d$; four attention biases add $4d$; two affine norms add $4d$. |
| Score entries | $BhT^2$ for dense self-attention. Weights do not depend on $T$; a learned positional embedding table can. |

## How it is asked

- **Trace an attention calculation:** form Q/K/V, score the designated query against each key, apply scaling if specified, softmax across that query's keys, combine the corresponding values, then apply $W^O$ if requested. Examples: [[week-1-exam-questions#da5004_2025T2_ET_AN--q199|2025 T2 ET AN Q199]], [[week-1-exam-questions#da5004_2025T2_Q1_NA--q146|2025 T2 Q1 Q146]], [[week-1-exam-questions#da5004_2026T1_Q1_NA--q13|2026 T1 Q1 Q13]]–[[week-1-exam-questions#da5004_2026T1_Q1_NA--q15|Q15]]. The latter three test additive score, normalized weight, and context in sequence.
- **Infer shape or score volume:** multiply compatible dimensions and distinguish the output tensor from the score tensor. Examples: [[week-1-exam-questions#da5004_2026T1_Q1_NA--q2|2026 T1 Q1 Q2]], [[week-1-exam-questions#da5004_2025T3_Q1_NA--q186|2025 T3 Q1 Q186]], [[week-1-exam-questions#da5004_2026T2_Q1_NA--q6|2026 T2 Q1 Q6]].
- **Count specified weights:** decide whether $W^O$, biases, LayerNorm, embeddings and position tables are included; sum only requested terms. Examples: [[week-1-exam-questions#da5004_2025T1_Q1_NA--q112|2025 T1 Q1 Q112]], [[week-1-exam-questions#da5004_2026T1_ET_FN--q2|2026 T1 ET FN Q2]], [[week-1-exam-questions#da5004_2026T1_ET_FN--q3|Q3]].
- **Explain a mechanism:** identify the fixed-context bottleneck, additive alignment, scaling or components of an encoder. Examples: [[week-1-exam-questions#da5004_2025T1_Q1_NA--q107|2025 T1 Q1 Q107]], [[week-1-exam-questions#da5004_2026T2_Q1_NA--q2|2026 T2 Q1 Q2]], [[week-1-exam-questions#da5004_2025T3_Q1_NA--q173|2025 T3 Q1 Q173]], [[week-1-exam-questions#da5004_2026T1_Q2_NA--q8|2026 T1 Q2 Q8]].
- **Reason about length or heads:** compare $T^2$ score storage with position-wise $T$ work, or evaluate head-count changes at fixed width. Examples: [[week-1-exam-questions#da5004_2025T3_ET_AN--q17|2025 T3 ET AN Q17]], [[week-1-exam-questions#da5004_2026T2_Q1_NA--q5|2026 T2 Q1 Q5]]. Dense score formation and value aggregation both scale quadratically in sequence length; see [[week-1-exam-questions#da5004_2025T3_ET_FN--q8|2025 T3 ET FN Q8]] for a question with potentially ambiguous single-choice wording.

## Traps and common mistakes

**The Dangerous Misconception:** Every seq2seq model uses two RNNs and one fixed vector. **Why this is false:** The fixed-context RNN is one historical sequence-to-sequence architecture; attention-based and Transformer encoder–decoders are also seq2seq ([[seq2seq-bottleneck]]).

**The Dangerous Misconception:** Attention weights are computed from values, and softmax runs over queries. **Why this is false:** Queries and keys form scores; each query's softmax runs over eligible keys; those weights combine values ([[self-attention]]).

**The Dangerous Misconception:** Scaling guarantees no saturation. **Why this is false:** It controls variance only under stated assumptions; learned logits can still be extreme ([[scaled-dot-product-attention]]).

**The Dangerous Misconception:** The number of heads always increases the projection weight count. **Why this is false:** At fixed model width with standard widths $d/h$, total weights stay $4d^2$ including $W^O$. A single-head layer may also have $W^O$ ([[multi-head-attention]]).

**The Dangerous Misconception:** FFN and attention mix the same axis. **Why this is false:** Attention mixes values across positions; the shared position-wise FFN transforms features separately at each position ([[position-wise-feed-forward]]).

**The Dangerous Misconception:** “Identical encoder layers” share weights. **Why this is false:** The original stack has the same architecture at each depth but separately learned parameters. Its residual addition is followed by LayerNorm; Pre-LN is a different arrangement ([[encoder-layer-parameter-count]]).

**The Dangerous Misconception:** A longer sequence adds layer projection parameters. **Why this is false:** Shared projections do not depend on $T$; activation size and computation do. An explicitly learned positional table is a separate possible exception.

## Self-check: ten questions with worked answers

1. **Why is a fixed context restrictive?**  
   **Answer:** The decoder receives one fixed-width source-dependent representation $c$ at every step. It must preserve information useful for all output positions. Additive attention instead computes $c_t$ from all encoder states with weights depending on decoder state; the recurrent decoder is still sequential.

2. **For scores $[0,0]$ and encoder states $h_1=[2,0]$, $h_2=[0,4]$, find the additive-attention context.**  
   **Answer:** Softmax is $[1/2,1/2]$ because $e^0=e^0=1$. Thus $c=\frac12[2,0]+\frac12[0,4]=[1,2]$.

3. **An input has shape `[3, 7, 12]`; a query projection has shape `[12, 4]`. What are the query and self-attention score shapes?**  
   **Answer:** $Q=XW^Q$ is `[3, 7, 4]`. A matching key has `[3, 7, 4]`, so $QK^\top$ is `[3, 7, 7]`.

4. **Why use $\sqrt{d_k}$?**  
   **Answer:** With independent zero-mean unit-variance components, the sum of $d_k$ independent products has variance $d_k$. Division by $\sqrt{d_k}$ divides variance by $d_k$, giving 1 under these assumptions and mitigating excessively large logits.

5. **For $d_k=1$, $q=1$, $k_1=k_2=0$, $v_1=[1,0]$, $v_2=[0,2]$, compute attention.**  
   **Answer:** Both scaled scores equal 0; softmax is $[1/2,1/2]$. The output is $[1/2,1]$. Values, rather than keys, are aggregated.

6. **Let $d=512$ and $h=8$ with standard widths. Find a head width and the concatenated width.**  
   **Answer:** $d_k=d_v=512/8=64$. Eight outputs of width 64 concatenate to $8\cdot64=512$; $W^O$ is `[512, 512]` in this configuration.

7. **What does $W^O$ do to concatenated $[1,2,3,4]$ if its two columns are $[1,0,1,0]^\top$ and $[0,1,0,1]^\top$?**  
   **Answer:** The row-by-matrix product gives $[1+3,2+4]=[4,6]$. This linear map can mix features from different heads.

8. **Count weight matrices only for standard MHA with $d=256$.**  
   **Answer:** Each of Q, K, V and output has $d^2$ total weights, so $4d^2=4\cdot256^2=262{,}144$. This excludes all biases and FFN weights.

9. **Count FFN weights and biases for $d=4$, $f=8$.**  
   **Answer:** $W_1$ and $W_2$ have $4\cdot8+8\cdot4=64$ weights; biases have $8+4=12$ entries. Total $64+12=76$. The same parameters are reused at every token position.

10. **For $B=4$, $h=2$, $T=12$, count dense score entries and find the result when $T$ doubles.**  
    **Answer:** $BhT^2=4\cdot2\cdot12^2=1{,}152$. At $T=24$ there are $4\cdot2\cdot24^2=4{,}608$, four times as many; shared projection parameters are unchanged.

The textbook's quick mnemonic “MHA is 1, FFN is 2, layer is 3” gives rough millions for the base encoder. For exact numbers and the ebook's corrected matrix orientations, see [[week-1-notes|the master note]] and its executed examples.
