---
title: "Week 3: Learn Language Modelling and Prepare for the Exam"
week: 3
date: 2026-11-13
status: complete
tags: [llm/week-03, llm/exam-study]
map: "[[Knowledge-Map]]"
parent: "[[week-3-notes]]"
questions: "[[week-3-exam-questions]]"
concepts: ["[[language-modelling-objective]]", "[[perplexity]]", "[[causal-language-modelling]]", "[[gpt-1-architecture]]", "[[pretrain-finetune-paradigm]]", "[[task-specific-input-transformations]]"]
---

# Week 3 — Learn Language Modelling and Prepare for the Exam

**Navigation:** [[week-3-notes|Full Week 3 notes]] · [[week-3-exam-questions|Routed questions]] · [[Knowledge-Map|Knowledge Map]] · [[week-4-notes|Week 4]]

## 1. What to study more, and why

**Practise conditional probability trees, causal masking and parameter counts first.** The routing manifest has **36 question occurrences**, with multiple topic tags on some questions. Counts are tag occurrences, not unique templates or a prediction of future marks. Repeated or related papers need not be independent.

| Priority | Canonical topic | Routed question occurrences |
| :--- | :--- | ---: |
| P1 | Language-modelling objective, chain rule and perplexity | 10 |
| P1 | Causal language modelling with a decoder-only stack | 8 |
| P1 | GPT-1 parameter count (117M) | 7 |
| P2 | Pre-training objective and teacher forcing | 5 |
| P2 | GPT-1 architecture and pre-train/fine-tune paradigm | 4 |
| P2 | Fine-tuning GPT and the fine-tuning loss | 4 |
| P3 | Task-specific input transformations | 1 |
| P3 | Ways to parameterise a language model | 0 |

**Checklist context:** No question is routed directly to model parameterisations; understand n-grams, RNNs and Transformers anyway. Perplexity is part of the canonical language-modelling topic but is not defined in the ebook; the count for that topic does not imply every question tests perplexity.

## 2. Formula sheet and meaning

| Quantity | Formula and use |
| :--- | :--- |
| Joint probability | $p(w_{1:T})=\prod_i p(w_i\mid w_{<i})$; specify the start context and termination. |
| Sequence log-likelihood | $\log p(w_{1:T})=\sum_i\log p(w_i\mid w_{<i})$. |
| Token cross-entropy | $-\sum_v y_{iv}\log p_{iv}=-\log p_i(w_i)$ for one-hot $y_i$. |
| Perplexity | $\exp[-N^{-1}\sum_i\log p(w_i\mid w_{<i})]$, natural log; compare matching tokens and protocols. |
| Causal attention | $\operatorname{softmax}(QK^\top/\sqrt{d_k}+M)V$, with $M_{jk}=-\infty$ if $k>j$ and zero otherwise. |
| Tied logits | $H E^\top$: `[B, T, d]` times `[d, V]` gives `[B, T, V]`. |
| Embedding parameters | $(V+T_{\max})d$ for token plus learned position tables. |
| Attention per block | $4d^2+4d$ with four projection biases; omit bias terms when the question asks weights only. |
| FFN per block | $2df+f+d$ including both biases, for widths $d,f$. |
| Two affine LayerNorms | $4d$ parameters per block. |
| GPT-1 transfer | Maximize $L_2+\lambda L_1$, where $L_2$ is task log-likelihood and $L_1$ is LM log-likelihood; equivalently minimize their negatives. |

GPT-1's worked convention uses $V=40{,}478$, $T_{\max}=512$, $d=768$, $f=3072$ and 12 blocks. Its count is 116,534,784 with tied embeddings; the paper's 40,000 BPE merges do not equal the final vocabulary size. A classification head is additional to this base count. See [[gpt-1-architecture]].

## 3. How it is asked

1. **Probability tree and sequence product.** [[week-3-exam-questions#da5004_2024T3_Q1_NA--q286|2024 T3 Q1 Q286]] and [[week-3-exam-questions#da5004_2025T1_Q1_NA--q119|2025 T1 Q1 Q119]] ask for a joint probability, sometimes scaled by 100 or 1000. Follow the actual branch at each prefix, multiply, then apply the requested scale. [[week-3-exam-questions#da5004_2026T2_Q1_NA--q12|2026 T2 Q1 Q12]] tests whether the given tree supplies enough information for the requested conditional; an unprovided conditional cannot be invented.
2. **Mask and finite context.** [[week-3-exam-questions#da5004_2026T1_Q2_NA--q13|2026 T1 Q2 Q13]] asks whether an attention matrix is causal. Inspect nonzero future entries row by row. [[week-3-exam-questions#da5004_2024T1_Q1_NA--q193|2024 T1 Q1 Q193]] asks about generation length for a model with a stated context; distinguish the specified fixed-window protocol from implementations that slide or crop the window. Do not claim an architecture-independent hard limit on total generated text.
3. **Components and transfer.** [[week-3-exam-questions#da5004_2026T2_Q1_NA--q13|2026 T2 Q1 Q13]] tests decoder-layer components; masked self-attention and FFN belong, encoder cross-attention does not. [[week-3-exam-questions#da5004_2025T1_Q1_NA--q108|2025 T1 Q1 Q108]] combines GPT architecture and fine-tuning. Evaluate each option independently in an MSQ.
4. **Counting with explicit exclusions.** [[week-3-exam-questions#da5004_2026T1_Q1_NA--q16|2026 T1 Q1 Q16]] through [[week-3-exam-questions#da5004_2026T1_Q1_NA--q20|Q20]] separate token table, position table, one head's Q/K/V, output projection and biased FFN. Use the dimensions in that question rather than memorized GPT-1 numbers. [[week-3-exam-questions#da5004_2024T1_Q2_NA--q135|2024 T1 Q2 Q135]] excludes embedding and output layers: changing vocabulary alone then does not change the counted decoder blocks if their dimensions stay fixed.
5. **Training and gradients.** [[week-3-exam-questions#da5004_2025T3_ET_AN--q24|2025 T3 ET AN Q24]] connects teacher forcing to all shifted target losses. [[week-3-exam-questions#da5004_2024T3_ET_AN--q177|2024 T3 ET AN Q177]] asks about rare vocabulary entries under tied weights: even tokens absent from the input batch can receive output-softmax gradients because they are in the denominator, assuming ordinary full softmax and nonzero activations. Tying additionally shares those updates with input embeddings.
6. **Task interface.** [[week-3-exam-questions#da5004_2026T2_Q2_NA--q8|2026 T2 Q2 Q8]] asks why entailment uses a delimiter; it separates premise from hypothesis. [[week-3-exam-questions#da5004_2026T2_Q2_NA--q9|2026 T2 Q2 Q9]] asks why the final hidden state is used; it can attend to the complete preceding sequence under the causal mask.

## 4. Traps and quick decisions

- **The Dangerous Misconception:** The chain rule requires independence. **Why this is false:** It holds with conditional dependence; independence replaces conditionals with marginals.
- **The Dangerous Misconception:** A causal mask alone prevents copying the current target. **Why this is false:** The current input is available through attention and residuals; shift input and target.
- **The Dangerous Misconception:** Training and inference both require sequential forward passes. **Why this is false:** Teacher-forced Transformer positions are parallel; generated tokens depend on earlier selections.
- **The Dangerous Misconception:** $L_2+\lambda L_1$ necessarily means weight regularization. **Why this is false:** GPT-1 names the supervised and auxiliary LM log-likelihoods $L_2$ and $L_1$. The separate code example with L1/L2 norms has a different meaning.
- **The Dangerous Misconception:** A decoder's final position is guaranteed to understand the sequence. **Why this is false:** It has causal access to earlier tokens, which is not a guarantee of task success.
- **The Dangerous Misconception:** 40,000 BPE merges is the vocabulary size. **Why this is false:** Initial symbols and special entries also matter; the worked model specifies 40,478 entries.
- **The Dangerous Misconception:** A lower perplexity always indicates a better model for every task. **Why this is false:** Tokenization, evaluation text and context must match, and likelihood alone does not measure factuality.

## 5. Closed-book self-check — 10 worked answers

1. **Question:** Factor $p(a,b,c)$ and evaluate it for conditionals $0.4,0.6,0.5$.  
   **Answer:** $p(a)p(b\mid a)p(c\mid a,b)=0.4\cdot0.6\cdot0.5=0.12$. No independence assumption is needed.
2. **Question:** The bigram approximation gives factors $0.4,0.3,0.2$. Calculate its probability and compare with question 1.  
   **Answer:** $0.4\cdot0.3\cdot0.2=0.024$; $0.12/0.024=5$. This is a property of the toy factors, not a universal ranking.
3. **Question:** Three observed probabilities are each $1/4$. What is perplexity?  
   **Answer:** Mean NLL is $-\log(1/4)=\log4$ nats/token, so $\exp(\log4)=4$.
4. **Question:** With input `[BOS, a, b]`, which targets align with the three states?  
   **Answer:** `[a, b, c]` when the original continuation is `c`. The BOS state predicts `a`, the `a` state predicts `b`, and the `b` state predicts `c`; a causal mask prevents later input from leaking into earlier states.
5. **Question:** How many future query–key pairs are blocked for five positions?  
   **Answer:** $4+3+2+1=10$, equivalently $T(T-1)/2=5\cdot4/2=10$. The diagonal is allowed.
6. **Question:** Calculate GPT-1-shaped token and position table sizes for $V=40{,}478$, $T=512$, $d=768$.  
   **Answer:** $40{,}478\cdot768=31{,}087{,}104$ and $512\cdot768=393{,}216$; total $31{,}480{,}320$.
7. **Question:** Calculate one FFN's weights and biases for $d=768$, $f=3072$.  
   **Answer:** Weights $2\cdot768\cdot3072=4{,}722{,}432$; biases $3072+768=3{,}840$; total $4{,}726{,}272$. Twelve blocks have $56{,}669{,}184$.
8. **Question:** Why is the output head not counted twice in the executed 116,534,784 total? How many parameters would an untied weight add?  
   **Answer:** It shares the `[40478, 768]` token table, whose transpose produces logits; shared storage is counted once. Untying adds $40{,}478\cdot768=31{,}087{,}104$ weights, assuming no extra bias.
9. **Question:** If supervised log-likelihood $L_2=-0.7$, LM log-likelihood $L_1=-2$ and $\lambda=0.1$, find the combined objective and equivalent minimized loss.  
   **Answer:** $L_3=-0.7+0.1(-2)=-0.9$; minimize $-L_3=0.9$. These are log-likelihoods, not parameter norms.
10. **Question:** How does GPT-1 score three answer choices for a question with context?  
    **Answer:** Serialize context, question and each candidate separately, pass all three through the shared Transformer, obtain one scalar score per final state, and softmax the three scores. For equal scores the probability is $1/3$ each; select the largest score otherwise.

## 6. Last-pass readiness sheet

Check the start token and target shift; multiply conditional probabilities before scaling; specify whether biases and output tying count; separate the paper's objective symbols from norm penalties; and state the protocol when making context-window or perplexity comparisons. The [[week-3-notes|full notes]] contain the executed code and figures, and the [[week-3-exam-questions|question bank]] supplies the routed source questions.
