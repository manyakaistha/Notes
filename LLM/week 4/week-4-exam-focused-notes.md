---
title: "Week 4: Decoding and BERT — Exam-Focused Notes"
week: 4
status: complete
date: 2026-11-13
notes: "[[week-4-notes|Week 4 Notes]]"
map: "[[Knowledge-Map]]"
tags: [llm/week-04, llm/exam, llm/bert]
---

# Week 4 — Decoding and BERT: Exam-Focused Notes

**Navigation:** [[week-3-notes|Week 3 Notes]] · [[Knowledge-Map|Knowledge Map]] · [[week-4-notes|Full Week 4 Notes]] · [[week-4-exam-questions|Exam Questions]]

## 1. What to study more, and why

The source routing manifest assigns 67 distinct questions to Week 4. The counts below are **topic assignments**, not mutually exclusive question counts: a question can be counted in multiple rows. They describe this supplied inventory, not the probability of future exam questions.

| Priority | Canonical topic | Routed questions |
|---|---|---:|
| P1 | Top-k sampling and temperature | 22 |
| P1 | Exhaustive and greedy search | 15 |
| P1 | Top-p (nucleus) sampling and repetition issues | 15 |
| P1 | Beam search and length normalisation | 13 |
| P2 | BERT fine-tuning heads and GPT vs BERT | 11 |
| P2 | NSP, [CLS]/[SEP] and the three input embeddings | 6 |
| P3 | BERT masked language modelling (80/10/10) | 3 |
| P3 | BERT sizes and parameter counts | 2 |

## 2. Search — count, trace, and compare

The chain rule is $P(y_{1:T}\mid x)=\prod_tP(y_t\mid x,y_{<t})$. Exactly length $T$ gives $V^T$ candidates, while lengths 1 through $T$ give $\sum_{t=1}^T V^t$ if every string is admitted. A complete next-token distribution for a prefix requires one logical decoder evaluation: step $t$ has $V^{t-1}$ prefixes and steps 1 through $T$ require $\sum_{j=0}^{T-1}V^j$ evaluations without batching or reuse. EOS constraints, caching, and counting conventions affect the answer.

**How it is asked.** [[week-4-exam-questions#da5004_2025T2_Q1_NA--q153|2025 T2 Q1 Q153]] asks for the decoder runs at step four; [[week-4-exam-questions#da5004_2025T3_Q1_NA--q178|2025 T3 Q1 Q178]] asks about maximum length; [[week-4-exam-questions#da5004_2025T3_Q1_NA--q180|2025 T3 Q1 Q180]] distinguishes $V^{L+1}$ from the increase $V^{L+1}-V^L$. For numerical greedy trees, multiply the chosen conditional probabilities, not unconditional entries; see [[week-4-exam-questions#da5004_2024T1_ET_AN--q148|2024 T1 ET AN Q148]].

Beam width $b$ preserves up to $b$ **complete prefixes** at each step. Expand all retained prefixes, multiply their existing probabilities by each conditional probability, and rank the combined candidates. $b=1$ agrees with greedy under identical stopping and tie rules. A beam of width $V$ is not generally exhaustive after pruning at later depths. Length normalization, e.g. $\sum_t\log p_t/T^\alpha$, changes the ranking and requires a stated EOS convention. [[week-4-exam-questions#da5004_2025T3_Q1_NA--q181|2025 T3 Q1 Q181]] tests beam counts; [[week-4-exam-questions#da5004_2026T2_Q1_NA--q17|2026 T2 Q1 Q17]] asks what width retains a specified path. Its minimum width depends on that path's rank at **every** pruning step, not just the last.

## 3. Top-k, temperature, and nucleus — P1

**Formula sheet.** For logits $z_i$ and $\tau>0$,
$$p_i(\tau)=\frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}},\quad \frac{p_i(\tau)}{p_j(\tau)}=e^{(z_i-z_j)/\tau}.$$
For retained set $S$, $q_i=p_i/\sum_{j\in S}p_j$ for $i\in S$, and zero otherwise. Top-$k$ selects the $k$ highest-probability entries. Top-$p$ keeps the smallest descending prefix with cumulative mass $\ge p$, including the crossing token. If the question defines another equality or tie rule, use that rule explicitly. At $p=1$, all positive-probability support remains.

**How it is asked.** [[week-4-exam-questions#da5004_2026T1_Q2_NA--q17|2026 T1 Q2 Q17]] asks to renormalize a top-two distribution; [[week-4-exam-questions#da5004_2025T2_ET_AN--q198|2025 T2 ET AN Q198]] asks for nucleus size; [[week-4-exam-questions#da5004_2025T2_Q1_NA--q145|2025 T2 Q1 Q145]] asks for a temperature-scaled softmax probability. [[week-4-exam-questions#da5004_2025T3_ET_AN--q2|2025 T3 ET AN Q2]] combines temperature and top-k: positive temperature preserves the logit order, so the top two remain apple and banana. [[week-4-exam-questions#da5004_2026T2_Q2_NA--q10|2026 T2 Q2 Q10]] asks how temperature affects subsequent top-p; flattening can enlarge the nucleus, depending on the distribution and threshold.

**The Dangerous Misconception:** Sampling must produce a different answer on every run. **Why this is false:** identical draws have positive probability; a fixed seed can reproduce a run. Greedy and beam are deterministic when their tie-breaking and runtime are deterministic, but observing repeated outputs does not prove which rule was used. See [[week-4-exam-questions#da5004_2024T1_ET_AN--q150|2024 T1 ET AN Q150]] and [[week-4-exam-questions#da5004_2025T3_Q1_NA--q182|2025 T3 Q1 Q182]].

**The Dangerous Misconception:** $T=2$ is low temperature. **Why this is false:** relative to $T=1$, it flattens logits. One source question labels $T=2$ “Low Temperature”; apply its numerical value, not that label. Temperature zero is undefined in the formula; as $T\to0^+$ tied maxima share the limiting mass. [[temperature-scaling]] and [[top-k-top-p-sampling]] give the full conventions.

## 4. BERT objectives and inputs

Original BERT selects 15% of eligible positions; of those, 80% receive `[MASK]`, 10% random replacement, and 10% remain unchanged. Expected rates over eligible positions are 12%, 1.5%, and 1.5%. Loss is computed for **all selected positions**, including unchanged ones. The unchanged category reduces reliance on `[MASK]` as the sole prediction cue. [[week-4-exam-questions#da5004_2026T1_Q2_NA--q3|2026 T1 Q2 Q3]] tests this motivation; [[week-4-exam-questions#da5004_2026T1_Q2_NA--q4|2026 T1 Q2 Q4]] contrasts BERT's bidirectional context with causal GPT. See [[bert-masked-language-modelling]].

NSP uses 50% actual continuation and 50% randomly sampled second segments, predicting the construction label from `[CLS]`; random does not guarantee semantic unrelatedness. Input is `[CLS] A [SEP] B [SEP]`, with position, token, and segment vectors **summed**, not concatenated. Segment IDs separate A and B; padding requires its own attention mask. [[week-4-exam-questions#da5004_2025T2_Q1_NA--q142|2025 T2 Q1 Q142]] tests the inputs; [[week-4-exam-questions#da5004_2026T1_ET_FN--q6|2026 T1 ET FN Q6]] tests negative pair construction. See [[next-sentence-prediction]] and [[bert-input-embeddings]].

## 5. Configuration, parameters, and task heads

For $V$ vocabulary entries, $P$ positions, $S$ segments, hidden width $H$, intermediate width $I$, and $L$ layers:
$$N_{\rm tables}=(V+P+S)H,\quad N_{\rm layer}=4H^2+2HI+9H+I,$$
$$N_{\rm encoder+pooler}=N_{\rm tables}+2H+L N_{\rm layer}+H^2+H.$$
The $2H$ term is embedding LayerNorm; the final term is the pooler. Specify any omitted biases or heads. Base $(L,H,\text{heads},I)=(12,768,12,3072)$ has 109,482,240 parameters at $V=30,522$ under this convention; large $(24,1024,16,4096)$ has 335,141,888. The paper reports approximately 110M and 340M and describes a 30,000-word vocabulary. See [[bert-parameter-count]] and [[week-4-exam-questions#da5004_2026T1_Q2_NA--q18|2026 T1 Q2 Q18]], which asks for embedding tables only.

For classification, use `[CLS]` and a `[H, C]` projection; mutually exclusive classes use softmax, independent multilabel classes use sigmoid. For named-entity recognition use a per-token head; for extractive QA use start and end position scores and enforce a valid span. [[week-4-exam-questions#da5004_2025T3_ET_AN--q5|2025 T3 ET AN Q5]] asks about four-class classification; [[week-4-exam-questions#da5004_2025T3_ET_AN--q6|2025 T3 ET AN Q6]] asks for disease mention extraction; [[week-4-exam-questions#da5004_2025T3_ET_FN--q6|2025 T3 ET FN Q6]] asks about multilabel abstracts. The original BERT versus GPT comparison concerns bidirectional encoder/MLM plus NSP versus causal decoder/next-token pretraining; architecture does not forbid using either family on other tasks.

## 6. Closed-book self-check — ten fully worked questions

**1. A vocabulary has six tokens. How many length-five strings and step-four prefix evaluations are there?**
There are $6^5=7,776$ strings. Step four has $6^3=216$ distinct length-three prefixes, so 216 logical evaluations supply all next-token vectors. EOS and batching assumptions are excluded.

**2. Greedy picks $A$ with 0.51 then $b$ with 0.60; an alternative picks $B$ with 0.49 then $a$ with 0.99. Which complete sequence is more probable?**
$P(A,b)=.51(.60)=.3060$ and $P(B,a)=.49(.99)=.4851$. The second is larger despite the first greedy step. A width-two beam can retain both at step one.

**3. Raw log scores are $-1.0$ for one token and $-1.5$ for three. What changes under mean log probability?**
Raw score favors $-1.0>-1.5$. Means are $-1.0/1=-1.0$ and $-1.5/3=-0.5$, so the longer candidate ranks first. This is a different scoring objective.

**4. Apply top-2 to $(.40,.25,.20,.10,.05)$. What is the normalized probability of the first token?**
Retained mass is $.40+.25=.65$; the first token receives $.40/.65=8/13\approx0.6154$. All three omitted probabilities become zero.

**5. Apply top-$p=0.8$ to $(.5,.2,.15,.1,.05)$. Give the nucleus and the first token's new probability.**
Cumulative sums are $.5,.7,.85$; keep the first three, including the crossing token. Renormalization gives $.5/.85=10/17\approx0.5882$. Equality with the threshold would already suffice under the stated convention.

**6. For logits $(2,1,0,-1)$ and $T=2$, compute the first token's probability.**
Scaled logits are $(1,0.5,0,-0.5)$, hence $p_1=e^1/(e^1+e^{0.5}+1+e^{-0.5})\approx2.71828/5.97357\approx0.4551$. This temperature flattens the original distribution.

**7. In 1,000 eligible positions, what are the expected MLM category counts?**
$1000(.15)=150$ selected; of these, $150(.8)=120$ mask replacements, $150(.1)=15$ random replacements, and 15 unchanged. All 150 are supervised; these are expectations, not guaranteed exact counts.

**8. Write a BERT sentence-pair layout and identify its embeddings and NSP labels.**
`[CLS] A [SEP] B [SEP]` sums token, position, and segment vectors per position. Original NSP uses actual-next and randomly selected B examples in equal proportions. `[CLS]` feeds a binary classifier; segment IDs label A/B portions, not attention directions.

**9. Compute the embedding table count for $V=5000$, $P=64$, $S=2$, $H=128$.**
The three tables contain $(5000+64+2)128=5066(128)=648,448$ parameters. If asked to include affine embedding LayerNorm, add $2(128)=256$ for 648,704. These exclude the encoder and pooler.

**10. Choose heads for spam classification, entity tagging, and extractive QA.**
Spam classification projects the final `[CLS]` vector into class logits. Entity tagging applies a shared class projection to each token vector, masking padding in the loss. QA projects each position to start and end scores and chooses a valid passage span with start $\le$ end; no-answer handling needs an explicit rule or score.

## 7. Last-pass formula and readiness sheet

- $V^T$ counts fixed-length strings; $\sum_{t=1}^{T}V^t$ counts all admitted lengths; $\sum_{j=0}^{T-1}V^j$ counts prefix evaluations through step $T$ under the stated model.
- Greedy is local; finite beam can prune the optimum. Mean log scoring changes the objective.
- $T\downarrow$ sharpens; $T\uparrow$ flattens. Top-$k$ fixes count; top-$p$ adapts count to cumulative mass. Normalize **after** filtering.
- MLM's 80/10/10 proportions are conditional on selection. NSP is pair classification, not autoregressive prediction.
- `[CLS]` supports sequence classification; `[SEP]` marks a boundary; token/segment/position embeddings are summed. BERT's padding attention mask is distinct from `[MASK]`.
- The source paper is Jacob Devlin et al. (2019), [BERT, NAACL-HLT](https://doi.org/10.18653/v1/N19-1423); Ari Holtzman et al. (2020), [The Curious Case of Neural Text Degeneration, ICLR](https://openreview.net/forum?id=rygGQyrFvH), motivates nucleus sampling.
