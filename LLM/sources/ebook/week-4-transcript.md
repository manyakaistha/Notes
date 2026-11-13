<!-- PDF p.45 / printed p.44 -->
# 4 Week 4 — Decoding Strategies & BERT (Bidirectional Encoder Representations)

**Chapter goals.** By the end of this chapter you will be able to:
1. Distinguish deterministic vs. stochastic decoding strategies.
2. Trace exhaustive search, greedy search, and beam search, with worked numerical examples.
3. Define and contrast top-$k$, top-$p$ (nucleus), and temperature sampling.
4. Explain why bidirectional context matters and why GPT-style models cannot provide it.
5. Derive the BERT pre-training objective: Masked Language Modelling (MLM) + Next Sentence Prediction (NSP).
6. Count BERT-base parameters ($\sim 110$M) and dissect its embedding scheme (token + segment + position).
7. Apply BERT to downstream tasks: classification, extractive question answering.

## Part A — Decoding Strategies
## 4.1 The Wishlist Restated
In Week 3 we built a Causal Language Model that, at each position, outputs a probability distribution $P(x_i \mid x_{<i})$ over the vocabulary. *How do we actually pick the next token from this distribution?*

> **Intuition**
>
> Different choices of decoding strategy turn the *same* pre-trained model into completely different generators:
> - For machine translation we want the *most likely* target sentence.
> - For story writing we want *creativity* — diverse outputs for the same prompt, while still being coherent.
> - For real-time chat we want *speed* on top of quality.
> There is no single best strategy. Decoding is a knob you turn at *inference time* to match the task.

**The two families of decoding strategies:**

| Family | Examples | Output for same prompt |
|---|---|---|
| Deterministic | Exhaustive search, Greedy search, Beam search | Always the same |
| Stochastic (Sampling) | Top-$k$, Top-$p$ (nucleus), Temperature, Contrastive decoding | Different each run |

<!-- PDF p.46 / printed p.45 -->
## 4.2 Exhaustive Search — The Theoretical Ideal
> **Definition**
>
> **Exhaustive search** enumerates *every* possible sequence of length $T$, computes the joint probability via the chain rule, and returns the one with the highest probability:
> $$\hat Y=\arg\max_{Y\in\mathcal V^T}\prod_{i=1}^{T}P(y_i\mid y_{<i}).$$

**Why it is impossible at scale.** With vocabulary $|\mathcal V|=40{,}000$ and length $T=5$, there are $|\mathcal V|^T=40{,}000^5\approx 10^{23}$ candidate sequences. Even at one billion sequences per second, this would take $\sim 3$ million years. So we need approximations.

### 4.2.1 Tiny illustration
Suppose $\mathcal V=\{A,B,C\}$ and we want length-2 sequences. At $t=1$ the decoder outputs $P(A),P(B),P(C)$. At $t=2$ we need to run the decoder *three times in parallel* (once for each possible $x_1$) to get $P(\cdot\mid A)$, $P(\cdot\mid B)$, $P(\cdot\mid C)$. That’s $3+9=12$ forward passes for $T=2$; at $T=3$ it explodes to 27. With $|\mathcal V|=40{,}000$ and $T=5$, the count is intractable.
> [possible-error p.46: The stated forward-pass count conflates candidate tokens with decoder evaluations; $40{,}000^5$ is approximately $10^{23}$ but takes about 3 million years at one billion per second.]

## 4.3 Greedy Search — The Other Extreme
> **Definition**
>
> **Greedy search** picks the most probable next token at each step, then conditions on that choice for the next step:
> $$y_i=\arg\max_{y\in\mathcal V}P(y\mid y_{<i}).$$

### 4.3.1 Worked example
Suppose $\mathcal V=\{\text{cold},\langle\text{stop}\rangle,\text{coffee},\text{I},\text{like},\text{water}\}$ and the model’s conditional probabilities at each of 5 time steps (greedy picks shown in bold) are:

| | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ |
|---|---:|---:|---:|---:|---:|
| cold | 0.10 | 0.10 | **0.45** | 0.15 | 0.10 |
| $\langle$stop$\rangle$ | 0.15 | 0.15 | 0.05 | 0.30 | **0.50** |
| coffee | 0.25 | 0.25 | 0.10 | **0.35** | 0.20 |
| I | **0.40** | 0.05 | 0.05 | 0.01 | 0.10 |
| like | 0.05 | **0.35** | 0.15 | 0.09 | 0.05 |
| water | 0.05 | 0.10 | 0.20 | 0.10 | 0.05 |

**Greedy output:** “I like cold coffee $\langle$stop$\rangle$”.
**Joint probability:**
$$0.40\times0.35\times0.45\times0.35\times0.50=0.011.$$

### 4.3.2 Limitations of greedy
- **Locally optimal $\neq$ globally optimal.** A locally low-probability token at $t=1$ might unlock a much higher-probability continuation. Greedy never explores that branch.
- **Deterministic.** For the same prompt, you always get the same output — no diversity.
- **Repetition.** “I like to think that I like to think that...” — once a phrase is locally most probable, the same loop repeats.

<!-- PDF p.47 / printed p.46 -->
> **Intuition**
>
> **Why greedy can be globally suboptimal.** If at $t=1$ we instead picked the second most probable token (say “coffee” with $P=0.25$), the conditional distributions at subsequent steps change. Suppose with that prefix we then get $0.55,0.65,0.80,0.50$ at subsequent steps, giving total $0.25\times0.55\times0.65\times0.80\times0.50=0.0357$ — *three times higher* than the greedy sequence’s 0.011! This is exactly why beam search is helpful.

## 4.4 Beam Search — The Practical Sweet Spot
> **Definition**
>
> **Beam search** maintains the $k$ most probable *partial sequences* (called *beams*) at every step. At step $t+1$, each beam is extended by all $|\mathcal V|$ possible tokens, and the top $k$ of the resulting $k\cdot|\mathcal V|$ candidates are kept. The parameter $k$ is the **beam size**.

### 4.4.1 Special cases
- $k=1\Rightarrow$ greedy search.
- $k=|\mathcal V|$ (or larger) $\Rightarrow$ exhaustive search.
- Typical practical values: $k=4,5,8$ for translation; up to 20 for careful research.
> [possible-error p.47: A beam width equal to vocabulary size does not generally make search exhaustive for sequences longer than one token.]

### 4.4.2 Worked beam-search example with $k=2$
Vocabulary $\{A,B,C\}$, sequence length 3. At $t=1$:
$$P(A)=0.5,\quad P(B)=0.4,\quad P(C)=0.1.$$
The top-2 beams are $\{A\}$ and $\{B\}$ (we drop $C$).

At $t=2$, for each surviving beam we look at the full conditional. Suppose
$$P(\cdot\mid A)=\{A:0.1,\ B:0.2,\ C:0.5\}\quad\text{(and others sum to 0.2)}$$
$$P(\cdot\mid B)=\{A:0.2,\ B:0.2,\ C:0.6\}$$
> [possible-error p.47: The stated vocabulary contains only A, B and C, so there are no other tokens to receive probability 0.2.]

We expand both beams to get 6 candidate length-2 sequences:

| Sequence | Probability | Sequence | Probability |
|---|---|---|---|
| $(A,A)$ | $0.5\times0.1=0.05$ | $(B,A)$ | $0.4\times0.2=0.08$ |
| $(A,B)$ | $0.5\times0.2=0.10$ | $(B,B)$ | $0.4\times0.2=0.08$ |
| $(A,C)$ | $0.5\times0.5=\mathbf{0.25}$ | $(B,C)$ | $0.4\times0.6=\mathbf{0.24}$ |

Top 2: $\{(A,C),\ (B,C)\}$. We keep these two beams forward to $t=3$.
At $t=3$, suppose $P(B\mid A,C)=0.75$ and $P(A\mid B,C)=0.55$. Then:
$$P(A,C,B)=0.25\times0.75=0.1875,$$
$$P(B,C,A)=0.24\times0.55=0.132.$$
Output: $(A,C,B)$ — the beam with highest final probability.

<!-- PDF p.48 / printed p.47 -->
### 4.4.3 Length normalisation
> **Formula Insight**
>
> **The length-bias problem.** Each multiplication by a probability $<1$ shrinks the joint. Longer sequences will always have *smaller* probabilities, so beam search naturally prefers short outputs — even if they truncate the sentence prematurely.
> **Fix:** divide by sequence length (or use a length penalty $\ell^\alpha$ with $\alpha\in[0.6,0.7]$):
> $$\operatorname{score}(Y)=\frac{1}{\ell(Y)}\sum_{i=1}^{\ell(Y)}\log P(y_i\mid y_{<i}).$$
> This is standard in production MT systems.

### 4.4.4 Cost and behaviour
Beam search requires $k\cdot|\mathcal V|$ score computations per step (and $k$ forward passes through the model in parallel, since each beam is its own context). Total: $\mathcal O(k\cdot|\mathcal V|\cdot T)$ compared to greedy’s $\mathcal O(|\mathcal V|\cdot T)$.

**Where beam search shines:** translation, summarisation, structured generation where “correct output” is well-defined.

**Where it disappoints:** open-ended generation (creative writing, dialogue) — it tends to produce bland, repetitive, “newspaper-headline” text, because the most probable sequences in human language are also the most generic ones.

## 4.5 Sampling — Embracing Stochasticity
> **Intuition**
>
> A 2019 study (Holtzman et al., *The Curious Case of Neural Text Degeneration*) plotted, position-by-position, the rank of the token that a human writer chose. Humans pick the top-1 rank only $\sim 20\%$ of the time. Beam search picks rank-1 $\sim 80\%$ of the time. **Surprise is what makes text interesting.**

To inject controlled surprise, we *sample* from the distribution instead of taking the argmax. Pure “ancestral” sampling (sample directly from $P$) is too noisy — it sometimes picks low-quality, low-probability tokens. The three refinements below tame the noise.

### 4.5.1 Top-$k$ sampling
> **Definition**
>
> **Top-$k$ sampling** keeps only the $k$ highest-probability tokens at each step, renormalises their probabilities to sum to 1, and samples from that truncated distribution.

**Example.** Suppose at $t=1$ the model gives $P=(0.40,0.25,0.15,0.10,0.05,0.05)$ over (I, coffee, like, cold, water, $\langle$stop$\rangle$). With $k=2$, we keep {I: 0.40, coffee: 0.25}, renormalise to {I: 0.61, coffee: 0.39}, and sample according to those probabilities.

**Limitation.** A *flat* distribution (many roughly-equal options) is truncated too aggressively — we lose creativity. A *peaked* distribution (one dominant option) admits too many implausible options. *There is no single $k$ that works for both.*

<!-- PDF p.49 / printed p.48 -->
### 4.5.2 Temperature scaling
> **Definition**
>
> **Temperature sampling** reshapes the distribution before sampling:
> $$P_T(x=u_\ell\mid x_{<i})=\frac{\exp(u_\ell/T)}{\sum_{\ell'}\exp(u_{\ell'}/T)},$$
> where $u_\ell$ is the pre-softmax logit and $T>0$ is the *temperature*.

**Behaviour:**
- $T=1$: original softmax distribution.
- $T\to0$: distribution becomes a one-hot at the argmax (deterministic / greedy).
- $T\to\infty$: distribution becomes uniform (pure noise).
- Typical range: $T\in[0.5,1.5]$ for creative generation.
> [possible-error p.49: The equation equates a token $x$ with a logit $u_\ell$; it should identify the token indexed by $\ell$. The zero-temperature limit can have multiple maxima.]

> **Formula Insight**
>
> Why dividing by $T$ works: writing $\operatorname{softmax}(u/T)=\operatorname{softmax}((u_1/T,u_2/T,\ldots))$, larger $T$ *shrinks* the differences between logits, flattening the distribution. Smaller $T$ *magnifies* differences, sharpening the distribution.

### 4.5.3 Top-$p$ (nucleus) sampling
> **Definition**
>
> **Top-$p$ (nucleus) sampling** chooses the smallest set of tokens whose cumulative probability exceeds a threshold $p\in(0,1)$, then samples from that set:
> 1. Sort tokens by probability in descending order.
> 2. Accumulate probabilities top-down until the cumulative sum exceeds $p$.
> 3. Renormalise the selected tokens and sample.

**Why it beats top-$k$.** The size of the kept set is *adaptive*:
- **Flat distribution** (e.g., “she said ‘I never __’” has many plausible continuations: *thought, knew, had, saw, said*, ...): with $p=0.6$, the model keeps perhaps 5 tokens.
- **Peaked distribution** (e.g., “I ate the pizza while it was still __” — “hot” dominates): with $p=0.6$, the model keeps only 1-2 tokens.

**Recommended setting (Holtzman et al.):** $p=0.9$ or $p=0.95$. This is the de facto standard for open-ended generation in modern LLMs.

### 4.5.4 Combining temperature with top-$p$
In practice, ChatGPT, Claude, and similar models use *both*: temperature reshapes the distribution, then nucleus sampling truncates it. Two knobs the user can turn:
- Lower temperature ($\sim 0.2$) + top-$p$ ($\sim 0.9$) $\to$ focused, deterministic-ish answers (factual QA).
- Higher temperature ($\sim 0.9$) + top-$p$ ($\sim 0.95$) $\to$ creative writing, brainstorming.

<!-- PDF p.50 / printed p.49 -->
### 4.5.5 Contrastive decoding (brief)
A newer family (Li et al., 2023): generate from a strong model, but *down-weight* tokens that a weaker reference model would also predict highly. The intuition: tokens that both the small and large model agree on are usually generic; the large model’s *distinctive* preferences are more informative. Used for higher-quality long generation.

## 4.6 Decoding Comparison Cheat Sheet
| | Greedy | Beam | Top-$k$ / $p$ + T | Contrastive |
|---|---|---|---|---|
| Output diversity | none | low | high (tunable) | moderate |
| Repetition tendency | high | medium | low | low |
| Best for | quick demos | translation, summarisation | dialogue, creative writing | long-form generation |
| Determinism | yes | yes | no (with fixed seed: yes) | yes |
| Latency | low | high ($k\times$) | low | medium |
| Quality bias | bland | generic | creative but risk noise | quality |

## Part B — BERT and Masked Language Modelling
## 4.7 Why We Need Bidirectional Context
GPT (Week 3) builds a representation *left-to-right*. This is great for generation, where the future is unknown anyway. But many tasks need *bidirectional* understanding:
- **Named Entity Recognition** (e.g., “**Nothing** has shipped its new OS to **Nothing Phone 2**”): the second mention of “Nothing” is a brand name — you need the *right* context (“Phone 2”) to disambiguate it from the negation.
- **Fill-in-the-blank:** “*I __ to read a __.*” To predict the first blank as “love”, the model should see the second blank’s context (“book”) as well.
- **Sentence classification:** any token in the sentence can be relevant to determining the label.

> **Intuition**
>
> GPT can read only the past; BERT reads the whole sentence at once. Both architectures *compute* self-attention over the full sequence, but GPT *masks* the future, while BERT does not. This single design choice is the core difference between the two families.

<!-- PDF p.51 / printed p.50 -->
## 4.8 Masked Language Modelling (MLM) — The BERT Pre-training Objective
> **Definition**
>
> In **Masked Language Modelling** we corrupt the input by replacing some tokens with a special `[MASK]` token, then ask the model to predict the original tokens:
> $$P(y_i\mid x_1,\ldots,x_{i-1},[\mathrm{MASK}],x_{i+1},\ldots,x_T)=\ ?$$
> The model must use *both* left and right context to recover the masked words.

### 4.8.1 Why we cannot just “mask attention weights” the way GPT does
In GPT, masking the upper triangle of the attention matrix forbids attending to the future. Could we do the analogue here — mask the column corresponding to the word to predict?

**No.** If position $i$ cannot attend to itself, then the *output* hidden state at position $i$ would not depend on the input at position $i$ at all — which is exactly the problem we want: don’t tell the model the answer. But then the *other positions* also cannot attend to position $i$ (and we don’t want that either). The correct trick is different:
> [possible-error p.51: Masking self-attention alone does not remove the input at position $i$ from its output because residual connections retain it.]

> **Remember This**
>
> **Insert a special `[MASK]` token into the input itself.** Add `[MASK]` as a new token in the vocabulary, learn an embedding for it, and use this replacement at the input positions we want to predict. Every other token can attend to `[MASK]` (it has its own learnable embedding), and the model is asked to recover the original token from the `[MASK]` position’s final hidden state. This is exactly the *denoising* objective of an autoencoder.

### 4.8.2 Self-supervision in one diagram
[Figure: `[MASK] enjoyed the [MASK] transformers` passes through bidirectional self-attention and a feed-forward network to predictions $\hat y_1$, $\hat y_4$, with losses $\mathcal L_1=-\log\hat y_1$ and $\mathcal L_4=-\log\hat y_4$.]

The loss is summed only over the masked positions, not over every position:
$$\mathcal L_{\mathrm{MLM}}=-\frac{1}{|M|}\sum_{i\in M}\log\hat y_i, \tag{26}$$
where $M$ is the set of masked positions and $\hat y_i$ is the model’s predicted probability of the true original token at position $i$.

### 4.8.3 The masking recipe
BERT’s exact recipe:

<!-- PDF p.52 / printed p.51 -->
- Sample 15% of tokens uniformly at random.
- Of those:
  - 80% are replaced by `[MASK]`.
  - 10% are replaced by a *random* token from the vocabulary.
  - 10% are left *unchanged*.

> **Formula Insight**
>
> **Why the 80/10/10 split?**
> If we always replaced with `[MASK]`, the model would learn that “the question is always at a `[MASK]` position”. But at fine-tuning and inference, there are no `[MASK]` tokens! The other 20% (10% random, 10% unchanged) trains the model to make predictions even at non-`[MASK]` positions, reducing the train/test mismatch.

### 4.8.4 Why 15%?
- Too *low* masking (e.g., 1%): gradient signal per example is tiny; training is wasteful and slow.
- Too *high* masking (e.g., 50%): the model loses too much context to learn good representations.
- 15% is the empirical sweet spot from the BERT paper. (Subsequent work — e.g., the “Should you mask 15%?” paper — shows that 40% can be optimal for larger models.)

## 4.9 Next Sentence Prediction (NSP)
MLM gives BERT *word-level* understanding. But many downstream tasks need *sentence-pair* understanding: question answering (does this passage answer the question?), entailment (does sentence A imply sentence B?), paraphrase detection.

> **Definition**
>
> **Next Sentence Prediction** is a binary classification task added during pre-training:
> - Input: two sentences A and B, separated by a special `[SEP]` token, with a special `[CLS]` token at the start.
> - Label: `IsNext` (B genuinely follows A in the source corpus) vs. `NotNext` (B is a random sentence).
> - Training: 50% positive pairs, 50% random pairs.
> The model uses the final hidden state at the `[CLS]` position to predict the label.

### 4.9.1 Combining MLM and NSP losses
BERT is pre-trained jointly: simultaneously predict masked tokens (MLM) and the `IsNext` label (NSP). The total loss is
$$\mathcal L=\underbrace{-\frac{1}{|M|}\sum_{i\in M}\log\hat y_i}_{\mathcal L_{\mathrm{MLM}}}+\underbrace{-\log\hat y_{\mathrm{CLS}}}_{\mathcal L_{\mathrm{NSP}}}. \tag{27}$$

> **Research / Industry Insight**
>
> NSP was later shown to be **harmful or redundant**. RoBERTa (Liu et al., 2019) removed NSP entirely and got better downstream performance by simply pre-training MLM for longer on more data. SpanBERT replaced NSP with a span-prediction task. Today, production-grade encoder-only LLMs mostly skip NSP.

<!-- PDF p.53 / printed p.52 -->
## 4.10 BERT’s Three-Embedding Input
Because BERT processes sentence pairs and must distinguish which sentence a token belongs to, the input embedding for each position is the *sum* of three learned embeddings:

| Embedding | Indexed by | Purpose |
|---|---|---|
| **Token embedding** $E_T$ | token id ($\in\{1,\ldots,|\mathcal V|\}$) | word meaning |
| **Segment embedding** $E_A/E_B$ | sentence A or sentence B | which side of `[SEP]` |
| **Position embedding** $E_0,E_1,\ldots,E_{511}$ | position index | order in the sequence |

The final input for each token:
$$E_{\mathrm{input}}[i]=E_T[\mathrm{tok}_i]+E_{A/B}[\mathrm{seg}_i]+E_{\mathrm{pos}}[i].$$

### 4.10.1 The two special structural tokens
- `[CLS]` (classification): always at position 0. Its final hidden state is used as a *sentence-pair summary* for NSP and for downstream classification.
- `[SEP]` (separator): marks the boundary between sentence A and sentence B, and also marks the end of the input.

> **Quiz Trap**
>
> **Trap.** “`[CLS]` is just like `[MASK]`: a placeholder token.”
> **Reality.** `[CLS]` is a structural token whose *final hidden state* is the sentence-level summary used for classification. `[MASK]` is a corruption token whose final hidden state is used to predict the *original* token. They are separate tokens with separate embeddings and separate roles.

## 4.11 BERT Architecture and Configurations
BERT is a stack of bidirectional Transformer encoder layers (Week 1’s encoder!) with no causal masking.

| Hyperparameter | BERT-base | BERT-large |
|---|---|---|
| Layers ($n$) | 12 | 24 |
| Hidden size ($d_{\mathrm{model}}, H$) | 768 | 1024 |
| Attention heads ($h$) | 12 | 16 |
| Intermediate FFN size ($4H$) | 3072 | 4096 |
| Vocabulary size | 30,522 (WordPiece) | 30,522 |
| Max context length ($T$) | 512 | 512 |
| Parameters | $\approx110$ M | $\approx340$ M |
| Pre-training corpus | BookCorpus (800M) + Wikipedia (2500M) | same |

## 4.12 Counting BERT-base Parameters — The Famous 110M
### 4.12.1 Embedding layer
- Token embeddings: $30{,}000\cdot768\approx23.0$ M.

<!-- PDF p.54 / printed p.53 -->
- Segment embeddings: $2\cdot768=1{,}536\approx0.002$ M (negligible).
- Position embeddings: $512\cdot768\approx0.39$ M.
- **Subtotal: $\approx23.4$ M.**

### 4.12.2 Per-layer attention + FFN
- Self-attention QKV: $3\cdot768\cdot64\cdot12\approx1.77$ M.
- Output projection $W^O$: $768\cdot768\approx0.59$ M.
- FFN: $2\cdot768\cdot3072+(3072+768)\approx4.72$ M.
- **Per layer: $\approx7$ M.**

### 4.12.3 Grand total
$$23.4\mathrm{M}+12\cdot7\mathrm{M}=23.4+84=107.4\mathrm{M}.$$
The paper rounds to $\sim110$M (the slight difference comes from the actual vocab size of 30,522, LayerNorm parameters, and the MLM head). **Memorise: BERT-base $\approx110$M, BERT-large $\approx340$M.**

> **Exam Tip**
>
> **Mnemonic comparison.** GPT-1 was $\sim117$M and BERT-base was $\sim110$M. Roughly the same size, but radically different objectives and capabilities. Scale alone is not destiny — the *objective* matters.

## 4.13 Adapting BERT to Downstream Tasks
There are two main strategies.

### 4.13.1 Strategy 1 — BERT as a Feature Extractor (Body Frozen)
1. Feed the input through pre-trained BERT.
2. Take the final hidden state at `[CLS]` as a *frozen* sentence vector.
3. Train any classical ML model (logistic regression, Naive Bayes, a small NN) on top of that frozen vector.

**Pros.** Fast, no GPU needed for the head, simple. **Cons.** Lower ceiling than fine-tuning.

### 4.13.2 Strategy 2 — Fine-tuning the Whole Model
1. Add a small randomly-initialised task-specific head (often just a linear layer).
2. Train *end-to-end* on the labelled task, with a small learning rate so the pre-trained weights drift slowly.
3. No masking of input tokens during fine-tuning — the model has already learned to handle non-`[MASK]` positions thanks to the 80/10/10 split.

**Pros.** Higher accuracy with fewer labelled examples. **Cons.** Needs GPU to fine-tune; updates millions of parameters.

### 4.13.3 Worked example: extractive question answering (SQuAD-style)
**Task.** Given a paragraph and a question, find the *span* (contiguous substring of the paragraph) that answers the question.

**Input format.**
`[CLS]` question tokens `[SEP]` paragraph tokens `[SEP]`

<!-- PDF p.55 / printed p.54 -->
**Output head.** Two learnable vectors $S,E\in\mathbb R^{d_{\mathrm{model}}}$ (start and end). For each paragraph position $i$ with hidden state $h_i$:
$$s_i=\frac{\exp(S\cdot h_i)}{\sum_j\exp(S\cdot h_j)}\quad\text{(probability that $i$ is the start token)},\tag{28}$$
$$e_i=\frac{\exp(E\cdot h_i)}{\sum_j\exp(E\cdot h_j)}\quad\text{(probability that $i$ is the end token)}.\tag{29}$$
At inference, the predicted answer span is $[i^*,j^*]$ with $i^*=\arg\max_i s_i$, $j^*=\arg\max_{j\ge i^*}e_j$, subject to $j\ge i$.

**Edge case.** If $j^*<i^*$ (i.e., the most likely end is before the most likely start), return the empty string — this typically signals “the answer is not in the passage”.
> [possible-error p.55: The constraint $j\ge i^*$ in the preceding definition prevents the stated edge case $j^*<i^*$.]

> **Intuition**
>
> This is a beautifully simple example of how to adapt BERT: *add two learnable vectors* $S,E$ (a couple of thousand parameters), inherit the rest from pre-training, fine-tune. With SQuAD-1.1, BERT-large reaches superhuman F1, beating many task-specific architectures designed before it.

## 4.14 GPT vs BERT — The Big Comparison
| | GPT-1 | BERT-base |
|---|---|---|
| Architecture | Decoder-only Transformer | Encoder-only Transformer |
| Attention | Masked (causal) self-attention | Bidirectional self-attention |
| Pre-training objective | Causal LM (predict next token) | MLM + NSP |
| Context direction | left-to-right only | both directions |
| Output for classification | last token hidden state | `[CLS]` hidden state |
| Best for | generation, dialogue, completion | understanding, classification, span tasks |
| Parameters | $\approx117$M | $\approx110$M |
| Pre-training data | BookCorpus (800M words) | BookCorpus + Wikipedia (3300M words) |
| Year | June 2018 | October 2018 |

> **Remember This**
>
> **When to use which?**
> - Need to *generate* text? GPT-style (decoder-only).
> - Need a *sentence embedding* for classification or retrieval? BERT-style (encoder-only).
> - Need to *transform* one sequence into another (translation, summarisation)? Encoder-decoder (T5, BART) is a natural fit, though modern decoder-only LLMs handle these too.

<!-- PDF p.56 / printed p.55 -->
## 4.15 Common Mistakes & Quiz Traps — Week 4
> **Common Mistake**
>
> **Mistake 1.** “Greedy search always produces the most probable sequence.”
> **Fix.** It produces the sequence that is *locally* most probable at each step. The globally most probable sequence may require an early choice that is not locally maximal.

> **Common Mistake**
>
> **Mistake 2.** “Beam search with $k=1$ gives diverse outputs.”
> **Fix.** $k=1$ *is* greedy search — fully deterministic.

> **Common Mistake**
>
> **Mistake 3.** “Top-$p=1$ samples from the full distribution.”
> **Fix.** Top-$p=1$ keeps all tokens whose cumulative probability is below 1. Since all probabilities sum to 1, this keeps the entire vocabulary — equivalent to plain ancestral sampling (without any truncation).
> [possible-error p.56: The alleged mistake is correct, and the stated fix agrees with it.]

> **Common Mistake**
>
> **Mistake 4.** “BERT uses causal masking like GPT, but with a different vocabulary.”
> **Fix.** BERT uses *bidirectional* attention. The only “mask” in BERT is the `[MASK]` input token, not an attention mask.
> [possible-error p.56: BERT implementations also use attention masks to exclude padding.]

> **Quiz Trap**
>
> **Trap.** “If we mask 100% of tokens, MLM would be ideal because it gives the strongest gradient.”
> **Reality.** With 100% masking, the model sees no input context and is asked to “predict everything from nothing” — a useless task. Empirically, $\sim15\%$ is the sweet spot.

> **Quiz Trap**
>
> **Trap.** “Lower temperature always improves text quality.”
> **Reality.** Low $T$ kills creativity and induces repetition. High $T$ injects noise and incoherence. The right $T$ depends on the task; ChatGPT defaults to around 1 and lowers it for code/factual queries.

## 4.16 Memory Tricks — Week 4
> **Memory Trick**
>
> **“E-G-B-K-T-N-C”** — the decoding ladder from most to most flexible: **E**xhaustive, **G**reedy, **B**eam, top-**k**, **T**emperature, **N**ucleus (top-$p$), **C**ontrastive.

> **Memory Trick**
>
> **“$T$ down = peaked, $T$ up = flat”.**
> $T\to0$ behaves like greedy (everything goes to the argmax); $T\to\infty$ becomes uniform random.

<!-- PDF p.57 / printed p.56 -->
> **Memory Trick**
>
> **“15% : 80 / 10 / 10”.**
> Mask 15% of tokens; of those, 80% become `[MASK]`, 10% become random tokens, 10% are unchanged.

> **Memory Trick**
>
> **“BERT = CLS + SEP + MASK”**
> Three special tokens: `[CLS]` for sentence-level summary, `[SEP]` for sentence boundary, `[MASK]` for the prediction target.

## 4.17 Quick Revision — Week 4 in One Page
> **Quick Revision**
>
> **Decoding strategies (deterministic).**
> - **Exhaustive:** optimal but $\mathcal O(|\mathcal V|^T)$, infeasible.
> - **Greedy:** argmax at every step. $k=1$ beam search.
> - **Beam:** keep top-$k$ partial sequences. Length-normalise to prevent bias toward short outputs.
>
> **Decoding strategies (stochastic).**
> - **Top-$k$:** sample from top-$k$ tokens after renormalising.
> - **Temperature $T$:** $\operatorname{softmax}(u/T)$. Lower = sharper, higher = flatter.
> - **Top-$p$ (nucleus):** keep smallest set with cumulative probability $\ge p$. Dynamic, adaptive.
>
> **BERT.**
> - Encoder-only Transformer; bidirectional self-attention.
> - **MLM** objective: mask 15% of tokens (80 / 10 / 10 split); predict originals.
> - **NSP** objective: predict if sentence B follows sentence A.
> - Three embeddings summed at input: token + segment ($E_A/E_B$) + position.
> - Specials: `[CLS]` (classification), `[SEP]` (separator).
> - **BERT-base:** 12 layers, $H=768$, $h=12$, $\sim110$M parameters.
> - Downstream: feature-extractor or fine-tune; QA uses learned start/end vectors $S,E$.
>
> **Both GPT and BERT are $\sim110$M parameters** but with opposite design philosophies (generation vs. understanding).

## 4.18 Practice Questions — Week 4
### Multiple Choice
1. **(Q1)** Greedy decoding is equivalent to beam search with:

   (A) $k=0$　(B) $k=1$　(C) $k=|\mathcal V|$　(D) any odd $k$.

   **Solution.** (B). With $k=1$, only the single best partial sequence is kept — which is exactly greedy.

2. **(Q2)** For open-ended text generation, the most commonly recommended decoding setting is:

   (A) Beam search with large $k$,

   (B) Greedy decoding,

   (C) Temperature $\approx1$ with top-$p\in[0.9,0.95]$,

<!-- PDF p.58 / printed p.57 -->
   (D) Random sampling with $T=100$.

   **Solution.** (C). Open-ended generation benefits from controlled stochasticity; (A) gives bland generic text, (B) repeats, (D) is incoherent.

3. **(Q3)** The percentage of tokens masked in BERT’s MLM pre-training is:

   (A) 5%　(B) 15%　(C) 30%　(D) 50%.

   **Solution.** (B).

4. **(Q4)** In BERT, the `[CLS]` token’s final hidden state is used for:

   (A) predicting the masked tokens,

   (B) sentence-level classification (e.g., NSP, downstream sentiment),

   (C) rotational position encoding,

   (D) separating sentence A from sentence B.

   **Solution.** (B). (D) is the `[SEP]` token’s job.

5. **(Q5)** Of the masked tokens in BERT MLM, the fraction replaced with a random vocabulary token is:

   (A) 80%　(B) 10%　(C) 50%　(D) 1%.

   **Solution.** (B). Recall the 80/10/10 split: 80% `[MASK]`, 10% random, 10% unchanged.

6. **(Q6)** Which of these is *deterministic* (same output for same prompt with no randomness)?

   (A) Top-$k$ sampling,

   (B) Nucleus (top-$p$) sampling,

   (C) Beam search,

   (D) Temperature sampling.

   **Solution.** (C). Beam search is deterministic. (A), (B), (D) all introduce randomness through sampling.

### Conceptual / Short Answer
7. **(Q7)** Explain in 2–3 sentences why beam search prefers shorter sequences unless length normalisation is applied.

   **Answer.** Each multiplication of a conditional probability ($\le1$) shrinks the joint. Longer sequences accumulate more such factors and end up with smaller joint probabilities — so beam search prematurely terminates. Length normalisation (dividing the log-score by sequence length or a length penalty $\ell^\alpha$) corrects this bias.

8. **(Q8)** Why is top-$p$ (nucleus) usually preferred over top-$k$?

   **Answer.** Because top-$p$’s set size is *adaptive*: for flat distributions it keeps many tokens (preserving creativity), and for peaked distributions it keeps few (preserving coherence). Top-$k$ uses the same $k$ for both regimes and fails on at least one of them.

9. **(Q9)** The 80/10/10 split in BERT’s MLM exists because:

   **Answer.** Always replacing with `[MASK]` would cause a train/inference mismatch: the model would only learn to predict at `[MASK]` positions, but downstream tasks have no `[MASK]` tokens.

<!-- PDF p.59 / printed p.58 -->
   The 10% random and 10% unchanged tokens force the model to make predictions even at “normal” positions, reducing the gap between pre-training and fine-tuning.

10. **(Q10)** Could we use a decoder-only model (like GPT) for fill-in-the-blank? Why or why not?

    **Answer.** Not natively. GPT’s causal mask blocks attention to the future, so at the blank position the model cannot use right-side context — it only sees the left. For fill-in-the-blank we need bidirectional context, which is exactly what BERT provides.

11. **(Q11)** For extractive QA with BERT, why do we need *two* learnable vectors $S$ and $E$ rather than one?

    **Answer.** We need to identify two distinct positions in the paragraph: the start of the answer span and the end. The dot products $S\cdot h_i$ and $E\cdot h_i$ let us score each position separately as a start candidate or end candidate. Both distributions are softmaxed across paragraph positions independently.

### Numerical / Derivation
12. **(Q12)** A model has output probabilities $P=(0.5,0.3,0.15,0.05)$ for the four candidate next tokens (in descending order). Compute the kept set for (a) top-$k$ with $k=2$, and (b) top-$p$ with $p=0.7$.

    **Solution.** (a) Top-2 $=\{0.5,0.3\}$, renormalised to $\{0.625,0.375\}$. (b) Cumulative: $0.5,0.8,0.95,1.0$. The first cumulative sum to exceed 0.7 is 0.8, so the nucleus is the first two tokens $\{0.5,0.3\}$, renormalised to $\{0.625,0.375\}$. In this case top-2 and top-$p$ at $p=0.7$ coincide.

13. **(Q13)** Verify the BERT-base parameter count starting from $H=768$, $h=12$, $n=12$ layers, $|\mathcal V|=30{,}000$, max length 512.

    **Solution.**
    - Embeddings: $30{,}000\cdot768+512\cdot768+2\cdot768\approx23.4$ M.
    - Attention per layer: $3\cdot768\cdot64\cdot12+768\cdot768\approx1.77+0.59=2.36$ M.
    - FFN per layer: $2\cdot768\cdot3072+3072+768\approx4.72$ M.
    - Total per layer: $\approx7.08$ M; for 12 layers: $\approx85$ M.
    - Grand total: $23.4+85\approx108$ M. Paper reports $\approx110$M (the difference is LayerNorm and MLM-head parameters).

14. **(Q14)** Suppose at $t=1$ the logits are $u=(4.0,3.0,2.0,1.0)$. Compute the probabilities under (a) $T=1$ and (b) $T=0.5$.

    **Solution.** (a) $\exp(u)=(54.6,20.1,7.4,2.7)$, sum $\approx84.8$. Probabilities $\approx(0.644,0.237,0.087,0.032)$.
    (b) $u/T=(8,6,4,2)$, $\exp\approx(2981,403,54.6,7.4)$, sum $\approx3446$. Probabilities $\approx(0.865,0.117,0.016,0.002)$.
    **Note how lower $T$ has sharpened the distribution onto the top choice.**

15. **(Q15)** A beam search with $k=4$, $|\mathcal V|=50{,}000$, and length $T=20$. Compare its cost (in score evaluations) with greedy.

    **Solution.**
    - Greedy: $|\mathcal V|\cdot T=50{,}000\cdot20=10^6$ evaluations.
    - Beam: $k\cdot|\mathcal V|\cdot T=4\cdot50{,}000\cdot20=4\times10^6$.
    - Beam costs $4\times$ more in score computation, plus $k$ parallel forward passes through the model.
