<!-- PDF p.21 / printed p.20 -->

# 2 Week 2 — Completing the Transformer: Decoder, Masked & Cross-Attention, Positional Encoding, LayerNorm, Residuals, Training

**Chapter goals.** By the end of this chapter you will be able to:
1. Explain the three sub-layers of the Transformer decoder and the role of each.
2. Understand *teacher forcing* and why it makes parallel training possible.
3. Derive and implement *masked* self-attention using $-\infty$ masking.
4. Explain encoder–decoder (*cross*) attention and contrast it with self-attention.
5. Derive the sinusoidal positional encoding and explain its design choices.
6. Distinguish *Batch Normalization* from *Layer Normalization* and explain why Transformers use the latter.
7. Trace the residual “Add & Norm” wiring, count the decoder parameters, and assemble the full Transformer architecture.

## 2.1 Where We Are and Where We Are Going

In Week 1 we built the encoder: word embeddings → multi-head self-attention → feed-forward → stack of $N=6$. The encoder produces a sequence of context-aware vectors $e_1,\ldots,e_T$, each in $\mathbb{R}^{512}$. Now we ask:

> **Intuition**
>
> **How does the decoder use these encoder outputs to actually *generate* a target sentence one word at a time, in parallel during training and autoregressively during inference?**

The decoder differs from the encoder in three important ways:
1. It has **three** sub-layers (masked self-attention, cross-attention, FFN) instead of two.
2. Its self-attention is **masked** so that position $t$ cannot peek at positions $>t$ during training.
3. It uses **cross-attention**: its queries come from the decoder, but the keys and values come from the encoder. This is how the encoder’s information flows into the decoder.

## 2.2 Decoder Stack at a Glance

[Figure: A decoder layer repeated $N=6$ times takes shifted-right output ⟨Go⟩, Naan, transformar, … through masked multi-head self-attention, multi-head cross attention receiving $e_{1:T}$ from the encoder, and a feed-forward network, to the next layer / linear+softmax.]

The decoder layer is wrapped (just like the encoder) with residual connections and layer normalisation around each sub-layer; we will study those shortly.

<!-- PDF p.22 / printed p.21 -->

## 2.3 Teacher Forcing — Why the Decoder Sees the Target

If you look at the decoder during training, it receives the *ground-truth* target sentence (shifted right) as input. *Why?*

### 2.3.1 The naive alternative: free running

We could feed the decoder its own previous prediction at every step. If at step 1 the decoder predicts “D” (wrong, instead of “Naan”), then at step 2 it conditions on “D” and most likely emits another wrong word, and the error compounds. This is called *exposure bias* or *error accumulation*.

### 2.3.2 The fix: Teacher Forcing

> **Definition**
>
> **Teacher Forcing** feeds the *ground-truth* previous token to the decoder at each step during training, regardless of what the decoder predicted. The decoder is “forced” to follow the teacher’s example word for word.

> **Intuition**
>
> You are learning to write a translation. Two ways:
> - **Free running:** You write word by word, never seeing the correct answer until the end. One early mistake derails the whole sentence.
> - **Teacher forcing:** After each word, the teacher tells you the correct previous word. Even if you slipped, the next word is conditioned on the right history. You learn faster and more stably.

> **Remember This**
>
> Teacher forcing enables **parallel training** of the decoder. Because the inputs at all positions are the ground-truth tokens (known in advance), the decoder can compute all output positions in *one* forward pass — not sequentially. This is critical: without teacher forcing, training a Transformer would be as slow as an RNN.

> **Common Mistake**
>
> **Mistake:** “Teacher forcing is used at inference too.”
> **Fix:** No. At inference, there *is* no ground-truth. The decoder runs autoregressively: it predicts $y_1$, feeds it back, predicts $y_2$, etc. The training/inference mismatch caused by teacher forcing is one motivation behind techniques like *scheduled sampling*.

## 2.4 Masked Self-Attention — Forbidding the Future

Here is the problem with putting the full target sentence into the decoder. If at decoder position 2 we let position 2 attend to position 3, the model can simply copy the future word and “cheat” — learning nothing.

> **Definition**
>
> **Masked self-attention** prevents position $t$ from attending to any position $t'>t$ by setting those positions’ attention scores to $-\infty$ *before* the softmax. After softmax, $\exp(-\infty)=0$, so the corresponding weights are exactly zero.

<!-- PDF p.23 / printed p.22 -->

### 2.4.1 The mask matrix

Let the unmasked attention score matrix be $A=Q^\top K/\sqrt{d_k}\in\mathbb{R}^{T\times T}$ (row $i$ = scores from query $q_i$ to all keys). Define the mask matrix

$$M_{ij}=\begin{cases}0&j\leq i\\-\infty&j>i.\end{cases}\tag{9}$$

Then masked attention is

$$Z=\operatorname{softmax}(A+M)V.\tag{10}$$

Explicitly, for $T=5$:

$$M=\begin{bmatrix}0&-\infty&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty&-\infty\\0&0&0&-\infty&-\infty\\0&0&0&0&-\infty\\0&0&0&0&0\end{bmatrix}.$$

The lower triangle (including the diagonal) is $0$ — those positions are visible. The upper triangle is $-\infty$ — those positions are forbidden.

> **Formula Insight**
>
> Adding $-\infty$ before softmax is the cleanest way to “delete” those entries: the softmax denominator naturally re-normalises over only the visible positions. Numerically, in PyTorch one usually adds a very large negative number like $-10^9$ rather than literal $-\infty$ to avoid NaN from $0\cdot\infty$.

> **Quiz Trap**
>
> **Trap:** “Masking can be applied to the value vectors $V$.”
> **Reality:** Masking is applied to the *scores*, before softmax. Masking $V$ would not zero out the attention; it would alter the values, which is wrong.

### 2.4.2 Encoder also uses masking — for padding

The encoder’s self-attention is not causally masked (every token can see every other token). But in practice it *does* use a different mask: the **padding mask**. When sentences in a batch have different lengths, shorter ones are padded with a special `[PAD]` token, and the mask prevents real tokens from attending to padding.

### 2.4.3 What happens at inference?

The decoder cannot use teacher forcing because the ground truth is unknown. Instead:
1. Feed ⟨Go⟩ as input. Predict $y_1$.
2. Feed ⟨Go⟩, $y_1$. Predict $y_2$.
3. Continue until ⟨EOS⟩ or max length.

The triangular mask is still useful even here, because batched implementations process multiple beams simultaneously.

<!-- PDF p.24 / printed p.23 -->

## 2.5 Multi-Head Cross Attention — Where Encoder Meets Decoder

In the encoder, self-attention’s Q, K, V all come from the same sentence (source). In the decoder’s masked self-attention, Q, K, V all come from the (shifted) target. In the **cross-attention** sub-layer:

> **Definition**
>
> **Cross-Attention** (encoder–decoder attention) computes:
> - **Queries** from the decoder’s previous sub-layer output $S=(s_1,\ldots,s_T)$: $Q_2=W_{Q_2}S$.
> - **Keys and Values** from the top encoder layer’s output $E=(e_1,\ldots,e_T)$: $K_2=W_{K_2}E$, $V_2=W_{V_2}E$.
>
> The attention itself is the same scaled dot-product:
> $$Z=\operatorname{softmax}\left(\frac{Q_2K_2^\top}{\sqrt{d_k}}\right)V_2.$$
> No causal mask is needed (the decoder can look at any encoder position — the source sentence is fully available).

> **Intuition**
>
> Cross-attention is the spiritual heir of Bahdanau attention from Week 1. The decoder *queries* the encoder’s representations and pulls in relevant context for the next target word. “Which source word should I look at to translate the next target word?”

### 2.5.1 Why use the top encoder’s output for *all* decoder layers?

In the original Transformer, every decoder layer’s cross-attention reads from the *same* encoder output (the top layer). This keeps the architecture simple. Variants (e.g., some recent multi-source models) allow different decoder layers to look at different encoder layers.

### 2.5.2 Self vs Cross Attention — a comparison table

| | Self-Attention | Cross-Attention |
|---|---|---|
| Source of Q | same sequence | decoder sub-layer output |
| Source of K, V | same sequence | encoder output |
| Used in | encoder & decoder | decoder only |
| Causal mask? | decoder yes; encoder no | no (source is fully visible) |
| Padding mask? | yes | yes (pad source positions) |
| Role | contextualise within a sentence | condition target on source |

## 2.6 Decoder Output: Linear + Softmax over Vocabulary

The top decoder layer produces a sequence of vectors $o_{1:T}\in\mathbb{R}^{T\times512}$. To turn each into a probability over the vocabulary $V$, we apply a linear projection $W_D\in\mathbb{R}^{512\times|V|}$ followed by softmax:

$$p(y_t=w\mid y_{<t},x)=\operatorname{softmax}(o_tW_D)_w.\tag{11}$$

<!-- PDF p.25 / printed p.24 -->

### 2.6.1 Why $W_D$ is parameter-heavy

Suppose $|V|=37,000$ (a typical BPE vocabulary). Then $W_D$ has $512\cdot37,000\approx1.9\times10^7\approx19$ million parameters. This *single matrix* accounts for $\approx19$ out of the $\approx65$ million parameters of the base Transformer (i.e., about 30% of the total!).

> **Research / Industry Insight**
>
> This observation motivates **weight tying**: share parameters between the input embedding matrix and $W_D^\top$ (since both map between $\mathbb{R}^{512}$ and $\mathbb{R}^{|V|}$). Weight tying cuts parameters and often improves BLEU. It is standard in GPT-2, GPT-3, and modern LLMs.

## 2.7 Counting Decoder Parameters Per Layer

Per decoder layer (base config, $d=512,h=8,l=2048$):
- Masked MHA: $\approx1$M (same as encoder MHA).
- Cross-attention MHA: $\approx1$M.
- FFN: $\approx2$M.
- **Per-layer total:** $\approx4$M (vs $\approx3$M per encoder layer).

With $N=6$ layers, decoder $\approx24$M. Add encoder $\approx18$M, plus embeddings and $W_D$, total $\approx65$M parameters — this is the famous Transformer “base” size.

> **Exam Tip**
>
> **Frequently asked:** per-layer parameter count.
> - Encoder layer: $\approx3$M (MHA 1M + FFN 2M).
> - Decoder layer: $\approx4$M (masked MHA 1M + cross-attn 1M + FFN 2M).

## 2.8 Positional Encoding — Reintroducing Order

### 2.8.1 Why we need it at all

> **Intuition**
>
> Self-attention treats the input as a *set* of vectors, not a sequence. If you shuffle the input tokens, the output set is the same (just shuffled). That is, self-attention is *permutation equivariant*. But “**dog bites man**” and “**man bites dog**” have very different meanings! We must inject position information *somewhere*.

In RNNs, position was encoded implicitly in the hidden state (you must walk through positions in order). In Transformers, every token is processed simultaneously, so we add a *positional encoding* $p_j$ to the word embedding $h_j$:

$$h'_j=h_j+p_j.\tag{12}$$

### 2.8.2 Candidate ideas (and why they fail)

- **Constant vector $p_j=(j,j,\ldots,j)$:** as $j$ grows, $p_j$ dominates the embedding $h_j$, drowning out semantics.
- **One-hot vectors over positions:** the Euclidean distance between any two one-hot vectors is always $\sqrt2$, so position 1 looks “equally far” from position 2 and from position 100. No notion of *relative* closeness.

<!-- PDF p.26 / printed p.25 -->

- **Learned position embeddings:** works fine up to the trained max length, but cannot generalise to longer sequences. (BERT and GPT use this anyway, since in practice their max length is fixed.)

### 2.8.3 The sinusoidal solution

Vaswani et al. proposed:

$$\begin{aligned}PE(j,2i)&=\sin\left(\frac{j}{10000^{2i/d_{\mathrm{model}}}}\right),\\ PE(j,2i+1)&=\cos\left(\frac{j}{10000^{2i/d_{\mathrm{model}}}}\right),\end{aligned}\tag{13}$$

for position $j$ and embedding-dimension index $i=0,1,\ldots,255$ (since $d_{\mathrm{model}}=512$, there are 256 sine/cosine pairs).

> **Formula Insight**
>
> Reading the formula carefully:
> - Each **dimension** $i$ corresponds to a sinusoid of a different frequency.
> - Lower dimensions ($i$ small) have very short wavelength ($\approx2\pi$), encoding fine-grained position.
> - Higher dimensions ($i$ near 255) have very long wavelength (up to $2\pi\cdot10000$), encoding coarse-grained position.
> - Even dims use sin, odd dims use cos: a sine/cosine pair lets the model recover relative position via the identity $\sin(a+b)=\sin a\cos b+\cos a\sin b$.

### 2.8.4 The two key properties the design satisfies

1. **Unique pattern per position.** Each $p_j$ is a vector of 256 sine/cosine values at different frequencies — no two positions share the same vector.
2. **Distance encodes (roughly) position offset.** The Euclidean distance $\|p_j-p_{j+\delta}\|$ depends on $\delta$ (the relative offset) and is roughly monotonically increasing in $|\delta|$. By contrast, one-hot distance is constant $\sqrt2$ for all pairs.

The lecturer demonstrated this graphically: for one-hot, the distance from position 0 to positions 1, 2, …, 100 is a flat horizontal line at $\sqrt2$ (uninformative). For sinusoidal PE, the distance grows roughly like log-distance with position offset (informative).

### 2.8.5 Linear-combination property (relative position learnable)

A beautiful property of sinusoidal encoding:

$$PE(j+\delta)=R_\delta\cdot PE(j)\tag{14}$$

for some matrix $R_\delta$ that depends only on $\delta$, not on $j$. This means the model can learn a linear function (in the $W_Q,W_K$ projections) that lets attention weights depend on the *relative* position $\delta$, not just absolute positions.

<!-- PDF p.27 / printed p.26 -->

> **Research / Industry Insight**
>
> This linear-combination property inspires **relative positional encodings** (Shaw et al., 2018) used in T5 and **rotary positional embeddings** (RoPE, Su et al. 2021) used in LLaMA, PaLM, Mistral, and most modern LLMs. RoPE encodes position by *rotating* the query/key vectors instead of *adding* a positional vector.

### 2.8.6 Visualisation in your head

For each position $j$ along the horizontal axis and each dimension $i$ along the vertical axis, $PE(j,i)$ produces a striped pattern: bottom rows oscillate rapidly with $j$, top rows oscillate slowly. The matrix is the famous “positional encoding heatmap”.

## 2.9 Training the Transformer — Residual Connections & Normalization

Counting layers loosely (each MHA + FFN block has $\sim3$ “layers”), the full Transformer with $N=6$ encoder and $N=6$ decoder layers is effectively $\sim42$ layers deep. Two questions become urgent:
1. How do we ensure gradients flow through this depth? → **Residual connections.**
2. How do we keep activations well-scaled so training converges fast? → **Layer normalization.**

### 2.9.1 Residual “skip” connections

> **Definition**
>
> A **residual connection** around a sublayer $\mathcal F$ adds the input to the sublayer output:
> $$\mathrm{out}=x+\mathcal F(x).$$
> This was popularised by ResNet (He et al., 2016). In Transformers, it is applied around *every* sub-layer (each MHA, each FFN, each cross-attention).

> **Intuition**
>
> Without a skip path, the only way information from layer 1 reaches the output is through 40+ non-linear transformations — a long, easily-decaying chain. The skip path gives a “highway” for both forward information and backward gradients. The sublayer learns the *residual* (small correction) on top of the identity.

### 2.9.2 Batch Normalization (refresher)

Batch Norm normalises activations *across the batch dimension* for each feature independently. Given a batch of $m$ activations for neuron $i$ in layer $l$:

$$\mu_i=\frac1m\sum_{j=1}^m x_i^j,\tag{15}$$
$$\sigma_i^2=\frac1m\sum_{j=1}^m(x_i^j-\mu_i)^2,\tag{16}$$
$$\hat{x}_i=\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\epsilon}},\tag{17}$$
$$\hat{y}_i=\gamma\hat{x}_i+\beta.\quad\text{(learnable scale and shift)}\tag{18}$$

<!-- PDF p.28 / printed p.27 -->

### 2.9.3 Why Batch Norm fails for Transformers

- **Tiny batches kill estimation:** for sequence-of-sequences workloads (long sentences, large model), batch size is often small. With $m=1$, $\sigma_i=0$ and BN becomes a no-op.

> [possible-error p.28: With a single batch value, the displayed training-time normalization gives zero normalized activation and output $\beta$, not the identity.]

- **Variable sequence lengths:** different sentences have different lengths and different distributions of activations. Averaging across the batch is meaningless when the underlying populations differ.
- **Empirical evidence:** BN underperforms in NLP tasks. The PowerNorm paper (Shen et al., 2020) systematically studied this and proposed a fix.

### 2.9.4 Layer Normalization

> **Definition**
>
> **Layer Normalization** normalises activations *across the feature dimension*, independently for each sample (and each sequence position). For a vector $x\in\mathbb R^H$ (the hidden representation of one token):
> $$\mu=\frac1H\sum_{i=1}^H x_i,\quad \sigma^2=\frac1H\sum_{i=1}^H(x_i-\mu)^2,$$
> $$\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\quad \hat{y}_i=\gamma_i\hat{x}_i+\beta_i.$$

Crucially, this normalisation **does not depend on the batch.** It works the same whether $m=1$ or $m=1024$, perfect for variable-length sequences.

### 2.9.5 BN vs LN — a comparison table

| | Batch Normalization | Layer Normalization |
|---|---|---|
| Normalises over | batch (across $m$ samples) | features (across $H$ neurons) |
| Statistics depend on batch? | yes | no |
| Works with batch size 1? | no (variance is 0) | yes |
| Sensitive to sequence length? | yes (different lengths per sample) | no |
| Default in CNN/image | yes | no |
| Default in Transformer/NLP | no | yes |
| Learnable params per layer | $2H$ ($\gamma,\beta$) | $2H$ ($\gamma,\beta$) |

### 2.9.6 Putting it together: “Add & Norm”

The complete sublayer recipe is:

$$\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).\tag{19}$$

<!-- PDF p.29 / printed p.28 -->

This is called the **Post-LN** (post-normalisation) configuration — the original Transformer. A variant called **Pre-LN** (used by GPT-2, BERT-large, and most modern LLMs) places LayerNorm *before* the sublayer:

$$x+\operatorname{Sublayer}(\operatorname{LayerNorm}(x)).\tag{20}$$

> [possible-error p.29: Original BERT-large uses post-layer normalization, not Pre-LN.]

Pre-LN is more stable to train and usually doesn’t require a learning-rate warmup.

> **Research / Industry Insight**
>
> The original Post-LN Transformer requires careful warmup (linear ramp of learning rate for the first 4000 steps in the paper) to prevent gradient explosion. Pre-LN was found to be much more forgiving (Xiong et al., 2020). Today, most production LLMs use Pre-LN or its variants (e.g., RMSNorm in LLaMA).

## 2.10 Putting It All Together — The Full Transformer

[Figure: Encoder input embedding plus positional encoding passes through multi-head attention, Add & Norm, feed forward, and Add & Norm; encoder output $e_{1:T}$ supplies keys and values to decoder cross-attention. Shifted output embedding plus positional encoding enters masked multi-head attention, Add & Norm, cross-attention, Add & Norm, feed forward, Add & Norm, linear and softmax to output probabilities.]

### 2.10.1 Data flow in 6 steps

1. Source tokens → input embeddings $h_1,\ldots,h_T\in\mathbb R^{512}$.
2. Add positional encodings $h_j+p_j$.
3. Pass through $N=6$ encoder layers → $e_1,\ldots,e_T$.
4. Target tokens (shifted right) → output embeddings, add PE, pass through masked self-attention.
5. Cross-attention: query from decoder, K and V from encoder output $e_{1:T}$.
6. FFN, $N=6$ times, then linear projection to $\mathbb R^{|V|}$, softmax. Done.

## 2.11 Other Training Tricks (briefly mentioned in the paper)

- **Label smoothing.** Instead of using a one-hot target, use $\tilde y=(1-\varepsilon)e_w+\varepsilon/|V|$ for $\varepsilon\approx0.1$. This penalises overconfident predictions and improves BLEU.
- **Learning-rate warmup.** The paper schedules $\eta_t=d_{\mathrm{model}}^{-1/2}\cdot\min(t^{-1/2},t\cdot t_{\mathrm{warmup}}^{-3/2})$ with $t_{\mathrm{warmup}}=4000$. Without warmup, Post-LN training is unstable.

<!-- PDF p.30 / printed p.29 -->

- **Dropout.** Applied to (i) attention weights, (ii) FFN, and (iii) the sum embedding+PE, with rate 0.1.
- **Adam optimizer** with $\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$.

## 2.12 Common Mistakes & Quiz Traps — Week 2

> **Common Mistake**
>
> **Mistake 1.** “The decoder is just an encoder with a mask.”
> **Fix:** A decoder layer has *three* sub-layers (masked self-attention, cross-attention, FFN), not two. The middle cross-attention block is the crucial bridge to the encoder.

> **Common Mistake**
>
> **Mistake 2.** “Positional encoding is added to the output of self-attention.”
> **Fix:** It is added to the *input* embeddings (before the first attention). Adding it post-attention would not give attention any positional information to use.

> **Common Mistake**
>
> **Mistake 3.** “LayerNorm normalises across the batch.”
> **Fix:** LayerNorm normalises across the *feature* dimension (per token, per sample). Batch dimension is irrelevant. This is the whole point of LN vs BN.

> **Quiz Trap**
>
> **Trap.** “Cross-attention requires a causal mask because the future shouldn’t leak.”
> **Reality:** The future leak we worry about is in the *target*, not the source. The source sentence is fully visible to the decoder — so cross-attention has *no* causal mask. Only padding masks (if applicable).

> **Quiz Trap**
>
> **Trap.** “Sinusoidal PE allows the model to extrapolate to any sentence length.”
> **Reality:** In theory yes, in practice the model has only seen positions up to its training max, and quality degrades beyond. RoPE and ALiBi handle extrapolation better.

## 2.13 Memory Tricks — Week 2

> **Memory Trick**
>
> **“Decoder has Three: Mask, Cross, FFN.”**
> The three decoder sub-layers in order: Masked self-attention → Cross-attention → FFN.

> **Memory Trick**
>
> **“$-\infty$ before softmax = exact zero after softmax.”**
> Masking is done *before* softmax by adding a $-\infty$ matrix to the scores. After softmax, those positions become exact 0.

<!-- PDF p.31 / printed p.30 -->

> **Memory Trick**
>
> **“BN normalises across rows, LN across columns”** (if rows = batch, columns = features). LayerNorm wins when batch is small or sequence length varies — exactly the NLP setting.

> **Memory Trick**
>
> **“Sin for even, Cos for odd”**, frequencies geometric from 1 to $1/10000$. Adjacent dims = a sine/cosine pair = a complex exponential, which is what gives the relative-position property.

## 2.14 Practical Applications

- **Translation (Google Translate, NLLB, MarianMT):** the original use case.
- **Summarisation (T5, BART, Pegasus):** same encoder-decoder pattern, longer outputs.
- **Speech recognition (Whisper):** audio features as encoder input, transcription as decoder output.
- **Image captioning (BLIP, CoCa):** image patches encoded → text decoded.
- **Decoder-only LLMs (GPT, LLaMA):** drop the encoder entirely; rely on masked self-attention only. This is the dominant modern paradigm.

## 2.15 Quick Revision — Week 2 in One Page

> **Quick Revision**
>
> **Decoder structure.** Three sub-layers per layer, each wrapped with Add&Norm: masked self-attention → cross-attention → FFN. Stacked $N=6$ times.
>
> **Teacher forcing.** Feed ground-truth shifted-right target to decoder during training. Enables parallel training. Requires causal mask to avoid cheating.
>
> **Masked self-attention.** $Z=\operatorname{softmax}(A+M)V$ where $M_{ij}=-\infty$ for $j>i$. Upper triangle is $-\infty$, lower triangle (incl. diagonal) is 0.
>
> **Cross-attention.** Q from decoder, K and V from *top* encoder layer. No causal mask.
>
> **Decoder output.** Linear $W_D\in\mathbb R^{d\times|V|}$ + softmax. $\sim19$M parameters.
>
> **Positional encoding (sinusoidal).** $PE(j,2i)=\sin(j/10000^{2i/d})$, $PE(j,2i+1)=\cos(j/10000^{2i/d})$. Added to input embeddings. Encodes both absolute and (via linear combinations) relative position.
>
> **Layer Normalization.** Normalises across the feature dimension, per token: $\hat{x}=(x-\mu)/\sqrt{\sigma^2+\epsilon}$, then $\gamma\hat{x}+\beta$. Works with batch size 1; LayerNorm is the right choice for NLP.
>
> **Residual connection.** $\mathrm{out}=\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$. Allows gradient flow through deep stack.
>
> **Parameter counts (base config).** Encoder layer $\approx3$M, decoder layer $\approx4$M, $W_D\approx19$M, total $\approx65$M.

## 2.16 Practice Questions — Week 2

### Multiple Choice

1. **(Q1)** In the Transformer decoder, masked self-attention is achieved by:

   (A) Multiplying the value matrix by a triangular mask.

   (B) Adding $-\infty$ to the upper triangle of the score matrix, before softmax.

<!-- PDF p.32 / printed p.31 -->

   (C) Dropping out attention weights with probability 0.5.

   (D) Using a different scaling factor for each position.

   **Solution.** (B). Mask the *scores* pre-softmax. $\exp(-\infty)=0$ in the denominator and numerator gives exact-0 weight on future positions.

2. **(Q2)** Cross-attention in the Transformer decoder uses:

   (A) Q, K, V all from the encoder.

   (B) Q, K, V all from the decoder.

   (C) Q from encoder; K, V from decoder.

   (D) Q from decoder; K, V from encoder.

   **Solution.** (D). The decoder asks a question; the encoder supplies the catalog and content.

3. **(Q3)** Which of the following is *not* a reason to use LayerNorm over BatchNorm in Transformers?

   (A) LayerNorm is independent of batch size.

   (B) LayerNorm handles variable sequence lengths better.

   (C) LayerNorm has fewer parameters than BatchNorm.

   (D) BatchNorm with small batches has noisy statistics.

   **Solution.** (C) is the odd one out — BN and LN both have $2H$ learnable parameters ($\gamma,\beta$).

4. **(Q4)** The sinusoidal positional encoding pairs sin and cos for adjacent dimensions *primarily* so that:

   (A) the embedding has zero mean,

   (B) relative position can be expressed as a linear transformation of absolute position,

   (C) the model can be trained with smaller batch sizes,

   (D) the maximum sentence length can be extended.

   **Solution.** (B). The pair acts like a complex exponential and rotates by a fixed amount when position shifts by $\delta$, which is a linear map of the pair.

5. **(Q5)** Teacher forcing in Transformer training means:

   (A) The teacher penalises the model for wrong answers more strongly.

   (B) The decoder receives its own previous prediction as input.

   (C) The decoder receives the ground-truth previous token as input.

   (D) The encoder forces specific attention patterns.

   **Solution.** (C).

### Conceptual / Short Answer

6. **(Q6)** Why must the decoder’s self-attention be masked but the encoder’s not?

   **Answer.** The encoder reads a fully-given source sentence; all positions are known. The decoder must *generate* the target one step at a time, so position $t$ must not see positions $>t$

<!-- PDF p.33 / printed p.32 -->

during training, or it would trivially “cheat” by copying the future target token. Masking enforces causality.

7. **(Q7)** Compute the maximum and minimum value of any element of the sinusoidal PE.

   **Answer.** Each entry is a single sin or cos value, so it lies in $[-1,1]$. This bounded range is one reason PE can be *added* to embeddings without overwhelming them (unlike a raw position scalar that grows unboundedly).

8. **(Q8)** Show the dimensions inside one cross-attention sub-layer when $T_{\mathrm{src}}=T_{\mathrm{tgt}}=T$, $d=512$, $h=8$.

   **Answer.** Per head: $Q_2\in\mathbb R^{T\times64}$ (from decoder), $K_2,V_2\in\mathbb R^{T\times64}$ (from encoder). Attention matrix: $Q_2K_2^\top\in\mathbb R^{T\times T}$. Head output: $T\times64$. Concat heads: $T\times512$. Output $W^O\in\mathbb R^{512\times512}$ produces $T\times512$.

9. **(Q9)** A student says “LayerNorm should use the same $\gamma,\beta$ for all tokens in a sequence.” Correct or not?

   **Answer.** Correct. LayerNorm’s $\gamma,\beta\in\mathbb R^H$ are per-feature parameters, applied identically to every token in the sequence (and every example in the batch). Each token is normalised independently, but the affine transform afterwards is shared.

### Numerical / Derivation

10. **(Q10)** Given attention scores $a=(2,1,5,3)$ and mask $m=(0,0,-\infty,-\infty)$, compute the softmax outputs.

    **Solution.** Effective scores: $(2,1,-\infty,-\infty)$. exp: $(e^2,e^1,0,0)\approx(7.389,2.718,0,0)$, sum $\approx10.107$. softmax $\approx(0.731,0.269,0,0)$. Position 3 and 4 are exactly zero, position 1 gets the largest weight as before.

11. **(Q11)** Compute the number of parameters added by LayerNorms in a single encoder layer (after MHA and after FFN), with $H=512$.

    **Solution.** Each LN has $2H=1024$ parameters ($\gamma,\beta$). Two LNs per encoder layer $\Rightarrow2048$ parameters per layer. Negligible compared to MHA ($\approx1$M) and FFN ($\approx2$M).

12. **(Q12)** Verify that for any position $j$ the sum of squares $\sum_{i=0}^{d/2-1}(\sin^2(\omega_i j)+\cos^2(\omega_i j))=d/2$, where $\omega_i=10000^{-2i/d}$.

    **Answer.** By $\sin^2+\cos^2=1$, each $i$ contributes exactly 1. There are $d/2$ pairs (for $d=512$, that is 256 pairs), so the total is $d/2=256$. *Hence $\|PE(j)\|_2=\sqrt{d/2}$ is the same for every position $j$ — norms encode no positional information; the angles do.*

13. **(Q13)** The base Transformer has $\approx65$M parameters. Roughly break this down by component (encoder, decoder, embeddings, $W_D$).

    **Solution (approximate).**
    - Encoder: $6\cdot3$M $\approx18$M.
    - Decoder: $6\cdot4$M $\approx24$M.
    - Source + target embedding matrices: $\sim2\cdot37000\cdot512/10^6\approx38$M *but* weight tying reduces this to one $\approx19$M chunk.
    - Output projection $W_D$ (shared with embedding via weight tying): already counted.
    - Total: $\approx18+24+19\approx61$M (within rounding of the paper’s 65M).
