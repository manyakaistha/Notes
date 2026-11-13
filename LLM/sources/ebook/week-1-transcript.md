<!-- PDF p.4 / printed p.3 -->

# Part I — Complete Notes (Weeks 1–12)

# 1 Week 1 — From Sequence Models to Transformers: Multi-Headed Self-Attention & Cross-Attention

> **Chapter goals.** By the end of this chapter you will be able to:
> 1. Explain *why* we moved from RNN seq2seq to Transformers.
> 2. Derive scaled dot-product self-attention from first principles.
> 3. Understand the role of the query, key, value matrices ($W_Q,W_K,W_V$).
> 4. Justify and compute multi-head attention.
> 5. Trace the data flow through one encoder layer, count its parameters, and explain every dimension.

## 1.1 Setting the Stage: What is a Sequence Model?

A *sequence model* processes an ordered list of tokens $x_1,x_2,\ldots,x_T$ and produces either a single output (e.g. a sentiment label) or another ordered list of tokens $y_1,y_2,\ldots,y_{T'}$ (e.g. a translation). Most early NLP successes used *recurrent neural networks* (RNNs) — including vanilla RNNs, LSTMs, and GRUs — to perform this mapping.

> **Definition**
>
> A **Sequence-to-Sequence (seq2seq) model** consists of two RNNs:
> - An **Encoder** that reads the source sequence and compresses it into a fixed-length vector (the *context* or *thought* vector).
> - A **Decoder** that, conditioned on this context vector, generates the target sequence one token at a time.
>
> Introduced by Sutskever et al. (2014) and Cho et al. (2014).

### 1.1.1 The Encoder-Decoder Picture

[Figure: An encoder RNN processes “I enjoyed the movie transformers” through states $h_0$–$h_5$; $h_5=s_0$ initializes a decoder RNN producing the target sequence. The output labels overlap.]

The final encoder state $h_5$ is called the **context vector, thought vector,** or **annotation**. It is supposed to summarise the *entire* input sentence into a single vector — a heavy demand for long sentences.

### 1.1.2 The Two Critical Limitations of RNN-Based seq2seq

1. **The information bottleneck.** A single fixed-size vector $h_T$ has to encode the whole sentence. The longer the sentence, the more crushing the bottleneck. Empirically, BLEU scores degrade sharply once sentence length exceeds $\sim 30$ tokens.
2. **No parallelism along the time axis.** To compute $h_t$ we need $h_{t-1}$, which needs $h_{t-2}$, etc. This makes training painfully slow on modern GPUs, which prefer wide parallel matrix multiplies. Vanishing and exploding gradients only worsen the situation.

<!-- PDF p.5 / printed p.4 -->

> **Intuition**
>
> Imagine you must summarise a 1000-word essay in a 5-word tweet, hand the tweet to a friend, and ask them to reconstruct the essay sentence-by-sentence. That’s the seq2seq decoder’s job using only $h_T$. Painful, right? **Attention** fixes this by giving the decoder *the entire essay*, not just the tweet — and letting the decoder *re-read* the most relevant parts for each output word.

## 1.2 The Attention Mechanism — A Quick Tour (Bahdanau, 2015)

The cure for the bottleneck is conceptually simple:

> **Intuition**
>
> **Don’t crush all encoder states into one vector. Keep them all, and at every decoder step, blend them with weights $\alpha_{ti}$ that depend on what the decoder needs *right now*.**

Formally, for decoder time-step $t$, define a context vector

$$\mathbf c_t=\sum_{i=1}^n \alpha_{ti}\mathbf h_i. \tag{1}$$

where $n$ is the input length, $\mathbf h_i$ is the $i$-th encoder hidden state, and $\alpha_{ti}$ is the attention weight on input position $i$ when producing output $y_t$. The weights are computed by softmax-normalising an *alignment score*:

$$\alpha_{ti}=\frac{\exp(\operatorname{score}(\mathbf s_{t-1},\mathbf h_i))}{\sum_{i'=1}^{n}\exp(\operatorname{score}(\mathbf s_{t-1},\mathbf h_{i'}))}. \tag{2}$$

A common Bahdanau-style scoring function uses a tiny feed-forward network:

$$\operatorname{score}(\mathbf s_{t-1},\mathbf h_j)=V_a^\top\tanh(U_{att}\mathbf s_{t-1}+W_{att}\mathbf h_j). \tag{3}$$

> **Formula Insight**
>
> Equation (3) has three vectors — the decoder state $s$, the encoder state $h$, and the score vector $v$ — combined by two linear transformations ($U_{att},W_{att}$), one non-linearity (tanh), and a final dot product with $V_a$. The output is a single scalar: “how relevant is encoder position $j$ for predicting the $t$-th output token?”

### 1.2.1 Decoding step-by-step with attention

At decoder step $t$, the recipe is:

1. Use the previous decoder state $\mathbf s_{t-1}$ to compute scores against *every* encoder state $\mathbf h_1,\ldots,\mathbf h_n$.
2. Softmax-normalise to get $\alpha_{t,1},\ldots,\alpha_{t,n}$.
3. Form $\mathbf c_t=\sum_i\alpha_{ti}\mathbf h_i$.
4. Feed $\mathbf c_t$ together with the previous decoder hidden state and the previously emitted token into the decoder RNN to compute $\mathbf s_t$ and emit $y_t$.

<!-- PDF p.6 / printed p.5 -->

### 1.2.2 What does the matrix $\alpha_{ti}$ look like?

Stacked over all output positions, $\alpha$ is a soft alignment matrix: high values are roughly diagonal (input/output positions align), but they pick up real linguistic facts — e.g. when translating English $\to$ Tamil, “transformers” aligns with the Tamil word at a non-monotonic position because of word-order differences. This was the breakthrough of Bahdanau et al. (2015): attention *discovers* alignments without supervision.

### 1.2.3 A crucial observation: attention parallelises

Look again at Eq. (2). At time $t$, $\mathbf s_{t-1}$ is fixed, and $\mathbf h_1,\ldots,\mathbf h_n$ are all known. Therefore $\alpha_{t1},\ldots,\alpha_{tn}$ can be computed *in parallel*. So can $\mathbf c_t$. That is, attention is intrinsically a parallel operation.

> **Remember This**
>
> **The big takeaway from Bahdanau attention:** the attention layer itself is parallelisable. The bottleneck is only the RNN that produces $\mathbf h_i$ and $\mathbf s_t$ — *not* the attention block.
>
> This question motivates: *What if we replace the RNN with something that is also parallelisable?* The answer is the Transformer.

## 1.3 The Wishlist that Births the Transformer

Summarising the limitations and desired properties, the “wishlist” is:

- Keep attention (it solved the bottleneck).
- Eliminate recurrence (to allow parallel training).
- Avoid vanishing / exploding gradients of deep RNNs.
- Allow each token to look at every other token in $O(1)$ path length (instead of $O(T)$ in RNNs).

The 2017 paper of Vaswani et al. — *Attention is All You Need* — delivers exactly this. The model is called the **Transformer**.

> **Transformer at a Glance**
>
> - Encoder: Word Embedding $\to$ Self-Attention $\to$ FFN (stacked $N=6$ times).
> - Decoder: Self-Attention $\to$ Encoder-Decoder Attention $\to$ FFN (stacked $N=6$ times).
> - All computation is matrix multiplication and softmax. No recurrence.

## 1.4 Self-Attention — The Heart of the Transformer

The key new idea is to replace the RNN encoder by a layer that lets every word attend to every other word *in the same sentence*. Because the words are attending to themselves (the same sequence), this is called **self-attention**.

### 1.4.1 Why “self”? A motivating example

Consider two sentences:

*“The animal didn’t cross the street because it was too tired”*

*“The animal didn’t cross the street because it was congested”*

<!-- PDF p.7 / printed p.6 -->

In the first, “it” refers to “animal”. In the second, “it” refers to “street”. *Resolving the referent* requires looking at *other words in the same sentence*. Self-attention is the mechanism that computes, for every word, a contextualised representation by weighting all the other words.

> **Intuition**
>
> For each word in the sentence, self-attention asks: “Which other words in this same sentence should I pay attention to, and how much?” — and produces a new vector that is a weighted mix of all the words’ value vectors.

### 1.4.2 The Goal Stated as a Table

Imagine an $8\times8$ table where row $i$ holds the attention scores of word $i$ towards every word $j$. In the running example, the row of “it” should have a large value in the column “animal” (or “street” depending on context). Filling this table for all rows is what self-attention does.

Importantly, all the input vectors (word embeddings) $h_1,\ldots,h_T$ are *available from the start* — unlike the seq2seq decoder where the decoder state is computed one step at a time. This means **all rows of the attention table can be computed simultaneously.**

## 1.5 Q, K, V — The Three Roles a Word Plays

Recall the Bahdanau score (Eq. 3) needs three vectors: $(s,h,v)$ where $s$ is the decoder state (the “query”), $h$ is the encoder state (the “key”), and $v$ is the same encoder state re-used as the “value”.

In self-attention, we have only *one* input: the word embedding vector $h_j$. So how do we produce three different vectors per word?

> **Solution: Three Linear Projections**
>
> For each word embedding $h_j\in\mathbb R^{d_{\mathrm{model}}}$ we produce three vectors by multiplying with three learnable matrices:
> $$q_j=W_Qh_j\quad(\text{query})\\ k_j=W_Kh_j\quad(\text{key})\\ v_j=W_Vh_j\quad(\text{value})$$
> $W_Q,W_K,W_V$ are learned during training (one set per attention head).

> **Intuition**
>
> A useful library analogy: in a library catalog system,
> - the **query** is the search term you type;
> - the **key** is the catalog index (each book has a key);
> - the **value** is the actual content of the book.
> The librarian (softmax) returns a blended summary of the values, weighted by how well their keys match your query.

### 1.5.1 Computing the first output $z_1$

1. Form $q_1$ from $h_1$ and form $k_j,v_j$ for every $j=1,\ldots,T$.
2. Compute raw scores $e_{1j}=q_1\cdot k_j$ for $j=1,\ldots,T$.
3. Softmax over $j$: $\alpha_{1j}=\operatorname{softmax}_j(e_{1j})$.

<!-- PDF p.8 / printed p.7 -->

4. Output $z_1=\sum_{j=1}^T\alpha_{1j}v_j$.

For the second output $z_2$, we use $q_2$ instead of $q_1$; every other step is identical and runs *in parallel* for all $i$.

> **Formula Insight**
>
> **Why the dot product?** The dot product $q_i\cdot k_j$ is a natural similarity measure between two vectors of the same dimension. The bigger the dot product, the more aligned (similar) $q_i$ and $k_j$ are, hence the more attention word $i$ pays to word $j$.

## 1.6 Vectorising into Matrices: One Equation Per Encoder Layer

Let $H\in\mathbb R^{d_{\mathrm{model}}\times T}$ stack all the input word embeddings as columns. Form

$$Q=W_QH\in\mathbb R^{d_q\times T},\tag{4}$$
$$K=W_KH\in\mathbb R^{d_k\times T},\tag{5}$$
$$V=W_VH\in\mathbb R^{d_v\times T}.\tag{6}$$

By convention $d_q=d_k=d_v=d_k$ (the paper uses $d_k=64$ per head).

### 1.6.1 Scaled dot-product attention

$$Z=\operatorname{softmax}\left(\frac{Q^\top K}{\sqrt{d_k}}\right)V^\top.\tag{7}$$

Let’s verify dimensions step-by-step (using the column-vector convention):
- $Q^\top\in\mathbb R^{T\times d_k}$ and $K\in\mathbb R^{d_k\times T}$, so $Q^\top K\in\mathbb R^{T\times T}$ — one similarity score per (query, key) pair.
- Dividing by $\sqrt{d_k}$ leaves the shape unchanged.
- Row-wise softmax produces a $T\times T$ matrix whose rows sum to one — these are the attention weights.
- $V^\top\in\mathbb R^{T\times d_v}$, so $Z\in\mathbb R^{T\times d_v}$ — one output vector per query.

> **Remember This**
>
> The **Vaswani** paper uses row vectors ($X\in\mathbb R^{T\times d}$), giving the more common form
> $$Z=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$
> The two formulations are transposes of each other — conceptually identical.

> [possible-error p.8: The displayed outputs in the two conventions are identical after transposing the input matrices, rather than transposes of each other.]

### 1.6.2 Why divide by $\sqrt{d_k}$?

> **Formula Insight**
>
> For random query and key vectors with components drawn i.i.d. with zero mean and unit variance, $q\cdot k=\sum_{m=1}^{d_k}q_mk_m$ has mean $0$ and variance $d_k$. Large variance pushes the softmax into saturating regions where gradients vanish. Dividing by $\sqrt{d_k}$ standardises the dot product to unit variance, keeping the softmax in a healthy regime.

<!-- PDF p.9 / printed p.8 -->

### 1.6.3 Scaled Dot-Product Attention block (visual)

[Figure: $Q$ and $K$ enter MatMul $Q^\top K$, followed by scaling by $1/\sqrt{d_k}$, softmax, and MatMul with $V$.]

## 1.7 Multi-Head Attention — Why One Head Is Not Enough

Empirically, a *single* attention head over-mixes information: it tends to average out distinct relationships (subject–verb agreement, coreference, syntactic role, etc.) into a single soup. A CNN solves this by using *many filters* per layer to learn different feature detectors. The Transformer copies this idea.

> **Definition**
>
> **Multi-Head Attention** runs $h$ scaled dot-product attentions in parallel, each with its own learned projections $\{W_Q^i,W_K^i,W_V^i\}_{i=1}^h$. The outputs of the heads are *concatenated* along the feature dimension and projected by a final linear $W^O$:
> $$\operatorname{MultiHead}(H)=W^O\operatorname{Concat}(\operatorname{head}_1,\operatorname{head}_2,\ldots,\operatorname{head}_h),\qquad \operatorname{head}_i=\operatorname{Attention}(W_Q^iH,W_K^iH,W_V^iH).$$

> [possible-error p.9: With the row-oriented output defined in Eq. (7), the output projection must multiply on the right, or the head outputs must be transposed.]

### 1.7.1 Why concatenation works without blowing up dimensions

In the original paper, $d_{\mathrm{model}}=512$ and $h=8$, so each head uses $d_k=d_q=d_v=512/8=64$. Concatenating 8 heads of width 64 gives back 512. The final $W^O\in\mathbb R^{512\times512}$ mixes information from all heads.

> **Intuition**
>
> Different heads end up specialising in different patterns: one head may track syntactic dependencies (subject ↔ verb), another may track coreference (it ↔ animal), another may track positional neighbours. The lecturer showed visualisations: a single-head model links “it” mainly to “was” (a syntactic neighbour), whereas a two-head model also links “it” to “animal” (semantic coreference) — two different patterns, two heads.

<!-- PDF p.10 / printed p.9 -->

### 1.7.2 Dimensions cheat-sheet (per encoder layer)

| Object | Shape | Comment |
|---|---|---|
| Input $X$ | $T\times512$ | sequence of word embeddings |
| $W_Q^i,W_K^i,W_V^i$ | $512\times64$ | per head, per role |
| $Q_i,K_i,V_i$ | $T\times64$ | per head outputs |
| Attention matrix | $T\times T$ | softmax over rows |
| Head output | $T\times64$ | per head |
| Concat | $T\times512$ | all 8 heads stitched |
| $W^O$ | $512\times512$ | final projection |
| MHA output | $T\times512$ | ready for the FFN |

> **Quiz Trap**
>
> **Do not confuse** $d_k$ (size of each head, e.g. 64) with $d_{\mathrm{model}}$ (size of the embedding, e.g. 512). They satisfy $h\cdot d_k=d_{\mathrm{model}}$ in the standard Transformer.

## 1.8 The Position-wise Feed-Forward Network (FFN)

After multi-head attention, every token’s vector passes through a small *point-wise* feed-forward network — the same network applied independently to each position (i.e. a $1\times1$ convolution if you prefer that lens):

$$\operatorname{FFN}(z)=\max(0,W_1z+b_1)W_2+b_2.\tag{8}$$

In the original paper, $W_1\in\mathbb R^{512\times2048}$ (expand!), and $W_2\in\mathbb R^{2048\times512}$ (project back). The hidden width $l=2048$ gives the layer expressive power to mix the attention-aggregated information.

> [possible-error p.10: The stated matrix shapes require $zW_1$, rather than $W_1z$, in the displayed FFN formula.]

> **Intuition**
>
> Think of MHA as “gathering information across the sentence” and the FFN as “thinking about each position independently with the information so far”. The two alternate, and stacking $N=6$ such pairs in the encoder gradually builds a deep contextual representation.

## 1.9 Counting Parameters in One Encoder Layer

Let $d_{\mathrm{model}}=512$, $h=8$, $d_k=d_q=d_v=64$, FFN width $l=2048$.

### 1.9.1 Multi-head attention parameters

- Per head: three projection matrices of shape $64\times512$. Parameters per head $=3\cdot64\cdot512=98,304$.
- For 8 heads: $8\cdot98,304=786,432$.
- Output projection $W^O\in\mathbb R^{512\times512}$: $512\cdot512=262,144$.
- **MHA total** $\approx1.05\times10^6$.

### 1.9.2 FFN parameters

$$512\cdot2048+2048+2048\cdot512+512\ \approx\ 2.1\times10^6.$$

### 1.9.3 Per-layer total and stack total

- Per layer: $\approx3$ million parameters.
- Stack of $N=6$ layers: $\approx18$ million parameters in the encoder.

<!-- PDF p.11 / printed p.10 -->

> **Exam Tip**
>
> A very frequent quiz question is: *“How many parameters are in one encoder layer of the Transformer (base configuration)?”* The standard answer is **“about 3 million”**, distributed as $\approx1$ M in MHA and $\approx2$ M in the FFN.

## 1.10 The Encoder Stack and Data Flow

The encoder consists of $N=6$ *identical* layers stacked vertically. Each layer reads $T\times512$ and outputs $T\times512$, so we can stack arbitrarily many. Crucially:

- Computation is **parallelised across the sequence dimension** (all $T$ token outputs of a layer are produced in one matrix multiplication).
- Computation is **not** parallel across layers: layer 2 needs the output of layer 1.

We denote the encoder’s final output sequence as $e_1,e_2,\ldots,e_T$. These will be fed to the decoder’s encoder-decoder (cross-)attention block — studied next week.

[Figure: Input embeddings $h_1,h_3,h_5$ enter encoder layer 1, followed by layer 2 and further layers through encoder layer 6, yielding $e_1,e_3,e_5$; intermediate dimensions are $T\times512$.]

## 1.11 What We Have *Not* Yet Discussed

To keep Week 1 focused, the following items are deferred to later weeks (but are part of the full Transformer):

- **Positional encodings** (Week 2) — self-attention is order-blind!
- **Residual connections and layer normalisation** (Week 2).
- **Masked self-attention in the decoder, and cross-attention** (Week 2).
- **Final softmax over the vocabulary and training objective** (Week 2).

## 1.12 Practical Applications of What We Learned So Far

Even the encoder alone — with multi-head self-attention and FFNs — is the workhorse behind:

- **BERT** and family (text understanding, embeddings used in search).
- **Sentence-transformers** (semantic search, RAG retrieval).
- **ViT, Swin** (vision transformers — self-attention on image patches).
- **Encoder of T5 / BART** (used in summarisation and translation).

<!-- PDF p.12 / printed p.11 -->

## 1.13 Common Mistakes & Quiz Traps

> **Common Mistake**
>
> **Mistake 1.** “Attention is just for the decoder.”
> **Fix:** *Self-attention* is used inside both the encoder and the decoder, plus *cross-attention* links them. The attention idea is universal in transformers.

> **Common Mistake**
>
> **Mistake 2.** “$W_Q,W_K,W_V$ are the same matrix.”
> **Fix:** They are three *different* learnable matrices. Sharing them turns the model into something else (e.g. Reformer-LSH attention has shared $Q=K$ by construction; standard Transformer does not).

> **Common Mistake**
>
> **Mistake 3.** “Self-attention is $O(T)$ per token.”
> **Fix:** It is $O(T)$ per *token’s* computation but $O(T^2d)$ for the layer because every query attends to every key. This quadratic cost is the central pain point we will attack in Week 11.

> **Quiz Trap**
>
> **Trap:** “Multi-head attention is more parameters than single-head attention of the same $d_{\mathrm{model}}$.”
> **Reality:** *No.* With $h$ heads of width $d_k=d_{\mathrm{model}}/h$, the total parameter count of the $W_Q,W_K,W_V$ matrices is the same as a single-head attention with $d_k=d_{\mathrm{model}}$. The only extra cost is the projection $W^O$. Multi-head “free lunch” = more expressive at (almost) the same parameter budget.

> [possible-error p.12: A single-head attention layer can also include an output projection, so $W^O$ is not inherently an extra multi-head cost.]

## 1.14 Memory Tricks

> **Memory Trick**
>
> **“Q is the question, K is the catalog, V is the content.”**
> A library analogy: I bring a *query* (Q). Books on the shelf have a *key* (K) summarising them. The actual book content is the *value* (V). The librarian (softmax) returns a weighted mix of contents.

> **Memory Trick**
>
> **“Scale to keep softmax sober.”**
> Without the $1/\sqrt{d_k}$ scaling, large dot products push softmax into saturation, killing gradients. Scaling keeps softmax “sober” — in a non-saturated, well-behaved regime.

> **Memory Trick**
>
> **“MHA = many filters of self-attention.”**
> Just as a CNN learns many filters to detect different features, MHA learns many heads to detect different relations.

<!-- PDF p.13 / printed p.12 -->

## 1.15 Comparison Table: RNN/LSTM vs Attention vs Transformer Encoder

| Property | Vanilla RNN/LSTM | Bahdanau attention (RNN + attention) | Transformer encoder |
|---|---|---|---|
| Sequence modelling unit | recurrent cell | recurrent cell | self-attention + FFN |
| Path length between distant tokens | $O(T)$ | $O(T)$ (still goes through RNN) | $O(1)$ |
| Parallelism over $t$ | no | RNN no, attention yes | yes (whole layer) |
| Long-range modelling | poor (gradient decay) | better | strong |
| Quadratic cost in $T$ | no | no (linear in $T$) | yes ($O(T^2d)$) |
| Parameters per layer (base config) | $O(d^2)$ | $O(d^2)$ | $\sim3$M ($d=512$) |
| Best for | short sequences | medium sequences (legacy MT) | most modern tasks |

## 1.16 Quick Revision — Week 1 in One Page

> **Quick Revision**
>
> **Concept tree.**
> - **seq2seq:** encoder RNN → context → decoder RNN. Bottleneck and not parallelisable.
> - **Attention (Bahdanau):** $c_t=\sum_i\alpha_{ti}h_i$ with softmax-normalised scores. Fixes the bottleneck and is itself parallelisable.
> - **Self-attention:** attention with $s_i,h_j$ both coming from the same sequence. Q, K, V from three linear projections of the same input.
> - **Scaled dot-product attention:** $Z=\operatorname{softmax}(Q^\top K/\sqrt{d_k})V^\top$.
> - **Multi-head attention:** $h=8$ heads, each $d_k=64$, concatenate → $W^O$.
> - **Encoder layer:** MHA → FFN; each layer outputs $T\times512$.
> - **Stack:** $N=6$ encoder layers $\Rightarrow\approx18$M parameters.
>
> **Must-remember formulas.**
> $$Z=\operatorname{softmax}\left(\frac{Q^\top K}{\sqrt{d_k}}\right)V^\top,$$
> $$\operatorname{FFN}(z)=\max(0,W_1z+b_1)W_2+b_2.$$
>
> **Why Transformer beats RNN.** (i) Path length $O(1)$ between any two positions; (ii) full parallelism across the sequence; (iii) no vanishing/exploding gradients along time; (iv) easier to scale to billions of parameters.

## 1.17 Practice Questions — Week 1

### Multiple Choice

1. **(Q1)** The reason for dividing $QK^\top$ by $\sqrt{d_k}$ in scaled dot-product attention is to:

<!-- PDF p.14 / printed p.13 -->

   (A) reduce memory usage,
   (B) prevent the softmax from saturating when $d_k$ is large,
   (C) normalise the output of attention to have unit norm,
   (D) match the dimensionality of $V$.

   **Solution.** Recall that for random $q,k$ with i.i.d. unit-variance components, $\operatorname{Var}(q\cdot k)=d_k$. Large variance pushes the softmax inputs into extreme regions where one entry dominates and gradients vanish. Dividing by $\sqrt{d_k}$ rescales the variance to 1. **Answer: (B).** (A) is wrong — it does not change memory; (C) is wrong — attention outputs are not unit-norm; (D) is wrong — the scaling is along the score axis, not the value axis.

2. **(Q2)** In multi-head attention with $h=8$ and $d_{\mathrm{model}}=512$, the per-head dimension $d_k$ is:
   (A) 8
   (B) 64
   (C) 256
   (D) 512

   **Solution.** $d_k=d_{\mathrm{model}}/h=512/8=64$. **Answer: (B).**

3. **(Q3)** Compared to single-head attention with $d_k=d_{\mathrm{model}}=512$, multi-head attention with $h=8,d_k=64$ has:
   (A) more $W_Q,W_K,W_V$ parameters in total,
   (B) fewer $W_Q,W_K,W_V$ parameters in total,
   (C) the same total number of $W_Q,W_K,W_V$ parameters,
   (D) zero additional parameters.

   **Solution.** Single-head: each $W$ has shape $512\times512$. Multi-head: 8 matrices of shape $64\times512$ each, total $8\cdot64\cdot512=512^2$ — the same. The only extra cost is $W^O$. **Answer: (C).**

4. **(Q4)** Which property is NOT shared by vanilla RNN seq2seq and the Transformer encoder?
   (A) Can process variable-length input.
   (B) Produces a fixed-size representation per token.
   (C) Parallelises computation across positions of one sequence.
   (D) Uses learnable parameters.

   **Solution.** The Transformer encoder parallelises across positions; the RNN does not. **Answer: (C).**

### Conceptual / Short Answer

5. **(Q5)** In one or two sentences, explain why a single context vector $h_T$ is a bottleneck for long sentences in seq2seq.

   **Answer.** Compressing arbitrary-length source information into a single fixed-size vector loses information that scales with sentence length; the decoder cannot recover details it never sees. Empirically, BLEU degrades sharply beyond $\sim30$ tokens. Attention solves this by exposing all encoder states $h_1,\ldots,h_n$ to the decoder.

<!-- PDF p.15 / printed p.14 -->

6. **(Q6)** Why are $W_Q,W_K,W_V$ separate matrices instead of one?

   **Answer.** The same word plays three roles — as a question (query), as an addressable index (key), and as content (value). Forcing all three to coincide would constrain the geometry of the attention space. Separate matrices let the model learn three different subspaces, each tuned for one role.

7. **(Q7)** Show that the attention weights for output position $t$ can be computed in parallel for all input positions $i$.

   **Answer.** The score $\operatorname{score}(s_{t-1},h_i)$ depends only on $s_{t-1}$ (fixed at step $t$) and $h_i$ (precomputed by the encoder). Therefore all $i$-indexed scores are independent and can be evaluated as one matrix multiplication, followed by a row-wise softmax.

8. **(Q8) Tensor-shape tracing.** Let $T=10,d_{\mathrm{model}}=512,h=8$. Give the shapes of (a) the per-head $Q_i,K_i,V_i$; (b) the per-head attention matrix $\alpha_i$; (c) the per-head output; (d) the concatenated output; (e) the final layer output.

   **Answer.** (a) $10\times64$; (b) $10\times10$; (c) $10\times64$; (d) $10\times512$ (concat of 8 heads); (e) $10\times512$ after $W^O$ then FFN.

### Numerical / Derivation

9. **(Q9)** Compute the number of learnable parameters in a single encoder *layer* (one MHA + one FFN, $d_{\mathrm{model}}=512,h=8,d_k=64$, FFN width $l=2048$). Ignore biases for MHA but include biases for FFN.

   **Solution.**
   $$\begin{aligned}\mathrm{MHA}:{}&3\cdot(h\cdot d_k)\cdot d_{\mathrm{model}}+d_{\mathrm{model}}^2\\&=3\cdot512\cdot512+512^2\\&=4\cdot512^2=1,048,576\\\mathrm{FFN}:{}&512\cdot2048+2048+2048\cdot512+512\\&\approx2.10\times10^6\\\text{Per layer}:{}&\approx3.15\times10^6.\end{aligned}$$

10. **(Q10)** Suppose $d_k=4$ and a single query $q=[1,0,1,0]$ attends to two keys $k_1=[1,0,1,0]$, $k_2=[0,1,0,1]$ with values $v_1=[2,0]$, $v_2=[0,2]$. Compute the output of scaled dot-product attention (single head).

    **Solution.** $q\cdot k_1=2$, $q\cdot k_2=0$. Scaled: $2/\sqrt4=1$ and $0/2=0$. Softmax: $\alpha=(e/(e+1),1/(e+1))\approx(0.731,0.269)$. Output $z=0.731\cdot(2,0)+0.269\cdot(0,2)\approx(1.462,0.538)$.

11. **(Q11)** Derive the softmax of a vector $x=(x_1,x_2)$ and show that adding a constant $c$ to both entries does not change the result. (Hint: this is why we can subtract $\max(x)$ in the safe-softmax trick.)

    **Solution.** $\operatorname{softmax}(x_1+c,x_2+c)_i=\frac{e^{x_i+c}}{e^{x_1+c}+e^{x_2+c}}=\frac{e^ce^{x_i}}{e^c(e^{x_1}+e^{x_2})}=\operatorname{softmax}(x)_i$. So softmax is translation-invariant — we exploit this in the safe-softmax used in Flash Attention (Week 11).

<!-- PDF p.16 / printed p.15 -->

## 1.18 Week 1 Addendum — Deeper Dives

This addendum revisits four topics that deserve more space than the main Week 1 narrative gave them. They are flagged in the supplementary lecture material (*Counting Encoder Parameters* and *MultiHeadAttention* handouts).

### 1.18.1 1A — A Fully Worked Multi-Head Attention Example (by hand)

> **Intuition**
>
> Equations are easy to nod along to. Until you grind through a 2-token, 2-head example by hand, multi-head attention does not really click. We will use *small numbers* specifically so we can verify every step.

**Setup.**

- Sequence length $t=2$ (two tokens).
- Number of heads $h=2$.
- Embedding dimension $d_{\mathrm{model}}=4$.
- Per-head $d_k=d_q=d_{\mathrm{model}}/h=2$.
- Per-head value size $d_v=2$.
- Input $X\in\mathbb R^{d_{\mathrm{model}}\times t}=\mathbb R^{4\times2}$:
  $$X=\begin{bmatrix}1&0\\0&2\\1&0\\0&2\end{bmatrix}.$$
  (Each column is one token; this convention follows the lecturer’s slides.)

**Head 1 weights** (we deliberately set $W_Q^1=W_K^1=W_V^1$ to keep the arithmetic transparent):
$$W_Q^1=W_K^1=W_V^1=\begin{bmatrix}1&1&1&1\\2&2&2&2\end{bmatrix}\in\mathbb R^{d_q\times d_{\mathrm{model}}}=\mathbb R^{2\times4}.$$

**Head 2 weights:**
$$W_Q^2=W_K^2=W_V^2=\begin{bmatrix}1&0&1&0\\1&1&2&1\end{bmatrix}.$$

**Output projection** $W^O\in\mathbb R^{d_{\mathrm{model}}\times(h\cdot d_v)}=\mathbb R^{4\times4}$:
$$W^O=\begin{bmatrix}0.5&0&0&0\\0&0.5&0&0\\0&0&1&0\\0&0&0&1\end{bmatrix}.$$

**Step 1 — Head 1: compute $Q^1,K^1,V^1$.**
$$Q^1=K^1=V^1=W_Q^1X=\begin{bmatrix}1&1&1&1\\2&2&2&2\end{bmatrix}\begin{bmatrix}1&0\\0&2\\1&0\\0&2\end{bmatrix}=\begin{bmatrix}2&4\\4&8\end{bmatrix}.$$

**Step 2 — Head 1: scaled dot product.** With $\sqrt{d_k}=\sqrt2$:
$$\frac{(Q^1)^\top K^1}{\sqrt{d_k}}=\frac1{\sqrt2}\begin{bmatrix}2&4\\4&8\end{bmatrix}\begin{bmatrix}2&4\\4&8\end{bmatrix}=\frac1{\sqrt2}\begin{bmatrix}20&40\\40&80\end{bmatrix}\approx\begin{bmatrix}14.14&28.28\\28.28&56.56\end{bmatrix}.$$

<!-- PDF p.17 / printed p.16 -->

**Step 3 — Head 1: row-wise softmax.** For row 1, $e^{14.14}/(e^{14.14}+e^{28.28})\approx0$ and the other entry $\approx1$. Same pattern for row 2:
$$\operatorname{softmax}_{\mathrm{row}}\left(\frac{(Q^1)^\top K^1}{\sqrt{d_k}}\right)\approx\begin{bmatrix}0&1\\0&1\end{bmatrix}.$$

Note that one of the entries is essentially 0 and the other essentially 1 — the softmax saturated because the score gap (14 vs 28) is huge. This is exactly the saturation problem that motivates the $1/\sqrt{d_k}$ scaling — and here, even with scaling, the toy weights are extreme enough to saturate it.

**Step 4 — Head 1: multiply by $V$.**
$$\operatorname{head}_1=\operatorname{softmax}(\ldots)(V^1)^\top=\begin{bmatrix}0&1\\0&1\end{bmatrix}\begin{bmatrix}2&4\\4&8\end{bmatrix}=\begin{bmatrix}4&8\\4&8\end{bmatrix}\in\mathbb R^{t\times d_v}=\mathbb R^{2\times2}.$$

Both tokens end up with the same head-1 output because both rows of attention weights were essentially identical $(0,1)$.

**Step 5 — Head 2: same procedure.**
$$Q^2=K^2=V^2=W_Q^2X=\begin{bmatrix}1&0&1&0\\1&1&2&1\end{bmatrix}\begin{bmatrix}1&0\\0&2\\1&0\\0&2\end{bmatrix}=\begin{bmatrix}2&0\\3&4\end{bmatrix}.$$

After scaled dot product and softmax (you can verify):
$$\operatorname{softmax}\left(\frac{(Q^2)^\top K^2}{\sqrt{d_k}}\right)\approx\begin{bmatrix}0.67&0.33\\0.06&0.94\end{bmatrix}.$$


This time the softmax did *not* saturate because the score differences are small.
$$\operatorname{head}_2\approx\begin{bmatrix}0.67&0.33\\0.06&0.94\end{bmatrix}\begin{bmatrix}2&0\\3&4\end{bmatrix}=\begin{bmatrix}1.34&1.33\\0.11&3.94\end{bmatrix}.$$

> [possible-error p.17: The displayed value matrix should be $(V^2)^\top=\begin{bmatrix}2&3\\0&4\end{bmatrix}$; the stated output also does not match the shown softmax weights.]

*(Slight differences from the slide are due to rounding.)*

**Step 6 — Concatenate heads.** Concatenate along the feature axis ($Z\in\mathbb R^{t\times(h\cdot d_v)}=\mathbb R^{2\times4}$):
$$Z=[\operatorname{head}_1\mid\operatorname{head}_2]=\begin{bmatrix}4&8&1.34&1.33\\4&8&0.11&3.94\end{bmatrix}.$$

**Step 7 — Apply $W^O$ to get the final MHA output.** The lecturer writes $\mathrm{MHA}=W^OZ^\top$ to keep tokens as columns (output $\in\mathbb R^{d_{\mathrm{model}}\times t}=\mathbb R^{4\times2}$):
$$\mathrm{MHA}=W^OZ^\top=\begin{bmatrix}0.5&0&0&0\\0&0.5&0&0\\0&0&1&0\\0&0&0&1\end{bmatrix}\begin{bmatrix}4&4\\8&8\\1.34&0.11\\1.33&3.94\end{bmatrix}=\begin{bmatrix}2&2\\4&4\\1.34&0.11\\1.33&3.94\end{bmatrix}.$$

The dimensions check out: $4\times2$ — one column per token, $d_{\mathrm{model}}=4$ features per column.

<!-- PDF p.18 / printed p.17 -->

> **Remember This**
>
> **What this exercise teaches.**
> 1. Each head independently produces a $t\times d_v$ matrix — here $2\times2$.
> 2. Concatenation gives a wide $t\times(h\cdot d_v)$ matrix — here $2\times4$.
> 3. $W^O$ projects from concatenated heads back to $d_{\mathrm{model}}$ — here it maps 4-d → 4-d. In the base Transformer, it maps 512-d → 512-d.
> 4. Different heads can produce different attention patterns: head 1 was saturated (each token mostly attends to token 2), head 2 was soft (mix of both tokens).
> 5. Even with very small numerical examples, the saturation effect of softmax is visible — this is why we scale by $\sqrt{d_k}$ and why initialisation matters.

### 1.18.2 1B — The Role of $W^O$ (Why We Need It)

> **Intuition**
>
> After concatenating $h$ heads of width $d_v$, you have a wide vector of $h\cdot d_v$ features — but the rest of the Transformer expects vectors of width $d_{\mathrm{model}}$. If $h\cdot d_v=d_{\mathrm{model}}$ (the standard case), $W^O$ is “just” a $d_{\mathrm{model}}\times d_{\mathrm{model}}$ linear map. So why bother?

**Reason 1: dimensional matching.** In the general case, the $h$ heads can have different $d_{v_i}$ values, so the concatenation is $\mathbb R^{d_{v_1}+d_{v_2}+\cdots+d_{v_h}}$. Then $W^O\in\mathbb R^{d_{\mathrm{model}}\times\sum_i d_{v_i}}$ maps to the canonical width needed by the next sub-layer.

**Reason 2: mixing across heads.** Without $W^O$, the $h$ heads would write into $h$ disjoint blocks of the output vector — they could never interact. $W^O$ lets the model *learn* how to combine information from different heads on a per-feature basis. This is what makes “many heads” more than just “many parallel attentions glued together.”

**Reason 3: optional dimensionality change.** You could choose $h\cdot d_v\ne d_{\mathrm{model}}$ (with appropriate $W^O$). Some architectures use this freedom, although base Transformer keeps the two equal.

> **Formula Insight**
>
> Putting it all together, the per-head and overall MHA equations are:
> $$\operatorname{head}_i=\operatorname{Attention}(W_Q^iX,W_K^iX,W_V^iX)$$
> $$\operatorname{MHA}(X)=W^O\operatorname{Concat}(\operatorname{head}_1,\operatorname{head}_2,\ldots,\operatorname{head}_h).$$
> The $W^O$ is the only *cross-head* learnable interaction in MHA.

### 1.18.3 1C — Alignment Diagonal, Diagonality, and Word Order

In slide 11, the lecturer showed a 4-output $\times$ 5-input attention heatmap for an English → Tamil translation. Four entries were highlighted:

- $\alpha_{11}$ — “**Naan**” (output 1) aligned with “**I**” (input 1). On the diagonal.
- $\alpha_{25}$ — “**transfarmar**” (output 2) aligned with “**transformers**” (input 5). *Off-diagonal* — Tamil moves the noun.
- $\alpha_{34}$ — “**padaththai**” aligned with “**film**”.
- $\alpha_{42}$ — “**rasiththen**” aligned with “**enjoyed**”. *Off-diagonal.*

<!-- PDF p.19 / printed p.18 -->

> **Intuition**
>
> For monotonically aligned language pairs (English-French), attention is roughly diagonal (output $t$ attends to input $t$). For free-word-order languages (English-Tamil, English-Japanese), attention develops non-monotonic patterns. Critically, the model *learns* this without any explicit alignment supervision — this is the breakthrough of the Bahdanau paper (*Neural Machine Translation by Jointly Learning to Align and Translate*, 2015).

> **Research / Industry Insight**
>
> Visualising attention has become a standard interpretability tool. Tools like BertViz, exBERT, and the Tensor2Tensor visualiser let you inspect attention heads layer-by-layer. Several heads in BERT correspond to interpretable linguistic relations: “next token”, “previous token”, “coreference”, “direct object”, etc. This was popularised by Clark et al. (2019) in *What Does BERT Look At?*.

### 1.18.4 1D — The Detailed Parameter Counting Walkthrough

In the main Week 1 narrative, we said “$\sim3$ million parameters per encoder layer”. The supplementary “Counting Encoder Parameters” slide does it in painstaking detail for $T=40,h=8,d_k=d_q=d_v=64,d_{\mathrm{model}}=512$, FFN width $l=2048$.

**Per-head $W_Q,W_K,W_V$: **
$$\begin{aligned}W_Q &:64\times512=32,768\quad\text{(per head; the slide rounds to 32,178 — a typo)}\\W_K &:64\times512=32,768\\W_V &:64\times512=32,768.\end{aligned}$$

**All 8 heads combined:**
$$\begin{aligned}8\text{ heads of }W_Q &:8\cdot64\cdot512=262,144\\8\text{ heads of }W_K &:262,144\\8\text{ heads of }W_V &:262,144\\\text{Subtotal}&:786,432.\end{aligned}$$

**Output projection:**
$$W^O:512\times512=262,144.$$

**Total MHA parameters per layer:**
$$786,432+262,144=1,048,576\approx1.05\times10^6\ \approx1\mathrm M.$$

**FFN parameters (with biases):**
$$\underbrace{512\cdot2048}_{W_1}+\underbrace{2048}_{b_1}+\underbrace{2048\cdot512}_{W_2}+\underbrace{512}_{b_2}=1,048,576+2048+1,048,576+512\approx2.1\times10^6\ \approx2\mathrm M.$$

**Total per encoder layer:** $\approx1.05\mathrm M+2.1\mathrm M\ \approx3.15\mathrm M$.

**Total encoder stack ($N=6$):** $\approx6\times3.15\mathrm M=18.9\mathrm M$.

<!-- PDF p.20 / printed p.19 -->

> **Exam Tip**
>
> **The Khapra mnemonic: “MHA is 1, FFN is 2, layer is 3.”**
> One million parameters in MHA, two million in FFN, three million per encoder layer. For $N=6$ layers: **18 million encoder parameters.**

> **Quiz Trap**
>
> **Trap.** “Increasing the number of heads $h$ (with $d_k=d_{\mathrm{model}}/h$) increases the parameter count.”
> **Reality.** No. The total parameter count of $W_Q,W_K,W_V$ is $3\cdot(h\cdot d_k)\cdot d_{\mathrm{model}}=3\cdot d_{\mathrm{model}}^2$ regardless of $h$. Doubling $h$ halves $d_k$ at no parameter cost. Only $W^O$ remains fixed at $d_{\mathrm{model}}^2$. The constraint $h\cdot d_k=d_{\mathrm{model}}$ is what makes this work.

### 1.18.5 1E — Extra Practice Questions for the Addendum

1. **(A1)** In the worked 2-head example above, why do both tokens get the same output from head 1 but different outputs from head 2?

   **Answer.** In head 1, both rows of the softmax were essentially $(0,1)$ — both queries gave nearly identical attention patterns due to large dot products and softmax saturation. So both tokens attend almost entirely to token 2’s value, producing the same head-1 output. In head 2, the weights are softer (e.g., row 1 is $(0.67,0.33)$), giving distinct outputs per token.

2. **(A2)** Suppose you remove $W^O$ from multi-head attention. What is lost?

   **Answer.** Without $W^O$: (i) different heads occupy disjoint slices of the output, never interacting; (ii) the output dimension is fixed at $h\cdot d_v$ without the ability to project to a different size; (iii) the model loses the only cross-head mixing parameter. Empirically this dramatically hurts performance.

3. **(A3)** If you double $d_{\mathrm{model}}$ from 512 to 1024 (and keep $h=8$, so $d_k=128$), by what factor do MHA parameters change per layer?

   **Solution.** MHA parameters scale as $4d_{\mathrm{model}}^2$ (three $W_Q,W_K,W_V$ totalling $3d^2$, plus one $W^O$ of $d^2$). Doubling $d$ quadruples the count: from $\sim1$M to $\sim4$M. *Lesson:* MHA cost grows quadratically in $d_{\mathrm{model}}$ — this is why scaling embeddings is expensive.

4. **(A4)** For an encoder with $N=12$ layers, $d_{\mathrm{model}}=768$, $h=12$, FFN width $l=3072$ (**BERT-base** config), estimate the total encoder parameters.

   **Solution.**
   - MHA per layer: $4\cdot768^2\approx2.36$M.
   - FFN per layer: $2\cdot768\cdot3072+3072+768\approx4.72$M.
   - Per layer total: $\approx7$M.
   - $N=12$ layers: $\approx85$M parameters.

   With embeddings ($\sim24$M for $|V|=30,000$), BERT-base has the well-known $\approx110$M total parameters. *Quick interview answer: BERT-base $\sim110$M.*

5. **(A5)** Why is the attention matrix *row-stochastic* (each row sums to 1) but *not* column-stochastic?

   **Answer.** Softmax is applied along the row direction (over keys, for each query). Each query’s attention weights sum to 1 — it must put 100% of its attention somewhere. But the same key can be attended to by many queries, so the column sum is unconstrained.
