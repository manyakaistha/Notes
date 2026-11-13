<!-- PDF p.34 / printed p.33 -->

# 3 Week 3 — Language Modelling, GPT, and the Pre-train / Fine-tune Paradigm

**Chapter goals.** By the end of this chapter you will be able to:
1. Explain why pre-training on raw text is the right answer to the “one model per task / not enough labels” problem.
2. Define a language model formally and apply the chain rule of probabilities.
3. Distinguish encoder-only, decoder-only, and encoder–decoder architectures for language modelling.
4. Derive **Causal Language Modelling (CLM)** and write the training objective.
5. Walk through the GPT-1 architecture, dataset (BookCorpus), hyperparameters, and parameter count ($\sim 117\mathrm{M}$).
6. Explain the supervised fine-tuning recipe for classification, entailment, multiple choice, and text generation.

## 3.1 Where We Stand: The Problem to Solve

In Weeks 1–2 we built the full Transformer for *machine translation*. At a high level, the architecture is just an encoder-decoder black box:

[Figure: source text → Transformer → translated text.]

A natural next question: *Can we use the same architecture for other NLP tasks — sentiment classification, summarisation, question answering, entailment, MCQs?*

### 3.1.1 The naive approach and its problems

We could train a separate Transformer for each task from scratch. But:

- **Convergence is slow.** Random initialisation + each task’s small labelled dataset = many epochs of training.
- **Labelled data is scarce.** High-quality labels for sentiment, QA, NLI cost money and time. Many tasks have only thousands of examples.
- **No knowledge sharing.** A model that learns “apple” is a fruit in a sentiment dataset cannot reuse that for QA.

> **Intuition**
>
> A human child does not learn each language task separately. We acquire language by reading and listening for years *without explicit supervision*, then specialise quickly to specific tasks. We say things like “*Wow, India has reached the moon*” and people immediately understand the sentiment, the syntax, the entailment — without having been trained on a sentiment classification dataset. Can a model learn the same way?

The answer is yes, and the key insight is:

> **Remember This**
>
> **The Pre-train + Fine-tune Paradigm.** First *pre-train* a model on a huge amount of *unlabelled* text using a self-supervised objective (predict the next word). Then *fine-tune* on a small labelled dataset for each downstream task. The pre-trained model has already learned “what language looks like”; fine-tuning only adapts it to the specific task.

<!-- PDF p.35 / printed p.34 -->

The empirical evidence (Radford et al., 2018, GPT-1 paper): the pre-trained model with a *linear classifier* on top reached 91.8% accuracy on sentiment classification — beating the state-of-the-art — and matched fully supervised systems while using **30–100× fewer labelled examples**. This was the empirical kick-off of the modern LLM era.

## 3.2 Language Modelling — Formal Definition

> **Definition**
>
> Let $\mathcal{V}$ be the *vocabulary* — the set of all unique tokens in a language. A sentence is a sequence $X_1,X_2,\ldots,X_n$ with each $X_i\in\mathcal{V}$.
> A **language model** is a function
> $$f:(X_1,X_2,\ldots,X_n)\longrightarrow[0,1]$$
> that assigns a probability to each sequence. More probable sequences should be the ones that “look like real language”.

**Tiny example.** Let $\mathcal{V}=\{\text{an, apple, ate, I}\}$. Some possible sequences:
- “An apple ate I.” — grammatical disaster, low probability.
- “I ate an apple.” — correct, high probability.
- “I ate apple.” — ungrammatical but plausible, medium probability.
- “An apple.” — short fragment.

A good language model would assign these very different probabilities.

### 3.2.1 The chain rule of probability

The joint probability of a sequence factorises:
$$P(x_1,x_2,\ldots,x_T)=P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_2,x_1)\cdots P(x_T\mid x_{T-1},\ldots,x_1),$$
or compactly
$$P(x_1,x_2,\ldots,x_T)=\prod_{i=1}^{T}P(x_i\mid x_1,\ldots,x_{i-1}). \tag{21}$$

### 3.2.2 The naive independence assumption fails

If we naively assumed words were independent: $P(x_1,\ldots,x_T)=\prod_i P(x_i)$. But then “I enjoyed reading a book” and “I enjoyed reading a thermometer” would get similar probabilities (both have common words). The *presence* of “enjoyed” + “reading” should make “book” far more likely than “thermometer”. Words are *conditional* on context. So the right model is Eq. (21).

> **Formula Insight**
>
> The chain rule turns a hard problem (modelling joint distribution over all possible sequences) into a sequence of *conditional* problems, each “predict the next word given the past”. This is exactly what a left-to-right neural language model computes.

## 3.3 Three Ways to Parameterise a Language Model

We have a Transformer at our disposal. We can use it three ways:

<!-- PDF p.36 / printed p.35 -->

| Architecture | Input / Objective | Example models |
|---|---|---|
| **Encoder-only** | Input has a `[MASK]` token; predict the masked word using both left and right context. | BERT, RoBERTa, ALBERT |
| **Decoder-only** | Input is $x_1,\ldots,x_{i-1}$; predict $x_i$. *Left context only (causal).* | GPT-1, 2, 3, 4; LLaMA; Mistral; Claude; Gemini |
| **Encoder–decoder** | Encode an input (possibly masked) and decode an output sequence. | T5, BART, Pegasus |

> **Intuition**
>
> **Encoder-only** models can use *bidirectional* context. They are best for *understanding* tasks (classification, NER, sentence embeddings).
> **Decoder-only** models see only the past. They are best for *generation* (continuing a text, writing, dialogue).
> **Encoder-decoder** models read one sequence and produce another. They are best when input and output are structurally different (translation, summarisation).
> Modern LLMs (GPT family, LLaMA, Claude) are overwhelmingly **decoder-only** because generation is the most general capability — and a decoder can be *prompted* to do any task by phrasing it as “predict what comes next”.

## 3.4 Causal Language Modelling (CLM)

The decoder-only language model is the canonical setup. It is sometimes called *Causal Language Modelling* because each prediction depends only on *past* (causal) context.

> **Definition**
>
> A **Causal Language Model** learns a parameterised distribution
> $$P(x_i\mid x_1,\ldots,x_{i-1};\theta)=f_\theta(x_i\mid x_1,\ldots,x_{i-1})$$
> typically implemented as a Transformer decoder with masked self-attention. The **training objective** is to maximise the log-likelihood over a corpus:
> $$\mathcal{L}(\theta)=\sum_{i=1}^{T}\log P(x_i\mid x_1,\ldots,x_{i-1};\theta).$$

### 3.4.1 Architecture: A pruned decoder block

We start from the original Transformer decoder (three sub-layers: masked self-attention, cross-attention, FFN). For language modelling there is *no encoder*, so we remove the cross-attention sub-layer:

$$\textbf{Causal LM block}=\text{Masked Self-Attention}\ \longrightarrow\ \text{FFN}.$$

Diagram — *this is the heart of GPT*:

<!-- PDF p.37 / printed p.36 -->

[Figure: Inputs $x_1,x_2,\ldots,x_{i-1}$ pass through masked multi-head (self-)attention, Add & Norm, feed forward network, and Add & Norm to produce $P(x_i)$.]

Stack $n$ such blocks; the original GPT-1 used $n=12$.

### 3.4.2 The output side: from hidden vectors to probabilities

Let the final-block hidden states be $h_n[1],h_n[2],\ldots,h_n[T]\in\mathbb{R}^{d_{\text{model}}}$. A single weight matrix $W_v\in\mathbb{R}^{d_{\text{model}}\times|\mathcal{V}|}$ projects each hidden state to vocabulary scores; softmax converts scores to probabilities:
$$P(x_i\mid x_1,\ldots,x_{i-1})=\operatorname{softmax}(h_n[i]W_v). \tag{22}$$
> [possible-error p.37: The equation uses $h_n[i]$ for predicting $x_i$, whereas the following section says the prediction uses the position $i-1$ hidden state.]

### 3.4.3 The masking trick again — why one forward pass suffices

> **Intuition**
>
> At every position $i$, the model predicts $P(x_i\mid\text{past})$ *from the position $i-1$ hidden state*. Because of the causal mask, position $i-1$ only attends to positions $1,\ldots,i-1$ — so by construction, the conditional probability respects causality.
> The genius: a single forward pass over $T$ tokens produces *all $T$ conditional probabilities* of Eq. (21) simultaneously. This is exactly what makes Transformer LM training *parallel*, unlike RNN LMs.

## 3.5 GPT-1 — The First Big Win for Pretraining + Decoder-Only Transformers

**Generative Pretrained Transformer 1** (Radford, Narasimhan, Salimans, Sutskever, 2018) used the following architecture:

| | |
|---|---|
| **Number of decoder blocks ($n$)** | 12 |
| **Embedding dimension ($d_{\text{model}}$)** | 768 |
| **Attention heads ($h$)** | 12 (so $d_k=64$) |
| **FFN hidden width ($l$)** | $4\cdot768=3072$ |
| **Context length ($T$)** | 512 |
| **Activation in FFN** | GELU |
| **Position embedding** | *Learned* (not sinusoidal) |
| **Vocabulary size ($|\mathcal{V}|$)** | 40,478 |
| **Tokenizer** | Byte-Pair Encoding (BPE) |
| **Optimizer** | Adam, cosine LR schedule |
| **Batch size** | 64 sequences |
| **Dataset** | BookCorpus |

<!-- PDF p.38 / printed p.37 -->

**BookCorpus —** 7,000 unique books, 74 million sentences, $\approx 1$ billion words spanning 16 genres. Importantly, it contains *long contiguous passages* (no sentence-level shuffling), which is critical for learning long-range dependencies.

> **Research / Industry Insight**
>
> The choice of learned (vs. sinusoidal) position embedding became standard in decoder-only LLMs — it’s a simple lookup table of size $T_{\max}\times d_{\text{model}}$. The downside is that the model cannot extrapolate beyond $T_{\max}$. GPT-2/3 inherited this; later models (GPT-NeoX, LLaMA) moved to **RoPE** (rotary position embeddings) which extrapolate better.

## 3.6 Counting GPT-1’s Parameters — The Famous 117M

This is one of the most-asked exam / interview questions, so let’s do it slowly. Three components contribute.

### 3.6.1 (a) Embeddings

- **Token embeddings:** $|\mathcal{V}|\cdot d_{\text{model}}=40,478\cdot768\approx31.1\ \mathrm{M}$.
- **Position embeddings (learned):** $T_{\max}\cdot d_{\text{model}}=512\cdot768\approx0.39\ \mathrm{M}$.
- **Subtotal:** $\approx31.3\ \mathrm{M}$.

> [possible-error p.38: The stated embedding components sum to approximately 31.5 M, not 31.3 M.]

### 3.6.2 (b) Per-block attention parameters

Per head: $W_Q,W_K,W_V$ are each $768\times64$, so $3\cdot768\cdot64\approx147,456$. With $h=12$ heads, that’s $12\cdot147,456\approx1.77\ \mathrm{M}$. Plus the output projection $W^O$ of $768\times768=589,824\approx0.59\ \mathrm{M}$.

**Per block:** $1.77+0.59\approx2.36\ \mathrm{M}$.

**For all 12 blocks:** $12\cdot2.36\approx28.3\ \mathrm{M}$ (the slide rounds to $27.6\ \mathrm{M}$).

> [possible-error p.38: 28.3 M cannot be rounded to 27.6 M.]

### 3.6.3 (c) Per-block FFN parameters

$$2\cdot(768\cdot3072)+3072+768=4,722,432+3072+768\approx4.72\ \mathrm{M}\text{ per block.}$$

For 12 blocks: $\approx56.7\ \mathrm{M}$.

### 3.6.4 Totals

| Layer | Parameters (M) |
|---|---:|
| Embedding (token + position) | 31.3 |
| Attention (12 blocks) | 27.6 |
| FFN (12 blocks) | 56.4 |
| **Total** | $\approx117\ \mathrm{M}$ |

> [possible-error p.38: The table entries sum to 115.3 M and differ from the preceding calculated component totals.]

> **Exam Tip**
>
> **Memorise: GPT-1 has $\sim117$M parameters.** Breakdown: roughly $1/4$ embedding, $1/4$ attention, $1/2$ FFN. (Note: the FFN dominates the parameter budget at this scale — a recurring pattern in Transformers.)

<!-- PDF p.39 / printed p.38 -->

## 3.7 Pre-Training Objective in One Equation

Given a corpus $\mathcal{X}$ of token sequences, the cross-entropy loss per sequence is:
$$\mathcal{L}_{\text{pre}}(\theta)=-\sum_{X\in\mathcal{X}}\sum_{i=1}^{T}\sum_{v\in\mathcal{V}}y_v^{(i)}\log\hat y_v^{(i)}, \tag{23}$$
where $y_v^{(i)}$ is 1 if the true next token at position $i$ is $v$ (one-hot), and $\hat y_v^{(i)}$ is the model’s predicted probability. Equivalently:
$$\mathcal{L}_{\text{pre}}(\theta)=-\sum_{i=1}^{T}\log P(x_i\mid x_{<i};\theta). \tag{24}$$

This is exactly the (negative) log-likelihood of Eq. (21).

### 3.7.1 Why teacher forcing again?

During pre-training we feed the *actual* ground-truth tokens at each position (not the model’s own predictions). This is teacher forcing (Week 2). It enables one parallel forward pass over all $T$ positions and dramatically speeds up training compared to auto-regressive sampling.

## 3.8 Fine-Tuning — Adapting GPT to Downstream Tasks

After pre-training, we have a model that has internalised a lot about language. To solve a specific labelled task, we:

1. **Initialise** all the Transformer parameters from the pre-trained checkpoint.
2. **Format the input** with special tokens (start $\langle s\rangle$, end $\langle e\rangle$, delimiter $\langle\$\rangle$ where appropriate) so the model knows what kind of task it is solving.
3. **Replace the LM head** ($W_v$ mapping to $|\mathcal{V}|$) with a **task-specific head** (e.g., a linear layer $W_y$ mapping to the number of classes).
4. **Train end-to-end** on the labelled dataset — typically all parameters are updated, but the LR is much smaller than pre-training.

### 3.8.1 Why use the *last* position’s hidden state for classification?

The output for classification is $\hat y=\operatorname{softmax}(W_y h_m^{(n)})$ where $h_m^{(n)}$ is the *last* token’s hidden state at the *last* (top) layer. Why the last token?

> **Intuition**
>
> Because of causal masking, only the *last* position can attend to *every* preceding token. Earlier positions see only a prefix. The last token therefore contains a summary of the entire sequence — it’s the only position that has “seen everything”.

### 3.8.2 Fine-tuning loss

$$\mathcal{L}_{\text{FT}}=-\sum_{(X,y)\in\mathcal{C}}\log P(y\mid x_1,\ldots,x_m). \tag{25}$$

(Optionally, GPT-1 used an auxiliary LM loss during fine-tuning too, mixed with the task loss — this regularises the model and slows down catastrophic forgetting.)

<!-- PDF p.40 / printed p.39 -->

## 3.9 Task-Specific Input Formats

GPT-1 used a clever idea: instead of building task-specific architectures, encode the *task* as part of the input through special delimiter tokens. Then a single Transformer solves all tasks. Examples:

### 3.9.1 Sentiment classification

$$\langle s\rangle\ \text{Wow, India has now reached the moon}\ \langle e\rangle\ \to\ \hat y\in\{+1,-1\}.$$

Linear head reads the hidden state at $\langle e\rangle$.

### 3.9.2 Textual entailment / NLI

$$\langle s\rangle\ \text{Premise}\ \langle\$\rangle\ \text{Hypothesis}\ \langle e\rangle\ \to\ \hat y\in\{\text{entail, neutral, contradict}\}.$$

The delimiter $\langle\$\rangle$ tells the model: “premise ends here, hypothesis begins”.

### 3.9.3 Multiple choice (MCQ)

For an MCQ with $K$ choices, run the model $K$ times, once per choice:
$$\text{Run }k:\quad\langle s\rangle\ \text{Question}\ \langle\$\rangle\ \text{Choice-}k\ \langle e\rangle.$$

Each run produces a scalar score (a linear layer reads the $\langle e\rangle$ hidden state). Stack the $K$ scores into a vector, apply softmax across choices, pick the largest.

### 3.9.4 Text generation

The simplest task. Feed a prompt:
$$\langle s\rangle\ \text{I like}$$
and run the model auto-regressively, sampling $\hat x_{i+1}\sim P(\cdot\mid x_{\leq i})$ each step, until $\langle e\rangle$ is produced or a maximum length is reached. Output: “I like to think that…”.

> **Remember This**
>
> **The pre-train + fine-tune recipe —** in three lines:
> 1. Pre-train a big decoder-only Transformer on raw text with the LM objective.
> 2. Add task-specific delimiters at the input; add a small linear head at the output.
> 3. Fine-tune end-to-end with the task’s labelled loss.
> This recipe is the foundation of every modern LLM (with subsequent refinements like RLHF and instruction tuning, covered in later weeks).

## 3.10 Wishlist for Text Generation (Preview of Week 4)

The same auto-regressive generation produces wildly different outputs depending on *how we sample* from $P(x_i\mid x_{<i})$. Three goals:

- **Discourage degenerate output:** avoid repetition (“I like to think that I like to think that…”) and incoherence.
- **Encourage creativity:** for the same prompt, allow diverse continuations.
- **Accelerate generation:** make sampling fast even for long outputs.

These goals are exactly the topic of **Week 4: decoding strategies**. We will study greedy decoding, beam search, top-$k$, top-$p$ (nucleus), and temperature.

<!-- PDF p.41 / printed p.40 -->

## 3.11 Common Mistakes & Quiz Traps — Week 3

> **Common Mistake**
>
> **Mistake 1.** “GPT uses both the encoder and the decoder of the Transformer.”
> **Fix.** GPT is **decoder-only**. There is no encoder, so no cross-attention. The block is just masked self-attention + FFN.

> **Common Mistake**
>
> **Mistake 2.** “Language modelling means assigning the same probability to every grammatical sentence.”
> **Fix.** A language model assigns higher probabilities to sentences that are *frequent in real text*, not just grammatical. “The cat is on the mat” beats “A purple electron contemplates Tuesday” even though both are grammatical.

> **Common Mistake**
>
> **Mistake 3.** “The chain rule of probability requires independence.”
> **Fix.** The chain rule *never* requires independence — it is always true ($P(A,B)=P(A)P(B\mid A)$ by definition of conditional probability). The naive independence *assumption* (which we *rejected*) was a separate, wrong, simplification.

> **Quiz Trap**
>
> **Trap.** “Decoder-only models cannot do classification.”
> **Reality.** They can — read the hidden state at the last token and put a linear classifier on it. The *last* token’s hidden state has attended to everything before it (because of causal masking), so it acts as a sentence summary.

> **Quiz Trap**
>
> **Trap.** “Pre-training and fine-tuning use the same loss function.”
> **Reality.** Pre-training uses the LM (cross-entropy over the next token). Fine-tuning uses a *task-specific* loss (cross-entropy over classes for classification, etc.). GPT-1 famously mixes the two during fine-tuning to reduce forgetting — an early hint at multi-task learning.

## 3.12 Memory Tricks — Week 3

> **Memory Trick**
>
> **“Mask Cross FFN → Mask FFN.”**
> The GPT block is the Transformer decoder block with the cross-attention sub-layer *removed*. From three sub-layers down to two.

> **Memory Trick**
>
> **“BERT looks both ways, GPT only back.”**
> BERT is encoder-only (bidirectional context, MLM objective). GPT is decoder-only (left-to-right context, CLM objective).

<!-- PDF p.42 / printed p.41 -->

> **Memory Trick**
>
> **“Predict next token = learn language.”**
> The simple-seeming objective of next-token prediction forces the model to learn syntax, semantics, world knowledge, reasoning, style … all to get the probabilities right. This is the central insight of modern LLMs.

> **Memory Trick**
>
> **“GPT-1 is 117M: 31 embedding + 27 attention + 56 FFN.”**
> Roughly $1/4+1/4+1/2$. The FFN typically dominates the parameter budget.

> [possible-error p.42: The listed components total 114 M, not 117 M.]

## 3.13 Practical Applications

- **All modern LLMs** (GPT-3/4, Claude, Gemini, LLaMA, Mistral) are decoder-only Transformers trained with the same CLM objective — only larger.
- **In-context learning** (few-shot prompting): instead of fine-tuning, just give the model a few examples *in the prompt* and it can solve the task. Discovered with GPT-3 in 2020.
- **Zero-shot inference:** simply ask the model in natural language to do the task; no examples needed. Possible only because pre-training has internalised such a wide distribution of tasks.
- **Instruction-tuned models** (ChatGPT, Claude): GPT/LLaMA further fine-tuned on (instruction, response) pairs and RLHF. Covered in Week 8.

## 3.14 Quick Revision — Week 3 in One Page

> **Quick Revision**
>
> **Big picture.** Pre-training on raw text (LM objective) + fine-tuning on labelled data (task-specific loss) = the modern recipe. GPT-1 (2018) demonstrated this works: SOTA sentiment with 30–100× less labelled data.
> **Language model.** A function $f$ that assigns probabilities to sequences: $f(x_1,\ldots,x_T)\in[0,1]$.
> **Chain rule.** $P(x_1,\ldots,x_T)=\prod_iP(x_i\mid x_{<i})$ — always true; no independence needed.
> **Three flavours.**
> - Encoder-only → MLM (BERT, RoBERTa).
> - Decoder-only → CLM (GPT, LLaMA, Claude).
> - Encoder-decoder → seq2seq (T5, BART).
> **GPT-1 block.** Masked self-attention → FFN. Stack 12 blocks.
> **LM head.** Linear $W_v\in\mathbb{R}^{d\times|\mathcal{V}|}$ + softmax.
> **GPT-1 config.** $n=12$, $d=768$, $h=12$, $l=3072$, $T=512$, BPE, BookCorpus, $|\mathcal{V}|=40,478$. **117M parameters total.**
> **Pre-training loss.** $-\sum_i\log P(x_i\mid x_{<i};\theta)$.
> **Fine-tuning.** Initialise from pre-trained checkpoint; add delimiters ($\langle s\rangle,\langle\$\rangle,\langle e\rangle$); swap LM head for task head; train end-to-end. Read last-token hidden state for classification.

## 3.15 Practice Questions — Week 3

### Multiple Choice

1. **(Q1)** GPT-1 is best described as:

   (A) Encoder-only Transformer with masked language modelling.

<!-- PDF p.43 / printed p.42 -->

   (B) Decoder-only Transformer with causal language modelling.

   (C) Encoder-decoder Transformer.

   (D) A recurrent network with attention.

   **Solution.** (B).

2. **(Q2)** In Causal Language Modelling, the conditional probability $P(x_i\mid x_{<i})$ is enforced by:

   (A) adding positional encoding,

   (B) the causal (triangular) attention mask,

   (C) batch normalisation,

   (D) the GELU activation.

   **Solution.** (B). Without the mask, position $i$ could attend to future positions and “cheat”.

3. **(Q3)** In GPT-1’s parameter budget ($\sim117$M), the layer with the most parameters is:

   (A) embedding,

   (B) attention,

   (C) FFN,

   (D) LayerNorm.

   **Solution.** (C). FFN $\sim56$M > embedding $\sim31$M > attention $\sim27.6$M $\gg$ LayerNorm.

4. **(Q4)** For classification fine-tuning with GPT-1, the linear head reads the hidden state at:

   (A) the first token $\langle s\rangle$,

   (B) every position (averaged),

   (C) the last token $\langle e\rangle$,

   (D) the [CLS] token.

   **Solution.** (C). Because of causal masking, only the last position has attended to every preceding token. (A) and (D) are BERT conventions, not GPT.

5. **(Q5)** Pre-training + fine-tuning is more sample-efficient than training from scratch because:

   (A) pre-training initialises parameters with rich linguistic priors,

   (B) pre-training reduces the number of parameters,

   (C) fine-tuning uses smaller batch sizes,

   (D) the optimizer converges in zero steps.

   **Solution.** (A). The pre-trained weights encode general language structure that fine-tuning only needs to adapt.

### Conceptual / Short Answer

6. **(Q6)** Why was BookCorpus preferred over the 1B-words benchmark for GPT-1 pre-training?

   **Answer.** BookCorpus contains *long contiguous passages* (entire books). The 1B-words dataset shuffles sentences and breaks paragraph-level continuity. To learn *long-range dependencies* (entailment between distant sentences, narrative coherence, referent tracking), the model needs contiguous text.

<!-- PDF p.44 / printed p.43 -->

7. **(Q7)** How does GPT-1 handle a multiple-choice question with 4 options at inference?

   **Answer.** Run the model 4 times, once per (question + choice) pair. Each run produces a scalar score (linear layer on last token’s hidden state). Stack scores into a 4-vector, softmax across choices, predict the argmax.

8. **(Q8)** Explain in one sentence why GPT-1 uses learned positional embeddings instead of sinusoidal ones.

   **Answer.** GPT-1 has a fixed max context $T_{\max}=512$, so a learned lookup table ($512\times768$) is sufficient, simpler, and lets the model freely learn any positional pattern. The downside (inability to extrapolate beyond 512) was acceptable in 2018.

9. **(Q9)** Compare the pre-training objective of BERT (MLM) and GPT (CLM) in terms of context.

   **Answer.**
   - BERT (MLM): randomly mask 15% of tokens; predict each masked token from *both left and right* context. Bidirectional.
   - GPT (CLM): predict each token from *left context only*. Unidirectional / causal.

   BERT is better for understanding; GPT is better for generation.

### Numerical

10. **(Q10)** Suppose we increase GPT-1’s context length from 512 to 2048, keeping $d_{\text{model}}=768$, $h=12$, $n=12$. By how many parameters do the *position embeddings* grow?

    **Solution.** Position embedding table grows from $512\times768$ to $2048\times768$, i.e., from $\approx0.39$M to $\approx1.57$M. Net increase $\approx1.18$M parameters. (The attention sub-layer parameters do *not* grow with context length — only the compute and KV-cache costs scale with $T$.)

11. **(Q11)** Compute the FFN parameters per GPT-1 block exactly.

    **Solution.**
    $$W_1:768\cdot3072=2,359,296,\quad b_1:3072,\quad W_2:3072\cdot768=2,359,296,\quad b_2:768.$$
    Total $=4,722,432\approx4.72\ \mathrm{M}$. Across 12 blocks $\approx56.7\ \mathrm{M}$.

> [possible-error p.44: The stated total omits the two biases; including them gives 4,726,272 parameters per block.]

12. **(Q12)** A model has $d_{\text{model}}=768$ and a vocabulary $|\mathcal{V}|=50,000$. What is the size of the LM head $W_v$, and what fraction of GPT-1’s total parameters would it represent?

    **Solution.** $W_v\in\mathbb{R}^{768\times50,000}$, size $\approx38.4\ \mathrm{M}$. As a fraction of 117M: $\approx33\%$. *This is why all modern LLMs use weight tying (share embedding and output projection) — it saves $\sim30\%$ of parameters!*

13. **(Q13)** Write down the log-likelihood of the sentence “$\langle\text{go}\rangle$ I like cats” under a CLM, treating $\langle\text{go}\rangle$ as the start token (no prediction loss on it).

    **Solution.**
    $$\log P(\text{“I like cats”}\mid\langle\text{go}\rangle)=\log P(\text{I}\mid\langle\text{go}\rangle)+\log P(\text{like}\mid\langle\text{go}\rangle,\text{I})+\log P(\text{cats}\mid\langle\text{go}\rangle,\text{I},\text{like}).$$
    Maximising this $\Leftrightarrow$ minimising the per-position cross-entropies.
