# AGENTS.md — LLM Track Specification

**Vault Root:** `LLM/`  
**Discipline:** Large Language Models, IIT Madras DA5004-style course  
**Parent Curriculum:** `Notes/`

This track consolidates ebook-scoped Weeks 1–4 into weekly notes and exam resources. Later-week content is archived. Follow the vault-wide standard and these track-specific conventions.

## 1. Directory Architecture

```text
LLM/
├── AGENTS.md
├── Knowledge-Map.md
├── pyproject.toml, uv.lock
├── generate_weekN_plots.py
├── notes/concepts/<concept-slug>.md
├── sources/                       # ebook, papers, topic list, decisions and exam sources
├── scripts/                       # inventory and routing tools
└── week N/
    ├── week-N-notes.md
    ├── week-N-exam-focused-notes.md
    ├── week-N-exam-questions.md   # planned where absent
    ├── assets/
    └── codeblocks/
```

Week directories contain literal spaces. Weekly notes consolidate the material; do not manufacture daily notes or month folders for this track. Store figures in the corresponding `assets/` directory and code with recorded output in `codeblocks/`. Embed figures using relative paths and immediately follow each with a numbered, technically informative caption. Quote frontmatter wikilinks and escape alias pipes in tables. Link every concept through the knowledge map.

## 2. Execution and Notation

Use `uv` exclusively for Python execution and dependency management; do not use system Python, pip, or unmanaged environments. Seed stochastic work with 42, including NumPy and PyTorch as applicable. Execute code and preserve actual output; deterministic figure scripts must reproduce their assets.

| Symbol | Meaning |
| :--- | :--- |
| $d_{model}, d_k, d_v$ | Model, per-head query/key, and per-head value widths |
| $h, d_{ff}$ | Attention head count and FFN hidden width |
| $T, B, V$ | Sequence length, batch size, vocabulary size |
| $Q,K,V$ | Query, key and value matrices; distinguish value matrix $V$ from vocabulary size by context |
| $W^Q,W^K,W^V,W^O$ | Query, key, value and output projection weights |
| $\theta, L$ | Model parameters and number of layers |
| $\epsilon_{ls}$ | Label-smoothing mass |
| $k,p,\tau$ | Beam width or top-k cutoff (context dependent), nucleus threshold, sampling temperature |
| $N$ | Token count in a sequence or evaluation set, as specified locally |

Use row-oriented tensors: inputs `[B, T, d_model]`; projected per-head queries `[B, h, T_q, d_k]`, keys `[B, h, T_k, d_k]`, values `[B, h, T_k, d_v]`; attention weights `[B, h, T_q, T_k]`. State masks and broadcasting explicitly. For standard equal-width heads, $hd_k=hd_v=d_{model}$.

## 3. Compact Formula Sheet

Scaled attention and multi-head attention, with additive mask $M$ (zero for allowed entries and $-\infty$ for forbidden entries):

$$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\quad \operatorname{MultiHead}=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O,$$
$$\mathrm{head}_i=\operatorname{Attn}(QW_i^Q,KW_i^K,VW_i^V).$$

Position-wise FFN and sinusoidal positional encoding, with zero-based position $pos$ and channel-pair index $i$:

$$\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2,$$
$$PE_{pos,2i}=\sin(pos/10000^{2i/d_{model}}),\quad PE_{pos,2i+1}=\cos(pos/10000^{2i/d_{model}}).$$

LayerNorm uses feature-wise population moments for each token; $\epsilon_{LN}$ is numerical stabilization, distinct from label smoothing:

$$\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon_{LN}}}+\beta.$$

For uniform smoothing over all $V$ classes, $q_j=(1-\epsilon_{ls})\mathbf{1}[j=y]+\epsilon_{ls}/V$ and $\mathcal L=-\sum_j q_j\log p_j$. Other conventions distribute mass only over non-target classes; name the convention. With step $s>0$ and warm-up $w$ the original schedule is $\mathrm{lr}(s)=d_{model}^{-1/2}\min(s^{-1/2},s w^{-3/2})$.

$$P_\theta(x_{1:N})=\prod_{t=1}^{N}P_\theta(x_t\mid x_{<t}),\quad \operatorname{PPL}=\exp\!\left(-\frac1N\sum_{t=1}^{N}\log P_\theta(x_t\mid x_{<t})\right).$$

**Parameter-count conventions:** Let $D=d_{model}$, $F=d_{ff}$, $P$ be position capacity, and $S$ the number of segment types. With biased projections and affine LayerNorms, an encoder layer is $4D^2+4D+2DF+F+D+4D$; a decoder layer adds another attention block and LayerNorm: $8D^2+8D+2DF+F+D+6D$. For $D=512$, $h=8$, $F=2048$, these give **3,152,384** and **4,204,032**, respectively, including biases and two encoder or three decoder LayerNorms. Recorded formula and PyTorch counts: `week 1/codeblocks/04_encoder_parameter_count.txt` and `week 2/codeblocks/07_decoder_parameter_count.txt`.

The recorded GPT-1 convention ties token and LM-head weights, includes projection/FFN biases and two affine LayerNorms per block, and has no LM-head bias:

$$C_{GPT}=VD+PD+L(4D^2+4D+2DF+F+D+4D).$$

For $V=40478$, $P=512$, $D=768$, $F=3072$, $L=12$, this is **116,534,784** (`week 3/codeblocks/04_gpt1_parameters.txt`).

The recorded BERT convention counts token, position and segment embeddings, embedding LayerNorm, encoder blocks and pooler; it excludes pre-training heads:

$$C_{BERT}=VD+PD+SD+2D+L(4D^2+4D+2DF+F+D+4D)+D^2+D.$$

With $V=30522$, $P=512$, $S=2$, base ($D=768,F=3072,L=12$) is **109,482,240** and large ($D=1024,F=4096,L=24$) is **335,141,888**, as recorded in `week 4/codeblocks/04_bert_parameters.txt`. Counts depend on vocabulary and included components.

## 4. Scope, Sources and Routing

`sources/topics.json` is the canonical topic list. A question belongs to Weeks 1–4 only when its tested content appears in those weeks' topics; archive later content rather than forcing it into an earlier week. Preserve provenance and uncertainty for each answer: all answers are derived, not official. The ebook is a scope guide, not an authority; when it conflicts with the primary paper, the paper wins and the discrepancy belongs in `sources/ebook-discrepancies.md`.

## 5. Rebuilding

From `LLM/`, run the existing weekly figure scripts and inventory/routing tools as needed:

```bash
cd LLM
uv run python generate_week1_plots.py
uv run python generate_week2_plots.py
uv run python generate_week3_plots.py
uv run python generate_week4_plots.py
uv run python scripts/build_inventory.py
uv run python scripts/route_draft.py
uv run python scripts/merge_routing.py
```

Bank, index and priority builders and `scripts/check_links.py` are planned and have no rebuild command until implemented. Review generated routing against the canonical topic list before use.

## 6. Quality Verification

- [ ] Weekly notes, exam-focused notes, question banks, assets and code outputs occupy their canonical locations; mark planned resources honestly.
- [ ] Regenerate figures through `uv` and verify deterministic output, relative image paths, and sequential captions with technical interpretations.
- [ ] Execute complete code blocks via `uv`; reproduce recorded terminal output without invented numbers.
- [ ] Parse YAML frontmatter, quote wikilinks, and escape table alias pipes.
- [ ] Check every internal wikilink and concept connection; distinguish explicitly planned targets from accidental dangling links.
- [ ] Verify topic routing against `sources/topics.json` and archive later material.
- [ ] Check calculations, dimensions, assumptions, references and discrepancies against primary sources; supply complete citations and DOIs where available.
- [ ] Verify answer provenance and identify unresolved or image-dependent evidence rather than presenting guesses as official answers.

## 7. Conventions decided in review

- The immediately following, sequentially numbered figure-caption rule applies to authored figures in study notes (`week-N-notes.md`, `week-N-exam-focused-notes.md`, and concept notes). It does **not** apply to question scans in exam banks: those images reproduce source evidence, not authored figures.
- Link to bank questions using their actual unique question headings in Obsidian (`week-N-exam-questions#2024 T1 Quiz 1 NA, Q186`). Stable full question IDs in HTML anchors identify records for scripts and external consumers; they are not a substitute for Obsidian heading or block links. Put topic-link targets in headings outside tables.
- Build scripts must apply `sources/exam-papers/routing-overrides.json`, `question-corrections.json`, and `solutions-overrides.json` when generating downstream routing, banks, and indexes. Preserve the original source extractions unedited. Treat shared-context identifiers in corrections as source passage anchors, include their passage text and images, and never include an answer merely to provide context. Existing builders must be updated to implement this policy before claiming override-aware rebuilds.
