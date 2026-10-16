# AGENTS.md — Vision Transformers Foundation Curriculum Specification

**Vault Root:** `Notes/Vision Transformers/`  
**Discipline:** Computer Vision, Convolutional Architectures, Self-Attention Mechanisms, and Vision Transformers (ViT)  
**Pedagogical Phase:** Month 1 Foundation Phase — Strong mathematical, architectural, and empirical fundamentals.

Any autonomous agent or human contributor working within this vault must treat this document as the authoritative standard. All weekly synthesis reports and daily working notes must maintain unwavering technical depth, pedagogical clarity, runnable verification, and unbroken graph connectivity.

---

## 1. Vault Topology & Directory Architecture

The `Vision Transformers` track is maintained as an independent, self-contained sub-vault under `Notes/Vision Transformers/`. It adheres strictly to the canonical layout established across the workspace:

```
Vision Transformers/
├── AGENTS.md                         # This file: track-specific conventions, standards, and rules
├── Knowledge-Map.md                  # Central Obsidian MOC / Vault Hub — the global graph entry point
├── pyproject.toml                    # uv project definition with pinned dependencies
├── generate_week<N>_plots.py         # Deterministic figure generation script per week
├── notes/
│   └── concepts/                     # Atomic concept notes (connective tissue of the knowledge graph)
│       ├── <concept-slug-1>.md       # Lowercase-hyphenated slug
│       └── <concept-slug-2>.md
└── month <M>/
    └── week <N>/                     # Directory literally contains a space: "month 1/week 1/"
        ├── week-<N>-report.md        # The gold-standard weekly academic synthesis article
        ├── week<N>-export.md         # Preserved raw curriculum / syllabus export
        ├── week-<N>-export.md        # Mirrored raw export
        ├── assets/                   # Every figure generated for and referenced by Week <N>
        │   ├── <fig_name_1>.png
        │   └── <fig_name_2>.png
        └── days/                     # Daily active working notes (Concept → Code → Output → Interpretation)
            ├── w<N>d1.md             # Canonical naming: w<N>d1.md through w<N>d5.md
            ├── w<N>d2.md
            ├── w<N>d3.md
            ├── w<N>d4.md
            └── w<N>d5.md
```

### Strict Naming Rules
1. **Week Folders:** Must contain literal spaces: `month 1/week 1/`, `month 1/week 2/`, etc.
2. **Weekly Reports:** Must be lowercase-hyphenated with no spaces: `week-1-report.md`.
3. **Daily Working Notes:** Must strictly follow the five-character slug: `w<N>d<D>.md` (e.g., `w1d1.md`, `w1d2.md` ... `w1d5.md`). Do not use verbose filenames for daily notes; the canonical slug allows instantaneous vault-wide wikilinking.
4. **Concept Notes:** Stored in `notes/concepts/<slug>.md` using lowercase-hyphenated nouns (e.g., `convolution.md`, `vision-transformer.md`, `receptive-field.md`).
5. **Raw Curriculum Exports:** Preserved as `week<N>-export.md` and mirrored as `week-<N>-export.md`.

---

## 2. Environment Management — Exclusively `uv`

All Python code execution, dependency installation, and script running must be managed through `uv`. **Never** use system `python`, `pip`, or unmanaged virtual environments.

```bash
# Initialize track environment (if not present)
uv init --name vision-transformers --no-readme

# Add required computational dependencies
uv add numpy matplotlib seaborn scipy torch torchvision

# Execute scripts strictly via uv
uv run python generate_week<N>_plots.py
uv run python <test_script.py>
```

### Script Execution & Determinism
- All randomness must be seeded deterministically using `np.random.default_rng(42)` or `torch.manual_seed(42)`.
- Rerunning any generation script must produce byte-identical figures and console outputs.
- Never write placeholder, pseudo-code, or unexecuted dummy blocks. Code must be tested and confirmed against the live environment.

---

## 3. Publication-Grade Visual Assets

Visual artifacts are mandatory for all physical, geometric, or algorithmic concepts (pixel channel decomposition, convolutional kernels, receptive field diagrams, feature embeddings, attention maps, transformer encoders).

### Plotting Standards
- **Style Stack:** Seaborn on top of Matplotlib.
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.set_theme(style="whitegrid", context="talk", palette="deep")
  plt.rcParams["figure.dpi"] = 150
  plt.rcParams["savefig.bbox"] = "tight"
  ```
- **File Format & Destination:** Save all figures as high-resolution PNGs directly into `month <M>/week <N>/assets/`.
- **Sequential Captioning:** Every embedded image in Markdown must be immediately succeeded by a bold numbered caption paragraph:
  ```markdown
  ![Comparative Receptive Field Growth](assets/receptive_field_vs_vanishing_gradient.png)

  **Figure 1.** *Empirical analysis of convolutional receptive field expansion versus layer depth alongside the corresponding gradient decay curve across deep feedforward stacks.*
  ```
- **Relative Linking Paths:**
  - Inside `week-<N>-report.md`: `![Caption](assets/<figure_name>.png)`
  - Inside `days/w<N>d<D>.md`: `![Caption](../assets/<figure_name>.png)` (must ascend one directory level)

---

## 4. The Golden Report Skeleton (`week-<N>-report.md`)

Each weekly report is a definitive, monograph-quality academic review article synthesizing the entire week's theoretical and practical material.

```markdown
---
title: "<Full Academic Title of the Review Article>"
week: <N>
month: <M>
status: complete
date: YYYY-MM-DD
prev: "[[week-<N-1>-report|Week <N-1>]]"
next: "[[week-<N+1>-report|Week <N+1>]]"
map: "[[Knowledge-Map]]"
tags:
  - vt/week-0<N>
  - vt/<primary-domain>
  - vt/<sub-domain>
concepts:
  - "[[concept-slug-1]]"
  - "[[concept-slug-2]]"
---

# <Full Academic Title of the Review Article>

> **Week <N> Review Article — Month <M> / Week <N>**  
> Foundation learning phase: establishing rigorous mathematical, architectural, and empirical principles.  
> Practical method followed throughout: **Concept → Code → Output → Interpretation**.

**Navigation:** ← [[week-<N-1>-report|Week <N-1>]] · [[Knowledge-Map|Knowledge Map]] · [[week-<N+1>-report|Week <N+1>]] →  
**Associated Concepts:** [[concept-slug-1]] · [[concept-slug-2]] · [[concept-slug-3]]

---

## Abstract
[4–6 dense, publication-grade sentences stating the exact mathematical and engineering question the week resolves, the methodologies developed, and the quantitative conclusions demonstrated.]

## Keywords
Keyword 1, Keyword 2, `Tensor [Shape]`, Operator $\mathcal{N}[u]$, Loss Formulation

---

## 1. Introduction & Week <N> Syllabus
[Articulate the governing paradigm, establish the inheritance from the preceding week, and define why standard methods fail or require this foundation.]

### Week <N> Syllabus & Laboratory Sequence
| Day | Core Focus | Laboratory / Practical Tasks |
| :--- | :--- | :--- |
| [[w<N>d1\|**W<N>D1**]] | <Core Theoretical Focus> | <Concrete Computational Lab> |
| [[w<N>d2\|**W<N>D2**]] | <Core Theoretical Focus> | <Concrete Computational Lab> |
| [[w<N>d3\|**W<N>D3**]] | <Core Theoretical Focus> | <Concrete Computational Lab> |
| [[w<N>d4\|**W<N>D4**]] | <Core Theoretical Focus> | <Concrete Computational Lab> |
| [[w<N>d5\|**W<N>D5**]] | <Core Theoretical Focus> | <Concrete Computational Lab> |

---

## 2. Theoretical Foundations
[Rigorous mathematical definitions, governing operations, spatial tensors, and theorems.]

## 3. Core Formulation & Algorithmic Machinery
[Detailed derivations, computational graphs, and discrete operator approximations.]

### 3.X Concept → Code → Output → Interpretation
[Fenced code block, unedited console output, and deep technical interpretation.]

---

## 4. Architectural Analysis & Comparative Benchmarks
[Comparison tables, receptive field vs. self-attention trade-offs, inductive bias vs. scale.]

---

## 5. Practical Implementation & Verification
[End-to-end Python/PyTorch pipeline executing a complete canonical problem.]

---

## 6. Real-World Applications & Computer Vision Formulations
[3–4 concrete engineering or scientific domains with Target, Governing Equations, and Failure Modes.]

---

## 7. Critical Challenges, Pathology & Failure Modes
[Quadratic complexity $\mathcal{O}(N^2)$, lack of inductive bias on small data, position resolution.]

---

## 8. Conclusion & Pedagogical Handoff
[Numbered recap of proven principles and an explicit forward bridge to Week <N+1>.]

---

## 9. Academic References & Annotated Literature
[Formal citations: Author(s), Year, Title, Journal/Conference, DOI link, accompanied by a 1-sentence "Contribution" summary.]

---

## Appendix A — Computational Reproducibility (UV + Python)
[Exact commands to replicate all numerical experiments.]

## Appendix B — Knowledge Graph Connections
[Explicit Upstream, Downstream, and Sibling Wikilinks.]
```

---

## 5. Daily Working Notes Standard (`days/w<N>d<D>.md`)

Daily notes are granular, working lab notebooks where theory meets direct execution.

### Daily Structure Requirements
1. **Frontmatter:** Complete YAML block with title, week, day, status, tags, quoted links to previous day, next day, parent report, and map.
2. **Navigation Banner:** Immediate reciprocal links.
3. **Overview & Day Goals:** Clear pedagogical purpose.
4. **Diagnostic / Foundational Verification:** 4–8 technical checkpoint questions verifying prerequisites with **complete, scientifically authoritative answers**.
5. **Core Topics:** Detailed theoretical derivations with LaTeX math.
6. **Concept → Code → Output → Interpretation:** Complete, self-contained Python scripts executed with exact terminal outputs pasted verbatim.
7. **Daily Knowledge Mastery Checkpoint:** Conceptual and numerical problems testing day material with full explanatory solutions.
8. **Summary & Bridge:** Handoff to the next day's topic.

---

## 6. The "Concept → Code → Output → Interpretation" Method

Every empirical claim or computational mechanism must follow this 4-stage pipeline:

```markdown
### 3.X Patch Linear Projection Tensor Mechanics

#### 1. Concept
An image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is partitioned into $N = \frac{HW}{P^2}$ non-overlapping spatial patches of shape $P \times P \times C$. Each patch is flattened into $\mathbf{x}_p \in \mathbb{R}^{P^2 C}$ and mapped to model dimension $D$ via a learnable linear projection $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$.

#### 2. Code
```python
import torch
import torch.nn as nn

B, C, H, W = 1, 3, 224, 224
P = 16
D = 768
N = (H // P) * (W // P)

x = torch.randn(B, C, H, W)
proj = nn.Conv2d(in_channels=C, out_channels=D, kernel_size=P, stride=P)
tokens = proj(x).flatten(2).transpose(1, 2)
print("Tokens shape:", tokens.shape)
```

#### 3. Output
```text
Tokens shape: torch.Size([1, 196, 768])
```

#### 4. Interpretation
The 2D strided convolution with `kernel_size=P` and `stride=P` operates identically to extracting flattened patches followed by matrix multiplication with $\mathbf{E}$. The resulting token tensor has shape $[B, 196, 768]$, matching $N = 196$ image tokens projected into dimension $D = 768$.
```

---

## 7. Writing Style & Pedagogical Tone

- **Uncompromising Academic Rigor:** Write with the precision of top-tier AI and computer vision venues (*CVPR*, *ICCV*, *ICLR*, *NeurIPS*, *IEEE TPAMI*).
- **Zero Fluff & Zero Conversational Filler:** Never use phrases like "In this section we will look at", "Let's explore", or "As we all know". State the mathematical and architectural claims directly.
- **Bolded Rhetorical Lead-ins:** Guide the reader's attention using structural markers:
  - `**The Vision Connection:**`
  - `**Mathematical Formulation:**`
  - `**The Dangerous Misconception:** ... **Why this is false:**`
  - `**Algorithmic Implication:**`
  - `**Design Rule of Thumb:**`
  - `**Output & Interpretation:**`
- **Expose Common Misconceptions:** Systematically dismantle intuitive fallacies (e.g., confusing translation equivariance with translation invariance; assuming ViT learns local features automatically without large-scale pretraining; confusing receptive field with effective receptive field).
- **Strict Dimensional Notation:** Format all tensor shapes, coordinate tuples, and vector dimensions in backticks: `[Batch, Channels, Height, Width]`, `[B, N, D]`, `(H/P \times W/P)`.
- **Rigorous LaTeX Equations:** Use inline math `$f(x)$` for variables; display math `$$...$$` for multi-line derivations, attention formulations, and loss functions.

---

## 8. Obsidian Knowledge Graph Integration

1. **Quoted Frontmatter Wikilinks:** All YAML frontmatter wikilinks must be wrapped in double quotes: `prev: "[[week-1-report|Week 1]]"`.
2. **Escaped Pipes in Tables:** When embedding wikilinks inside Markdown pipe tables, escape the alias pipe: `[[w1d1\|**W1D1**]]`. Unescaped pipes break Markdown table column parsers.
3. **Atomic Concept Notes (`notes/concepts/<slug>.md`):**
   - Provide a 100–250 word precise formal definition.
   - Outline "Why it matters here".
   - Include an `**Appears in:**` line backlinking to weekly reports and daily notes.
   - Include a `## Related` section linking sibling concept notes.
4. **Knowledge-Map Hub (`Knowledge-Map.md`):**
   - Main spine table tracking Week, Report Link, Core Question, and Status.
   - Dedicated daily notes listing linking every `w<N>d<D>` file.
   - Concept taxonomy grouping related nodes.
   - Central governing paradigm diagram.

---

## 9. Quality Verification Protocol

Before declaring any report or daily module complete, execute the following verification checklist:

- [ ] All required files are in their canonical directories (`week-<N>-report.md`, `w1d1.md`, `week1-export.md`, `assets/*.png`).
- [ ] Every figure is generated deterministically via `generate_week<N>_plots.py`.
- [ ] Every image reference resolves to a valid file on disk (`assets/...` in reports; `../assets/...` in day notes).
- [ ] Every image has a sequential `**Figure N.**` caption detailing its technical interpretation.
- [ ] All code blocks are complete, syntactically valid, and verified.
- [ ] Terminal outputs in Markdown fences represent real console executions—zero invented numbers.
- [ ] All YAML frontmatter blocks parse cleanly and quote their wikilinks.
- [ ] Syllabus table in Section 1 of the weekly report correctly links `[[w<N>d1\|**W<N>D1**]]`.
- [ ] All internal wikilinks resolve to existing files with zero orphaned or dangling links.
- [ ] Formal references are academically complete with authors, titles, publication venues, years, and DOIs.
