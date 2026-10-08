# AGENTS.md — General Multi-Track Technical Curriculum Specification

**Vault Root:** `Notes/`  
**Purpose:** Universal architecture and operational standards for building, maintaining, and verifying publication-grade, mathematically rigorous study vaults (e.g., `NeuroScience/`, `PINN/`, and future computational disciplines).

Any autonomous agent or human contributor working within this vault must treat this document as the authoritative standard. All weekly synthesis reports and daily working notes must maintain unwavering technical depth, pedagogical clarity, runnable verification, and unbroken graph connectivity.

---

## 1. Vault Topology & Directory Architecture

Every technical discipline in this repository is maintained as an independent, self-contained sub-vault under its own root folder (e.g., `Notes/PINN/`, `Notes/NeuroScience/`). Each track adheres to the following canonical layout:

```
<TrackName>/                          # e.g., PINN/ or NeuroScience/
├── AGENTS.md                         # Discipline-specific instructions, conventions, and formulas
├── Knowledge-Map.md                  # Central Obsidian MOC / Vault Hub — the global graph entry point
├── pyproject.toml                    # uv project definition with exact pinned/managed dependencies
├── uv.lock                           # Deterministic dependency lockfile
├── generate_week<N>_plots.py         # One deterministic figure generation script per week
├── notes/
│   └── concepts/                     # Atomic concept notes (connective tissue of the knowledge graph)
│       ├── <concept-slug-1>.md       # Lowercase-hyphenated slug
│       └── <concept-slug-2>.md
└── month <M>/
    └── week <N>/                     # Directory literally contains a space: "month 1/week 1/"
        ├── week-<N>-report.md        # The gold-standard weekly academic synthesis article
        ├── week<N>-export.md         # Preserved raw curriculum / syllabus export
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
3. **Daily Working Notes:** Must strictly follow the five-character slug: `w<N>d<D>.md` (e.g., `w1d1.md`, `w1d2.md` ... `w4d5.md`). Do not use verbose filenames for daily notes; the canonical slug allows instantaneous vault-wide wikilinking.
4. **Concept Notes:** Stored in `notes/concepts/<slug>.md` using lowercase-hyphenated nouns (e.g., `physics-residual.md`, `automatic-differentiation.md`).
5. **Raw Curriculum Exports:** Preserved as `week<N>-export.md` (and mirrored as `week-<N>-export.md`).

---

## 2. Environment Management — Exclusively `uv`

All Python code execution, dependency installation, and script running must be managed through `uv`. **Never** use system `python`, `pip`, or unmanaged virtual environments.

```bash
# Initialize a track environment (if not present)
uv init --name <track-name> --no-readme

# Add required computational dependencies
uv add numpy matplotlib seaborn scipy torch

# Execute scripts strictly via uv
uv run python generate_week<N>_plots.py
uv run python <test_script.py>
```

### Script Execution & Determinism
- All randomness must be seeded deterministically using `np.random.default_rng(42)`, `torch.manual_seed(42)`, or `torch.cuda.manual_seed_all(42)`.
- Rerunning any generation script must produce byte-identical figures and console outputs.
- Never write placeholder, pseudo-code, or unexecuted dummy blocks. Code must be tested and confirmed against the live environment.

---

## 3. Publication-Grade Visual Assets

Visual artifacts are mandatory for all physical, geometric, or algorithmic concepts (waveforms, loss landscapes, vector fields, residual distributions, network architectures, Pareto frontiers).

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
- **Unit Annotations:** Every axis must feature physical dimensions or mathematical units (e.g., $x \in [0, 1]$, Time $t$ (s), Residual $||\mathcal{R}||_2$, Loss $\mathcal{L}$).
- **Sequential Captioning:** Every embedded image in Markdown must be immediately succeeded by a bold numbered caption paragraph:
  ```markdown
  ![Comparative Loss Trajectories](assets/loss_comparison.png)

  **Figure 1.** *Empirical convergence of Adam versus L-BFGS optimization on the 1D harmonic oscillator PINN. The second-order optimizer eliminates high-frequency residual oscillations and achieves machine-precision boundary adherence within 200 epochs.*
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
  - <track>/week-0<N>
  - <track>/<primary-domain>
  - <track>/<sub-domain>
concepts:
  - "[[concept-slug-1]]"
  - "[[concept-slug-2]]"
---

# <Full Academic Title of the Review Article>

> **Week <N> Review Article — Month <M> / Week <N>**  
> Foundation learning phase: establishing rigorous mathematical, physical, and algorithmic principles.  
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
[Rigorous mathematical definitions, governing differential/algebraic formulations, boundary conditions, and theorems.]

## 3. Core Formulation & Algorithmic Machinery
[Detailed derivations, computational graphs, and discrete operator approximations.]

### 3.X Concept → Code → Output → Interpretation
[Fenced code block, unedited console output, and deep technical interpretation.]

---

## 4. Numerical Analysis & Comparative Benchmarks
[Comparison tables, convergence trade-offs, stability criteria.]

---

## 5. Practical Implementation & Verification
[End-to-end Python/PyTorch pipeline executing a complete canonical problem.]

---

## 6. Real-World Applications & Scientific Formulations
[3–4 concrete engineering or scientific domains with Target, Governing Equations, and Failure Modes.]

---

## 7. Critical Challenges, Pathology & Failure Modes
[Stiffness, gradient pathologies, spectral bias, boundary violations.]

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
4. **Diagnostic / Previous-Day Verification:** 4–8 technical checkpoint questions verifying prerequisites with **complete, scientifically authoritative answers**.
5. **Core Topics:** Detailed theoretical derivations with LaTeX math.
6. **Concept → Code → Output → Interpretation:** Complete, self-contained Python scripts executed via `uv run` with exact terminal outputs pasted verbatim.
7. **Daily Knowledge Mastery Checkpoint:** Conceptual and numerical problems testing day material with full explanatory solutions.
8. **Summary & Bridge:** Handoff to the next day's topic.

---

## 6. The "Concept → Code → Output → Interpretation" Method

Every empirical claim or computational mechanism must follow this 4-stage pipeline:

```markdown
### 3.2 Physics Residual Computation via Automatic Differentiation

#### 1. Concept
The neural network $u_\theta(t, x)$ acts as a differentiable surrogate. We differentiate $u_\theta$ with respect to spatial coordinate $x$ and temporal coordinate $t$ using PyTorch's reverse-mode automatic differentiation engine (`torch.autograd.grad`), constructing the differential operator without discrete grid approximations.

#### 2. Code
```python
import torch

# Coordinates requiring gradient tracking
x = torch.tensor([[0.5]], requires_grad=True)
t = torch.tensor([[1.0]], requires_grad=True)

# Surrogate MLP approximation
u = torch.sin(x) * torch.exp(-0.1 * t)

# First- and second-order derivatives via autodiff
u_t = torch.autograd.grad(u, t, create_graph=True)[0]
u_x = torch.autograd.grad(u, x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, create_graph=True)[0]

# PDE Residual: Heat equation R = u_t - alpha * u_xx (alpha = 0.1)
residual = u_t - 0.1 * u_xx
print(f"u: {u.item():.6f}")
print(f"u_t: {u_t.item():.6f}, u_xx: {u_xx.item():.6f}")
print(f"Physics Residual R: {residual.item():.6f}")
```

#### 3. Output
```text
u: 0.433854
u_t: -0.043385, u_xx: -0.433854
Physics Residual R: 0.000000
```

#### 4. Interpretation
The analytical derivative matches the autograd derivative to within float32 precision. The computed residual $\mathcal{R} = 0.0$ confirms that the analytical function exactly satisfies the homogeneous heat equation. In a PINN, $\mathcal{R}$ is minimized across thousands of collocation points $\mathbf{x}_c \sim \Omega$.
```

---

## 7. Writing Style & Pedagogical Tone

- **Uncompromising Academic Rigor:** Write with the precision of a high-impact journal review (e.g., *Nature Reviews*, *SIAM Review*, *Journal of Computational Physics*).
- **Zero Fluff & Zero Conversational Filler:** Never use phrases like "In this section we will look at", "Let's explore", or "As we all know". State the physical and mathematical claims directly.
- **Bolded Rhetorical Lead-ins:** Guide the reader's attention using structural markers:
  - `**The Physical Connection:**`
  - `**Mathematical Formulation:**`
  - `**The Dangerous Misconception:** ... **Why this is false:**`
  - `**Algorithmic Implication:**`
  - `**Design Rule of Thumb:**`
  - `**Output & Interpretation:**`
- **Expose Common Misconceptions:** Systematically dismantle intuitive fallacies (e.g., confusing finite differences with automatic differentiation; assuming zero training loss guarantees PDE compliance).
- **Strict Dimensional Notation:** Format all tensor shapes, coordinate tuples, and vector dimensions in backticks: `[Batch, 1]`, `(N_colloc, 2)`, `\mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}`.
- **Rigorous LaTeX Equations:** Use inline math `$f(x)$` for variables and parameters; use display math `$$...$$` for multi-line derivations, loss functionals, and boundary operators.

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

- [ ] All required files are in their canonical directories (`week-<N>-report.md`, `w<N>d1.md`–`w<N>d5.md`, `week<N>-export.md`, `assets/*.png`).
- [ ] Every figure is generated deterministically via `uv run python generate_week<N>_plots.py`.
- [ ] Every image reference resolves to a valid file on disk (`assets/...` in reports; `../assets/...` in day notes).
- [ ] Every image has a sequential `**Figure N.**` caption detailing its technical interpretation.
- [ ] All code blocks are complete, syntactically valid, and have executed via `uv run python`.
- [ ] Terminal outputs in Markdown fences represent real console executions—zero invented numbers.
- [ ] All YAML frontmatter blocks parse cleanly and quote their wikilinks.
- [ ] Syllabus table in Section 1 of the weekly report correctly links `[[w<N>d1\|**W<N>D1**]]` through `[[w<N>d5\|**W<N>D5**]]`.
- [ ] All internal wikilinks resolve to existing files with zero orphaned or dangling links.
- [ ] Formal references are academically complete with authors, titles, publication venues, years, and DOIs.
