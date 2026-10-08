# AGENTS.md — Instructions for Future Models & Contributors

Repo: **PINN** — a 4-week foundational learning phase in **Physics-Informed Neural Networks (PINNs) and Scientific Machine Learning (SciML)**. The goal of this 4-week curriculum is **unshakable mathematical, algorithmic, and computational fundamentals**, bridging classical differential equations and modern deep learning.

`month 1/week 1/week-1-report.md`, `month 1/week 2/week-2-report.md`, `month 1/week 3/week-3-report.md`, and `month 1/week 4/week-4-report.md` constitute the gold-standard academic monographs. All daily working notes in `days/w<N>d<D>.md` must maintain identical rigor, runnable verification, and unbroken graph connectivity.

Read this file completely. It encodes conventions established across all four weeks.

---

## 1. What this Repo Produces

```
PINN/
├── AGENTS.md                         # this file
├── Knowledge-Map.md                  # Obsidian-style MOC / hub — the graph entry point
├── pyproject.toml                    # uv environment specification
├── uv.lock                           # uv lockfile
├── generate_week<N>_plots.py         # one deterministic figure script per week (1 through 4)
├── notes/concepts/                   # atomic concept notes (the graph's connective tissue)
│   └── <concept-slug>.md
└── month <N>/week <N>/
    ├── week-<N>-report.md            # the weekly review article (gold standard)
    ├── week<N>-export.md             # preserved raw curriculum export / backup
    ├── week-<N>-export.md            # mirror copy for robust linking
    ├── assets/                       # every figure referenced by that week (*.png)
    └── days/                         # daily active working notes (Concept → Code → Output → Interpretation)
        ├── w<N>d1.md                 # e.g., w1d1.md, w1d2.md, ..., w4d5.md
        ├── w<N>d2.md
        ├── w<N>d3.md
        ├── w<N>d4.md
        └── w<N>d5.md
```

### Strict Naming Rules
- Week directories literally contain a space: `month 1/week 1/`, `month 1/week 2/`, `month 1/week 3/`, `month 1/week 4/`.
- Weekly review report filenames are lowercase-hyphenated with no spaces: `week-1-report.md`, `week-2-report.md`, `week-3-report.md`, `week-4-report.md`.
- Raw curriculum backup files are named `week<N>-export.md` and mirrored as `week-<N>-export.md`.
- Daily working notes live in `days/` and **must strictly be named `w<N>d<D>.md`** (e.g., `w1d1.md` through `w4d5.md`). This guarantees every file's week and day are instantly unambiguous across the Obsidian vault.
- Quote paths in all shell commands to safely handle folder spaces.

---

## 2. Environment — Exclusively `uv`

The project is `uv`-managed. **Never** use bare `python`, `pip install`, or `python3` to run or install anything.

```bash
uv add numpy matplotlib seaborn scipy torch   # manage dependencies
uv run python generate_week1_plots.py         # execute figure generation
uv run python <test_script.py>                # execute verification scripts
```

Plotting stack: **Seaborn for visual theme, Matplotlib underneath**. Use `sns.set_theme(style="whitegrid", context="talk", palette="deep")` at the top of every figure script, plus `figure.dpi=150` and `savefig.bbox="tight"`.

---

## 3. Figures — Mandatory and Deterministic

A weekly report or day note is **not** finished without embedded figures whenever concepts are visual (a function and its derivative, a vector gradient field, a computational graph, an activation function comparison, an optimization trajectory, a collocation point distribution, a differential operator residual field, or an ODE/PDE solution contour).

### Hard Rules
- **Destination:** Save PNGs directly into that week's `assets/` folder, never inline/base64, never external URLs.
- **Embedding paths:**
  - In `week-<N>-report.md`: Embed with `![Descriptive caption](assets/figure_name.png)`.
  - In `days/w<N>d<D>.md`: Embed with `![Descriptive caption](../assets/figure_name.png)`.
- **Captions:** Every figure gets a numbered caption paragraph immediately after it, in the form `**Figure N.** <what it shows> <the technical/physical point it makes>.` Number sequentially in document order.
- **One generation script per week:** `generate_week<N>_plots.py` generates all figures for that week deterministically.
- **Determinism:** Set `np.random.default_rng(42)` and `torch.manual_seed(42)` wherever randomness appears.

---

## 4. The 4-Week PINN Spine & Central Paradigm

The governing problem paradigm of Physics-Informed Neural Networks is:

$$\mathbf{x} \in \Omega \xrightarrow{\quad\text{Neural Network } u_\theta(\mathbf{x})\quad} u \xrightarrow{\quad\text{Automatic Differentiation}\quad} \left\{ \frac{\partial u}{\partial t}, \nabla u, \Delta u, \dots \right\} \xrightarrow{\quad\text{Differential Operator }\mathcal{N}[u] - f\quad} \mathcal{R}_{\text{physics}}(\mathbf{x})$$

$$\mathcal{L}_{\text{total}}(\theta) = w_{\text{data}} \mathcal{L}_{\text{data}}(\theta) + w_{\text{pde}} \mathcal{L}_{\text{pde}}(\theta) + w_{\text{bc}} \mathcal{L}_{\text{bc}}(\theta) + w_{\text{ic}} \mathcal{L}_{\text{ic}}(\theta)$$

| Week | Title | Core Question | Mathematical Engine |
| :--- | :--- | :--- | :--- |
| **Week 1** | Mathematical Foundations for Physics-Informed Neural Networks | How do continuous calculus, differential equations, and discretization errors lay the groundwork for physics constraints? | Differential calculus, ODE/PDE classification, initial/boundary value problems, finite difference residuals |
| **Week 2** | Fundamentals of Neural Networks and Optimization for Scientific Machine Learning | How do multi-layer perceptrons, forward-backward propagation, and gradient-based optimizers approximate continuous functions? | Affine maps $\mathbf{W}\mathbf{x} + \mathbf{b}$, non-linear activations $\sigma(\cdot)$, chain rule autograd, MSE loss surfaces, Adam & L-BFGS |
| **Week 3** | Physics-Informed Neural Networks: Concepts, Formulation and Learning Framework | How does embedding governing differential equations into the loss functional transmute deep learning into a gridless scientific solver? | Collocation point sampling, dual-graph automatic differentiation, residual loss formulation, soft constraint trade-offs |
| **Week 4** | Basic Implementation and Applications of Physics-Informed Neural Networks | How do we implement, train, balance, and validate a full PINN solver from scratch on canonical ODEs and PDEs? | PyTorch `autograd.grad`, harmonic oscillator ODE, 1D Heat / Burgers PDE, residual evaluation, analytical error quantification |

---

## 5. Report Structure — The Golden Template

Every weekly review report (`week-<N>-report.md`) follows the exact academic structure:
1. Frontmatter (YAML) with quoted wikilinks.
2. Academic Title (H1) and Blockquote Header.
3. In-body Navigation line and Associated Concepts.
4. Abstract (4–6 dense sentences) and Keywords.
5. Section 1: Introduction, governing rationale, inheritance from prior week, and Syllabus Table linking `[[w<N>d1\|**W<N>D1**]]` through `[[w<N>d5\|**W<N>D5**]]`.
6. Numbered Sections: Gapless, covering theoretical foundations, derivations, and canonical algorithms.
7. Concept → Code → Output → Interpretation subsections for every implementation.
8. Comparison Tables (bolded first column, limitation/pathology column).
9. Concrete Scientific Applications (Governing PDE, formulation, failure mode).
10. Critical Challenges & Failure Modes (e.g., spectral bias, stiff gradients, multi-objective Pareto front conflict).
11. Conclusion & Forward Pedagogical Handoff.
12. Academic References with verified authors, year, title, journal, DOI, and 1-sentence Contribution.
13. Appendix A (Computational Reproducibility with `uv`) & Appendix B (Obsidian Knowledge Graph Connections).

---

## 6. Daily Working Notes Convention (`days/w<N>d<D>.md`)

Daily working notes provide exhaustive, modular coverage across the 5 workdays of each week:
1. Frontmatter with `w<N>d<D-1>`, `w<N>d<D+1>`, `week-<N>-report`, and `Knowledge-Map`.
2. Section 1: Day Overview, Learning Objectives, and Position in the Weekly Pipeline.
3. Section 2: Diagnostic / Previous-Day Verification Questions with authoritative, mathematically thorough solutions.
4. Section 3–5: In-depth theoretical lecture notes, mathematical derivations, and physical interpretations.
5. Practical Lab Section: Complete, runnable Python script adhering to the **Concept → Code → Output → Interpretation** protocol.
6. Daily Knowledge Mastery Checkpoint: 4–6 rigorous problems with complete analytical and numerical solutions.
7. Pedagogical Bridge to the next day.

---

## 7. Quality Bar Checklist

- [ ] All 4 weekly reports created at `month 1/week <N>/week-<N>-report.md`.
- [ ] All 20 daily working notes created at `month 1/week <N>/days/w<N>d<D>.md`.
- [ ] All raw export backups created at `week<N>-export.md` and `week-<N>-export.md`.
- [ ] All plot generation scripts `generate_week1_plots.py` through `generate_week4_plots.py` executed cleanly via `uv run python`.
- [ ] All figures exist on disk in `month 1/week <N>/assets/` and render properly.
- [ ] All code blocks run without error and display exact console outputs.
- [ ] `Knowledge-Map.md` links every report and daily note without broken links.
- [ ] Atomic concept notes created in `notes/concepts/`.
- [ ] Academic citations include verified DOIs and contribution summaries.
