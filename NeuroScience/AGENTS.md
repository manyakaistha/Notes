# AGENTS.md — Instructions for Future Models

Repo: **NeuroScience** — a 4-week foundation-learning phase in **AI + Neuroscience** (team of 5, weekly review articles). The goal of the first four weeks is **strong fundamentals**, not novel research, not SOTA models.

**`month 1/week 1/week-1-report.md` and `month 1/week 2/week-2-report.md` are the gold-standard reference implementations.** New reports and daily working notes must be *indistinguishable in style* from them, and must *continue the same argument* rather than restart it. Read Weeks 1 and 2 in full before writing anything.

Read this file completely. It encodes conventions established across Weeks 1 and 2.

---

## 1. What this repo produces

```
NeuroScience/
├── AGENTS.md                    # this file
├── Knowledge-Map.md             # Obsidian-style MOC / hub — the graph entry point
├── generate_week<N>_plots.py    # one figure script per week
├── notes/concepts/              # atomic concept notes (the graph's connective tissue)
│   └── <concept-slug>.md
└── month <N>/week <N>/
    ├── week-<N>-report.md       # the weekly review article (gold standard)
    ├── week<N>-export.md        # preserved raw export / backup of curriculum text
    ├── assets/                  # every figure referenced by that week
    │   └── *.png
    └── days/                    # daily active working notes (Concept → Code → Output → Interpretation)
        ├── w<N>d1.md            # e.g., w2d1.md, w2d2.md, ...
        ├── w<N>d2.md
        ├── w<N>d3.md
        ├── w<N>d4.md
        └── w<N>d5.md
```

Naming conventions:
- Week directories literally contain a space: `month 1/week 1/`, `month 1/week 2/`.
- Weekly review report filenames are lowercase-hyphenated with no spaces: `week-1-report.md`, `week-2-report.md`.
- Raw curriculum backup files are named `week<N>-export.md` (and mirrored as `week-<N>-export.md`).
- Daily working notes live in `days/` and **must strictly be named `w<N>d<D>.md`** (e.g., `w2d1.md`, `w2d2.md`, `w2d3.md`, `w2d4.md`, `w2d5.md`). This guarantees every file's week and day are instantly unambiguous across the Obsidian vault.
- Quote paths in all shell commands to safely handle folder spaces.

---

## 2. Environment — always use `uv`

The project is `uv`-managed. **Never** use bare `python`, `pip install`, or `python3` to run or install anything.

```bash
uv init --name neuroscience --no-readme   # only if no pyproject.toml exists yet
uv add numpy matplotlib seaborn scipy     # add deps this way
uv run python <script.py>                # ALWAYS execute via uv run
```

Current dependencies: `numpy`, `matplotlib`, `seaborn`, `scipy`. Add more with `uv add`, never `pip`.

**Plotting stack: seaborn for UI, matplotlib underneath.** Use `sns.set_theme(style="whitegrid", context="talk", palette="deep")` at the top of every figure script, plus `figure.dpi=150` and `savefig.bbox="tight"`. Prefer `sns.lineplot`, `sns.heatmap`, `sns.barplot` over raw matplotlib calls where they fit.

---

## 3. Figures — mandatory, and how to make them

A weekly report or day note is **not** finished without embedded figures whenever concepts are visual (a waveform, an aliased trace, a filter response, a spectrum, a heatmap, a trade-off, a tensor shape). Generate plots with `uv run python` and embed them.

Hard rules:

- **Save PNGs into that week's `assets/` folder**, never inline/base64, never external URLs.
- **Embedding paths:**
  - In `week-<N>-report.md`: Embed with `![Descriptive caption](assets/figure_name.png)`.
  - In `days/w<N>d<D>.md`: Embed with `![Descriptive caption](../assets/figure_name.png)`.
- **Every figure gets a numbered caption paragraph immediately after it**, in the form `**Figure N.** <what it shows> <the point it makes>.` Number sequentially in document order.
- **One generation script per week** (`generate_week<N>_plots.py`) that generates *all* that week's figures deterministically. Print a manifest of saved files at the end.
- **Determinism:** Set `np.random.default_rng(42)` or `np.random.seed(42)` wherever noise/randomness appears, so reruns are byte-identical.
- **Self-check before writing:** Verify every `assets/...` reference in the markdown resolves to a real file on disk. A broken image link is a defect.
- **Standard figure types** (reuse these):
  - Side-by-side comparison panels (e.g. continuous vs. discrete sampling, synchrony vs. asynchrony)
  - Stacked subplots sharing an x-axis (component decomposition, multi-stage filtering, per-channel traces)
  - `sns.heatmap` for anything `[Channels × Time]` or matrix-shaped
  - Scatter/annotated plot for trade-off maps (space vs. time)
  - Shared-axis band comparison (frequency bands on a common µV scale)
  - Filter magnitude frequency response plots with cutoff annotations
  - Dimensionality compression diagrams
- Label axes with real physical units (`Time (s)`, `µV`, `Frequency (Hz)`), title every panel, and make the caption state the *interpretation*, not just the contents.
- Figures must never contradict the text. If a number appears in prose, it must match the plot.

---

## 4. Report Structure — The Golden Template

Every weekly review report (`week-<N>-report.md`) follows this exact skeleton:

```markdown
---
title: <Full Academic Title>
week: <N>
month: <M>
status: complete            # planned | in-progress | complete
date: YYYY-MM-DD
prev: "[[week-<N-1>-report|Week <N-1>]]"
next: "[[week-<N+1>-report|Week <N+1>]]"
map: "[[Knowledge-Map]]"
tags:
  - ns/week-0<N>
  - ns/<topic>
  - ns/eeg
concepts:
  - "[[concept-slug-1]]"
  - "[[concept-slug-2]]"
---

# <Full Academic Title>

> **Week <N> Review Article — Month <M> / Week <N>**  
> Foundation learning phase: strong fundamentals before research.  
> Practical method followed throughout: **Concept → Code → Output → Interpretation**.

**Navigation:** ← [[week-<N-1>-report|Week <N-1>]] · [[Knowledge-Map|Knowledge Map]] · [[week-<N+1>-report|Week <N+1>]] →  
**Associated Concepts:** [[concept-slug-1]] · [[concept-slug-2]]

## Abstract                 # 4–6 sentences, states the single question the week answers
## Keywords                 # comma-separated, backticks for code/tensor tokens

## 1. Introduction          # Question answered, inheritance from prior week, and syllabus table
### Week <N> Syllabus & Learning Sequence
| Day | Core Focus | Laboratory / Practical Tasks |
| --- | --- | --- |
| [[w<N>d1\|**W<N>D1**]] | <Core Focus> | <Practical Tasks> |
| [[w<N>d2\|**W<N>D2**]] | <Core Focus> | <Practical Tasks> |
| [[w<N>d3\|**W<N>D3**]] | <Core Focus> | <Practical Tasks> |
| [[w<N>d4\|**W<N>D4**]] | <Core Focus> | <Practical Tasks> |
| [[w<N>d5\|**W<N>D5**]] | <Core Focus> | <Practical Tasks> |

## 2. Fundamental Concepts
## 3. <Topic A>             # detailed technical discussion with Concept → Code → Output → Interpretation
## 4. <Topic B>
...
## N. Comparison / Discussion
## N+1. Applications        # concrete ML/BCI paradigms with Target, Formulation, and Failure Modes
## N+2. Challenges / Limitations
## N+3. Conclusion           # numbered recap + forward-looking bridge to next week
## N+4. References           # full verified academic citations with authors, venues, years, DOIs

## Appendix A — Reproducibility (UV + Python)
## Appendix B — Knowledge Graph Connections
```

Section numbering must be **sequential and gapless** — if you delete a section, renumber everything after it and fix cross-references.

House style:
- H1 = article title. Numbered top-level sections (`## 1.` …). Numbered subsections (`### 2.1`, `### 2.2`).
- Bolded lead-ins carry the argument: `**The AI Connection:**`, `**Intuition:**`, `**The Science:**`, `**Why we care:**`, `**Crucial Distinction:**`, `**The Takeaway:**`, `**The Dangerous Misconception:**`, `**Why this is false:**`, `**Interpretation:**`, `**Output & Interpretation:**`, `**Design rule of thumb:**`.
- GitHub-style callouts (`> [!NOTE]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`, `> [!TIP]`) for critical constraints, biophysical laws, and modeling warnings.
- Comparison tables are markdown pipe tables with a **bolded first column**. Where a table lists concepts, a pitfall/limitation column is mandatory.
- Lists of named items follow a fixed field order: `**Biological Event:**` → `**Neural Activity:**` → `**Neural Data:**` → `**Target / Label:**` → `**AI Task:**` → `**Primary Failure Mode:**`.

---

## 5. Writing Style

- **Plain, direct, technically precise prose.** Short paragraphs. First sentence states the claim; the rest supports it.
- **Explain, do not just assert.** Every non-obvious claim gets a *why* — grounded in biophysics, electronics, or linear algebra.
- **Consistently surface common misconceptions and correct them**, using `**The Dangerous Misconception:** … **Why this is false:** …`. (e.g., amplitude $\ne$ computational effort, filtering $\ne$ artifact elimination, high-pass filtering eliminates DC mean).
- **Bridge biology → data → AI** with explicit transition sentences.
- **State sensor constraints.** Never let the reader overclaim what the physical measurement supports.
- **Units inline and always** (µV, µV², Hz, ms, s, voxels). Allowed symbols: `µV`, `Hz`, `×`, `→`, `≈`, `±`, `·`, en/em dashes, Greek letters (θ, α, β, γ, δ). Use standard LaTeX `$...$` and `$$...$$` for formal mathematics.
- **Shapes and identifiers in backticks**: `[Channels × Time]`, `[Samples × Features]`, `(8, 500)`, `filtfilt`.
- **No hype, no filler, no "in this report we will explore".** State the claim directly.

---

## 6. Cross-Report Linking & The Knowledge Graph (Obsidian)

The repository is built as an **Obsidian vault** (`.obsidian/` enabled). Reports are nodes; concept notes and daily files are edges.

### 6.1 Frontmatter — Required on All Reports
YAML frontmatter is mandatory at the top of `week-<N>-report.md`:

```yaml
---
title: Fundamentals of Neural Signal Processing for Artificial Intelligence
week: 2
month: 1
status: complete
date: 2026-10-02
prev: "[[week-1-report|Week 1]]"
next: "[[week-3-report|Week 3]]"
map: "[[Knowledge-Map]]"
tags:
  - ns/week-02
  - ns/signal-processing
  - ns/eeg
concepts:
  - "[[eeg-sampling-rate]]"
  - "[[nyquist-criterion]]"
  - "[[artifact-contamination]]"
  - "[[band-pass-filtering]]"
  - "[[time-domain-features]]"
  - "[[band-power-features]]"
---
```

Quote every wikilink inside YAML (`"[[...]]"`) so YAML parsers treat them as strings while Obsidian indexes them as links.

### 6.2 In-Body Navigation Banner
Immediately below the blockquote header, include navigation and concept links:

```markdown
**Navigation:** ← [[week-1-report|Week 1: Neural Data Fundamentals]] · [[Knowledge-Map|Knowledge Map]] · [[week-3-report|Week 3]] →  
**Associated Concepts:** [[eeg-sampling-rate]] · [[nyquist-criterion]] · [[artifact-contamination]]
```

### 6.3 Appendix B — Knowledge Graph Connections
Every weekly report ends with `## Appendix B — Knowledge Graph Connections`:

```markdown
## Appendix B — Knowledge Graph Connections

**Central MOC Hub:** [[Knowledge-Map]]
**Upstream:** [[week-1-report|Week 1]] — supplies the `[Channels × Time]` EEG geometry, sampling-rate intuition, and frequency-band vocabulary.
**Downstream:** [[week-3-report|Week 3]] — feature matrices `[Samples × Features]` from §6 become inputs for ML classifiers.
**Sibling concepts:** [[nyquist-criterion]] · [[artifact-contamination]] · [[band-pass-filtering]] · [[time-domain-features]] · [[band-power-features]]
```

### 6.4 Concept Notes — The Connective Tissue
Atomic concept notes live in `notes/concepts/<concept-slug>.md` (lowercase-hyphenated, no spaces). Create one when a concept is referenced across multiple weeks or is fundamental.

```markdown
---
title: Nyquist Criterion
slug: nyquist-criterion
tags: [ns/concept, ns/sampling]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-2-report|Week 2]]"]
---

# Nyquist Criterion

<50–150 word definition: the mathematical claim, the why, and practical implications.>

**Why it matters here:** <one sentence tied to a real modeling consequence.>

**Appears in:** [[week-1-report|Week 1]] · [[week-2-report|Week 2]] · [[w2d2|W2D2]]

## Related
- [[eeg-sampling-rate]]
- [[aliasing]]
```

Every concept note must carry an **"Appears in"** backlink line.

### 6.5 Daily Working Notes Convention (`days/w<N>d<D>.md`)

In addition to weekly synthesis reports, daily active working notes are maintained:

1. **Location:** `month <N>/week <N>/days/`
2. **Strict Naming Standard:** Files **must strictly be named `w<N>d<D>.md`** (e.g., `w2d1.md`, `w2d2.md`, `w2d3.md`, `w2d4.md`, `w2d5.md`). Do not use long descriptive filenames for day files; the canonical slug identifies week and day instantly.
3. **Daily Frontmatter Template:**
   ```yaml
   ---
   title: "W<N>D<D>: <Topic Title>"
   week: <N>
   day: <D>
   month: <M>
   status: complete
   date: YYYY-MM-DD
   tags:
     - ns/week-0<N>
     - ns/daily-notes
     - <topic-tags>
   prev: "[[w<N>d<D-1>|W<N>D<D-1> Notes]]"
   next: "[[w<N>d<D+1>|W<N>D<D+1> Notes]]"
   report: "[[week-<N>-report|Week <N> Report]]"
   map: "[[Knowledge-Map]]"
   concepts:
     - "[[concept-slug]]"
   ---
   ```
4. **Daily Navigation Line:**
   `**Navigation:** ← [[w<N>d<D-1>|W<N>D<D-1>]] · [[week-<N>-report|Week <N> Report]] · [[w<N>d<D+1>|W<N>D<D+1>]] →`
5. **Asset Embedding from `days/`:**
   Because day files live in `days/`, relative image links must traverse up one level:
   `![Descriptive caption](../assets/<image_name>.png)`
6. **Integration with `week-<N>-report.md`:**
   The syllabus table in Section 1 of the weekly report must hyperlink each day row using `[[w<N>d1\|**W<N>D1**]]` through `[[w<N>d5\|**W<N>D5**]]`.
7. **Content Requirements for Daily Notes:**
   - Detailed lecture / concept exposition
   - Complete runnable code implementations with printed outputs
   - Daily Knowledge Checkpoint questions with **full, authoritative, scientifically reasoned solutions**
   - Embedded diagnostic figures from `../assets/`

### 6.6 Knowledge-Map.md — The Vault Hub

The root `Knowledge-Map.md` is the graph entry point. It contains:
- The 4-week spine table
- **Dedicated Daily Notes subsections** linking every daily working note:
  ```markdown
  ### Week 2 Daily Notes
  - [[w2d1|W2D1]]: Signals, Time Series & Sampling Fundamentals (incl. Week 1 Diagnostic Verification)
  - [[w2d2|W2D2]]: The Nyquist Intuition & Artifact Contamination
  - [[w2d3|W2D3]]: Basic Filtering — The Mathematics of Signal Rescue
  - [[w2d4|W2D4]]: Feature Extraction — From Matrices to Meaning
  - [[w2d5|W2D5]]: The End-to-End Pipeline & Final Review (incl. Week 2 Mastery Test)
  ```
- Concept spine categorized by Data & Measurement, Representation, and Learning.
- Central paradigm definition.

### 6.7 Link Syntax Rules

| Link Type | Syntax | Reason |
| :--- | :--- | :--- |
| File to file (vault-wide) | `[[week-2-report\|Week 2]]` or `[[w2d1\|W2D1]]` | Folders contain spaces (`month 1/week 1/`). Obsidian resolves wikilinks by base filename globally without broken path encodings. |
| In markdown tables | `[[w2d1\|**W2D1**]]` (escaped pipe `\|`) | Unescaped pipes `|` split table columns and corrupt rendering. |
| Report to asset | `![caption](assets/figure.png)` | Direct relative path within the week folder. |
| Day note to asset | `![caption](../assets/figure.png)` | Must step up from `days/` into `assets/`. |

---

## 7. Code Blocks — Required and Honest

Every report and daily note must contain runnable code formatted as:

```
### N.M Concept → Code → Output → Interpretation
```

1. **Code:** A fenced ````python```` block runnable as written using the exact project virtual environment. Comment stages clearly with `# 1.`, `# 2.` and label biological units in comments.
2. **Output:** A fenced ````text```` block containing the *exact, unedited console output* obtained by executing the code with `uv run python`. **Never invent, estimate, or hand-wave numerical output.**
3. **Interpretation:** Technical prose explaining the exact physical, mathematical, and machine learning implications of the numbers.

---

## 8. Continuity Protocol — How Weeks Connect

1. **Open by naming the inheritance.** `## 1. Introduction` must state, in the opening sentences, which specific representation and mathematical construct from the prior week this week depends on.
2. **Close by handing something forward.** `## Conclusion` ends with a forward-looking bridge naming the next week's core question.
3. **Reuse vocabulary verbatim.** Maintain identical terms, units, and notation (`sampling_rate`, `Channels × Time`, `[Samples × Features]`, `T = 1/f`, `µV`, `Hz`).
4. **Preserve Raw Exports:** Whenever raw prompt text, curriculum briefs, or unedited course notes are imported, store them as `week<N>-export.md` (and `week-<N>-export.md`) alongside `week-<N>-report.md`. The report file itself remains the polished, academic review.

---

## 9. Academic References & Citations

All weekly reports and daily notes must include formal academic citations in `## References`:
- Format: Author(s), Year, Title, Journal / Publisher, DOI link.
- Include a 1-sentence **Contribution** summary stating the exact finding or theoretical principle relevant to the report.
- Reference seminal literature:
  - *Kandel et al. (2021)* — *Principles of Neural Science*
  - *Nunez & Srinivasan (2006)* — *Electric Fields of the Brain: The Neurophysics of EEG*
  - *Cohen (2014)* — *Analyzing Neural Time Series Data*
  - *Craik et al. (2019)* — *Deep Learning for EEG Classification Tasks*
  - *Berger (1929)* — *Über das Elektrenkephalogramm des Menschen*
  - *Buzsáki (2006)* — *Rhythms of the Brain*

---

## 10. Quality Bar Checklist Before Finishing

- [ ] Weekly report created at `month <N>/week <N>/week-<N>-report.md`.
- [ ] Raw export preserved at `month <N>/week <N>/week<N>-export.md`.
- [ ] Daily working notes created in `month <N>/week <N>/days/` and named strictly as `w<N>d1.md` through `w<N>d5.md`.
- [ ] Inter-day navigation and frontmatter valid across all daily notes.
- [ ] All figures generated deterministically via `uv run python generate_week<N>_plots.py` into `assets/*.png`.
- [ ] Image paths resolve correctly (`assets/` for reports; `../assets/` for day notes). Every image has a sequential `**Figure N.**` caption stating an interpretation.
- [ ] All code blocks executed with `uv run python` and verified against exact console output.
- [ ] Frontmatter present, valid YAML, wikilinks quoted (`"[[...]]"`).
- [ ] Navigation line below blockquote header; Appendix B present with explicit Upstream / Downstream links.
- [ ] `Knowledge-Map.md` updated with the weekly spine row and the `### Week <N> Daily Notes` list.
- [ ] Syllabus table in `week-<N>-report.md` Section 1 links to `[[w<N>d1\|**W<N>D1**]]` through `[[w<N>d5\|**W<N>D5**]]`.
- [ ] All wikilinks resolve to existing vault files without broken references (verified with a python scan).
- [ ] Academic references complete with DOIs and contribution annotations.
