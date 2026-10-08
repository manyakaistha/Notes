# AGENTS.md — Speech Technology Track Specification

**Vault Root:** `Speech Technology/`  
**Discipline:** Speech Technology & Acoustic Engineering (IIT Madras BSEE4001 / BS Degree)  
**Parent Curriculum:** `Notes/`

Any autonomous agent or human contributor working within this vault must treat this document as the authoritative standard for the `Speech Technology/` course. 

> [!IMPORTANT]
> **Track-Specific Convention Notice:**
> The `Speech Technology` course uses a **simplified 12-week structure** that differs intentionally from other vaults (such as PINN and NeuroTech):
> 1. **NO Month Folders:** Do NOT nest weeks inside `month 1/`, `month 2/`, etc. Weeks are organized directly as `week 1/`, `week 2/`, ... `week 12/`.
> 2. **NO Daywise Notes:** Do NOT create `days/` subfolders or individual day notes (`w1d1.md`, etc.). All content for a week is consolidated into a single comprehensive weekly note.
> 3. **NO Raw Exports:** Do NOT create raw `weekN-export.md` syllabus dumping files. Everything needed is integrated directly into the weekly notes.

---

## 1. Directory Architecture & File Naming

The vault layout spans weeks 1 through 12:

```text
Speech Technology/
├── AGENTS.md                         # Authoritative track rules & conventions
├── Knowledge-Map.md                  # Central Map of Content (MOC) & 12-week spine
├── pyproject.toml                    # UV virtual environment configuration
├── uv.lock                           # Locked Python dependencies
├── generate_week1_plots.py           # Week 1 deterministic plotting script
├── notes/
│   └── concepts/                     # Atomic concept notes
│       ├── speech-chain.md
│       ├── nyquist-sampling-theorem.md
│       ├── linear-time-invariant-systems.md
│       ├── discrete-fourier-transform.md
│       ├── rational-filters-poles-zeros.md
│       ├── autocorrelation-pitch-estimation.md
│       ├── short-time-fourier-transform.md
│       ├── source-filter-model.md
│       ├── mel-scale-filter-banks.md
│       └── cepstral-analysis-mfcc.md
├── week 1/
│   ├── week-1-notes.md               # Unified, comprehensive Week 1 master note
│   └── assets/                       # High-resolution publication plots for Week 1
│       ├── 01_sampling_aliasing.png
│       ├── 02_fourier_series_decomposition.png
│       ├── 03_lti_convolution_filtering.png
│       ├── 04_pole_zero_constellation.png
│       ├── 05_autocorrelation_pitch.png
│       ├── 06_wideband_vs_narrowband_spectrogram.png
│       ├── 07_source_filter_cepstral_separation.png
│       └── 08_mel_filterbank_frequency_response.png
├── week 2/                           # (Planned: LPC & Parametric Modeling)
│   ├── week-2-notes.md
│   └── assets/
└── ... up to week 12/
```

### Canonical File Naming Rules
- **Weekly Directory:** `week N/` (with a literal space, lowercase, e.g., `week 1/`, `week 2/`).
- **Weekly Note:** `week N/week-N-notes.md` (e.g., `week 1/week-1-notes.md`).
- **Assets Directory:** `week N/assets/`.
- **Image Embedding:** In `week-N-notes.md`, image links point directly to `![Caption](assets/<figure_name>.png)`. Every figure must have a bold caption paragraph immediately following it: `**Figure N.** *Description...*`.

---

## 2. Weekly Note Structural Skeleton

Each `week-N-notes.md` is a standalone master technical document organized into clear, rigorous sections:

1. **Frontmatter:**
   ```yaml
   ---
   title: "Week N: <Descriptive Title>"
   week: N
   status: complete
   date: YYYY-MM-DD
   tags: [speech/week-0N, ...]
   concepts: ["[[concept-1]]", "[[concept-2]]"]
   prev: "[[week-(N-1)-notes|Week N-1 Notes]]" # or "[[Knowledge-Map|Knowledge Map]]" for week 1
   next: "[[week-(N+1)-notes|Week N+1 Notes]]"
   map: "[[Knowledge-Map]]"
   ---
   ```
2. **Abstract & Overview:** Dense technical summary of theoretical and algorithmic scope.
3. **Theoretical Foundations:** Physiological and physical acoustics, continuous formulations, and governing theorems.
4. **Algorithmic Machinery & Derivations:** Mathematical derivations with step-by-step proofs.
5. **Visual Analysis & Figures:** High-resolution figures embedded with clear, bold numbered captions.
6. **Concept → Code → Output → Interpretation:** Self-contained, runnable Python code blocks accompanied by verbatim terminal outputs.
7. **Comparative Benchmarks & Standards:** Industry standards (telephony, codecs, ASR acoustic feature sizes).
8. **Worked Practice Problems & Exam Solutions:** Authentic problem sets with complete numerical derivations.
9. **Pathologies & Traps:** Real-world failure modes, edge cases, and numerical instability mechanisms.
10. **References & Master Formula Sheet:** Formal academic citations and symbol reference tables.

---

## 3. Core Scientific Conventions for Speech Processing

1. **Sampling Criterion:** Always enforce $f_s \ge 2 f_{\max}$. Explicitly distinguish between:
   - **Nyquist Rate:** The minimum continuous sampling rate required to avoid aliasing ($2 f_{\max}$).
   - **Nyquist Frequency:** The highest unaliased discrete frequency resolved by sampling rate $f_s$ ($f_s / 2$).
2. **Quantization & Dynamic Range:** Apply the standard uniform quantization formula $\mathrm{SQNR} \approx 6.02 B + 1.76\ \mathrm{dB}$.
3. **Convolution vs. Correlation:**
   - **Discrete Convolution:** Time-reversal and shift: $y[n] = x[n] * h[n] = \sum_k x[k] h[n - k]$.
   - **Cross-Correlation:** Pure shift without reflection: $r_{xy}[l] = \sum_n x[n] y[n - l]$.
   - **Autocorrelation:** Even symmetry $r_{xx}[l] = r_{xx}[-l]$, maximum at zero lag $r_{xx}[0] = E_x$, and preservation of fundamental periodicity $N_0$.
4. **The Source-Filter Paradigm:** Speech spectrum $S(\omega) = E(\omega) V(\omega) R(\omega)$.
   - Excitation $E(\omega)$: Rapidly varying pitch comb / glottal pulses.
   - Filter $V(\omega)$: Slowly varying vocal tract formant envelope.
   - Radiation $R(\omega)$: $+6\ \mathrm{dB/octave}$ lip differentiator.
   - Homomorphic Deconvolution: $\log |S(\omega)| = \log |E(\omega)| + \log |V(\omega)|$.
5. **Time-Frequency Uncertainty:** Governed by $\Delta t \cdot \Delta f \ge 1 / (4\pi)$. Short windows ($3\text{--}10\ \mathrm{ms}$) yield Wideband spectrograms (vertical pitch striations); long windows ($40\text{--}100\ \mathrm{ms}$) yield Narrowband spectrograms (horizontal harmonic bands).

---

## 4. Computational Verification Standard

- All code must run deterministically via `uv run python`.
- Signal processing operations should prefer pure NumPy and Matplotlib/Seaborn to ensure robust cross-platform portability.
- Code blocks must adhere to the 4-stage pipeline: **Concept → Code → Output → Interpretation**.
