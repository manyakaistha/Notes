---
title: Rational Digital Filters, Poles & Zeros
tags: [speech/concept, dsp/filters, z-transform]
---

# Rational Digital Filters, Poles & Zeros

A **Rational Digital Filter** is an LTI discrete-time system governed by a linear constant-coefficient difference equation relating input $x[n]$ and output $y[n]$:

$$y[n] - \sum_{k=1}^N a_k y[n-k] = \sum_{r=0}^M b_r x[n-r]$$

Applying the bilateral Z-transform yields the rational system transfer function $H(z)$:

$$H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{r=0}^M b_r z^{-r}}{1 - \sum_{k=1}^N a_k z^{-k}} = K \frac{\prod_{i=1}^M (z - z_i)}{\prod_{k=1}^N (z - p_k)}$$

- **Zeros ($z_i$):** Roots of the numerator polynomial where $H(z) = 0$, suppressing frequency transmission.
- **Poles ($p_k$):** Roots of the denominator polynomial where $H(z) \to \infty$, causing sharp spectral amplification (resonances).

### Stability and Causality
1. **Causality:** The Region of Convergence (ROC) must lie strictly outside the radius of the outermost pole: $\mathrm{ROC} = \{z : |z| > \max_k |p_k|\}$.
2. **BIBO Stability:** The ROC must encompass the unit circle $|z| = 1$.
3. **Causal & Stable:** All system poles must reside strictly **inside the unit circle** ($|p_k| < 1$).

### Why It Matters Here
The human vocal tract functions physically as an acoustic tube resonator. When modeled as an all-pole filter ($M=0$, $b_r = 0$ for $r \ge 1$), the complex pole pairs near the unit circle directly correspond to vocal tract **formants** (resonant peaks $F_1, F_2, F_3$). This rational representation forms the mathematical bedrock of Linear Predictive Coding (LPC) and CELP speech compression.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[linear-time-invariant-systems]]
- [[source-filter-model]]
- [[cepstral-analysis-mfcc]]
