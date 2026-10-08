---
title: Linear Time-Invariant Systems
tags: [speech/concept, dsp/lti-systems]
---

# Linear Time-Invariant Systems

A **Linear Time-Invariant (LTI) System** is a mathematical operator $\mathcal{T}\{\cdot\}$ mapping an input discrete sequence $x[n]$ to an output sequence $y[n]$ that satisfies two rigorous mathematical axioms:

1. **Linearity (Superposition):**
   $$\mathcal{T}\{a x_1[n] + b x_2[n]\} = a \mathcal{T}\{x_1[n]\} + b \mathcal{T}\{x_2[n]\}$$
2. **Time-Invariance (Shift-Invariance):**
   $$\text{If } \mathcal{T}\{x[n]\} = y[n], \quad \text{then } \mathcal{T}\{x[n - k]\} = y[n - k] \quad \forall k \in \mathbb{Z}$$

An LTI system is completely characterized by its **impulse response** $h[n] = \mathcal{T}\{\delta[n]\}$. Any arbitrary input sequence can be decomposed into an infinite linear combination of shifted impulses $x[n] = \sum_k x[k] \delta[n-k]$. Consequently, system output is governed universally by the **discrete convolution sum**:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]$$

By the Convolution Theorem, convolution in the time domain maps to point-wise complex multiplication in the frequency domain: $Y(\omega) = X(\omega) H(\omega)$.

### Why It Matters Here
The human vocal tract is modeled as an acoustic pipe filter that behaves as an LTI system over short, quasi-stationary intervals ($20\text{--}30\ \mathrm{ms}$). Convolution enables analytical modeling of speech as the glottal volume velocity sequence convolved with the vocal tract impulse response.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[source-filter-model]]
- [[rational-filters-poles-zeros]]
- [[discrete-fourier-transform]]
