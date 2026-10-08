---
title: Discrete Fourier Transform
tags: [speech/concept, dsp/fourier, algorithms]
---

# Discrete Fourier Transform

The **Discrete Fourier Transform (DFT)** is the foundational mathematical operator that transforms a finite, discrete-time sequence of $N$ complex or real numbers $x[n]$ into an equivalent finite sequence of $N$ discrete frequency bins $X[k]$:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$

The synthesis equation (Inverse Discrete Fourier Transform, IDFT) perfectly reconstructs the time sequence:

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j \frac{2\pi}{N} k n}, \quad n = 0, 1, \dots, N-1$$

Each complex coefficient $X[k]$ encodes the magnitude $|X[k]|$ and phase $\angle X[k]$ of the projection of $x[n]$ onto the orthogonal basis sinusoid of discrete angular frequency $\omega_k = \frac{2\pi k}{N}$ radians/sample (corresponding to physical frequency $f_k = \frac{k f_s}{N}\ \mathrm{Hz}$). While naive evaluation requires $\mathcal{O}(N^2)$ complex operations, the **Fast Fourier Transform (FFT)** algorithmic family evaluates the exact DFT in $\mathcal{O}(N \log N)$ complexity.

### Why It Matters Here
Computers cannot directly compute continuous frequency spectra (such as the CTFT or DTFT). The DFT provides the discrete mathematical engine for extracting spectral envelopes, identifying vocal tract formant resonances, and generating time-frequency spectrograms.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[short-time-fourier-transform]]
- [[mel-scale-filter-banks]]
- [[linear-time-invariant-systems]]
