---
title: Cepstral Analysis and Mel-Frequency Cepstral Coefficients
tags: [speech/concept, audio/mfcc, cepstrum]
---

# Cepstral Analysis and Mel-Frequency Cepstral Coefficients

**Cepstral Analysis** (coined as an anagram of *spectral*) is a homomorphic signal processing technique designed to separate signals combined through convolution. In speech processing, the acoustic spectrum is multiplicative: $S(\omega) = E(\omega) V(\omega)$. Taking the complex or real logarithm converts this into an additive formulation:

$$\log |S(\omega)| = \log |E(\omega)| + \log |V(\omega)|$$

Because $\log |V(\omega)|$ (vocal tract formants) varies slowly across frequency (low "quefrency") and $\log |E(\omega)|$ (glottal harmonics) varies rapidly (high "quefrency"), applying an Inverse DFT (or Discrete Cosine Transform) maps these components to distinct, separable regions along the **quefrency** axis (pseudo-time).

### MFCC Extraction Pipeline
1. **Pre-emphasis:** First-order high-pass filter $y[n] = x[n] - \alpha x[n-1]$ ($\alpha \approx 0.97$) flattening the $-6\ \mathrm{dB/octave}$ spectral tilt.
2. **Frame Windowing:** 25 ms analysis window shifted by 10 ms with Hamming tapering.
3. **Short-Time DFT:** Computing discrete magnitude spectrum $|X[k]|$.
4. **Mel Filter Bank:** Integrating $|X[k]|^2$ into 80 triangular Mel channels.
5. **Logarithmic Energy Compression:** $S_m = \log \sum_k |X[k]|^2 H_m[k]$, linearizing source and filter.
6. **Discrete Cosine Transform (DCT-II):**
   $$c_n = \sum_{m=1}^{M} S_m \cos\left(\frac{\pi n (m - 0.5)}{M}\right)$$
   Yields 13 decorrelated static MFCC coefficients, typically augmented with velocity ($\Delta$) and acceleration ($\Delta\Delta$) deltas.

### Why It Matters Here
Before deep self-attention encoders, MFCCs combined with Gaussian Mixture Models (GMM-HMMs) formed the universal acoustic foundation for all commercial speech recognizers. Today, Log-Mel filter bank energies remain the standard input to deep neural networks.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[source-filter-model]]
- [[mel-scale-filter-banks]]
- [[discrete-fourier-transform]]
