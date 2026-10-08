---
title: Mel Scale and Filter Bank Architectures
tags: [speech/concept, perception/psychoacoustics, features]
---

# Mel Scale and Filter Bank Architectures

The **Mel Scale** is an empirical psychoacoustic frequency scale relating physical frequency $f_{\mathrm{Hz}}$ to perceived pitch $f_{\mathrm{mel}}$, introduced by Stevens, Volkmann, and Newman (1937) and formulated analytically as:

$$f_{\mathrm{mel}} = 1125 \ln\left(1 + \frac{f_{\mathrm{Hz}}}{700}\right)$$

### Biological Origin & Nonlinearity
The human cochlea's basilar membrane does not distribute frequency sensitivity linearly. Tonotopic hair cell density is high at low frequencies and declines logarithmically above $1000\ \mathrm{Hz}$. Hence, human listeners easily resolve a $50\ \mathrm{Hz}$ difference between $200\ \mathrm{Hz}$ and $250\ \mathrm{Hz}$, but cannot distinguish between $4000\ \mathrm{Hz}$ and $4050\ \mathrm{Hz}$.

### Triangular Mel Filter Bank Construction
To emulate this non-uniform biological resolution, acoustic spectral energies $|X[k]|^2$ are integrated across a bank of $M$ overlapping triangular bandpass filters:
- **Low-Frequency Band (< 1 kHz):** Linearly spaced with narrow bandwidths ($~100\ \mathrm{Hz}$), providing fine formant frequency resolution.
- **High-Frequency Band (> 1 kHz):** Logarithmically spaced with exponentially increasing bandwidths, providing broad energy pooling.

Standard configurations specify **80 triangular filters** for wideband $16\ \mathrm{kHz}$ audio and **40 triangular filters** for narrowband $8\ \mathrm{kHz}$ telephony audio.

### Why It Matters Here
Mel filter banks perform critical dimension reduction (e.g., condensing 513 DFT bins down to 80 channels) and match the spectral representation directly to human perceptual sensitivity, serving as the dominant front-end for modern Conformer, Whisper, and Wav2Vec2 architectures.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[short-time-fourier-transform]]
- [[cepstral-analysis-mfcc]]
- [[source-filter-model]]
