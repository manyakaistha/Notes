---
title: Cepstral Analysis and Mel-Frequency Cepstral Coefficients
tags: [speech/concept, audio/mfcc, cepstrum]
---

# Cepstral Analysis and Mel-Frequency Cepstral Coefficients

**Formal definition.** Mel-frequency cepstral coefficients (MFCCs) are a compact representation derived by applying a cosine transform to log energies pooled over perceptually spaced frequency bands. Given frame spectrum $X_t[k]$, Mel filters $H_m[k]$ produce $E_t[m]=\sum_k |X_t[k]|^2H_m[k]$; flooring may be applied before taking $L_t[m]=\log(E_t[m]+\epsilon)$. A DCT-II maps this vector to coefficients $c_t[r]=\sum_m L_t[m]\cos[\pi r(m+1/2)/M]$, with normalization depending on convention. Retaining low-order coefficients smooths spectral-envelope detail; optional energy, temporal delta, and delta-delta features add complementary information. MFCCs are not intrinsically pitch-free: low-order cepstra emphasize the envelope, but residual excitation, windowing, and truncation can preserve pitch cues. The DCT approximately decorrelates common speech data; it does not guarantee statistical independence or exact decorrelation for arbitrary distributions.

**Why it matters here.** Week 2 places cepstral analysis and MFCCs within frequency analysis, perceptual filtering, and speech feature extraction. The transform produces a compact feature vector from perceptually pooled log-spectrum values.

**Appears in:** [[week-2-notes|Week 2 Notes]]

## Related
- [[mel-scale-filter-banks]]
- [[short-time-fourier-transform]]
- [[speech-perception-and-masking]]
- [[gaussian-distribution]]
