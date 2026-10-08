---
title: Nyquist-Shannon Sampling Theorem
tags: [speech/concept, dsp/sampling, information-theory]
---

# Nyquist-Shannon Sampling Theorem

The **Nyquist-Shannon Sampling Theorem** establishes the fundamental mathematical condition under which a continuous-time, bandlimited analog signal $x(t)$ with maximum frequency component $f_{\max}$ can be completely represented by and reconstructed from a discrete-time sequence of samples $x[n] = x(n T_s)$. 

The theoretical criterion dictates that the sampling frequency $f_s = 1/T_s$ must be strictly greater than or equal to twice the highest frequency present in the continuous signal:

$$f_s \ge 2 f_{\max}$$

The threshold $2 f_{\max}$ is defined as the **Nyquist Rate**. Conversely, half the sampling frequency, $f_{\mathrm{Nyquist}} = f_s / 2$, is the **Nyquist Folding Frequency**. If $f_s < 2 f_{\max}$, spectral copies spaced at integer multiples of $f_s$ overlap in the frequency domain, causing irreversible **aliasing distortion**, where high-frequency spectral components map identically onto lower frequencies.

### Why It Matters Here
Human vocal articulation generates perceptible acoustic energy up to approximately $7.5\ \mathrm{kHz}$. Consequently, modern telecommunication and speech recognition pipelines enforce a wideband sampling standard of $16\ \mathrm{kHz}$ ($f_s \ge 2 \times 7.5 = 15\ \mathrm{kHz}$). Legacy telephone channels pass bandlimited audio between $300\ \mathrm{Hz}$ and $3.4\ \mathrm{kHz}$, necessitating the standard narrowband sampling rate of $8\ \mathrm{kHz}$.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[discrete-fourier-transform]]
- [[linear-time-invariant-systems]]
- [[short-time-fourier-transform]]
