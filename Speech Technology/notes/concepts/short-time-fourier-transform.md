---
title: Short-Time Fourier Transform & Spectrograms
tags: [speech/concept, dsp/time-frequency, spectrograms]
---

# Short-Time Fourier Transform & Spectrograms

Because speech is intrinsically non-stationary—the vocal tract articulators continuously reshape resonant cavities over time—a single Fourier transform over a full utterance averages disparate phonemic spectra and destroys temporal sequencing. The **Short-Time Fourier Transform (STFT)** resolves this by assuming speech is **quasi-stationary** over short intervals ($20\text{--}30\ \mathrm{ms}$):

$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m R] e^{-j \omega n}$$

where $w[n]$ is a tapered analysis window (e.g., Hamming window) of duration $L$ samples, $R$ is the frame hop size (shift), and $m$ indexes the analysis frame. The 2D energy distribution $|X(m, \omega)|^2$ constitutes the **Spectrogram**.

### The Time-Frequency Uncertainty Trade-Off
Governed by the Gabor limit ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$):
- **Wideband Spectrograms ($L \approx 5\text{--}10\ \mathrm{ms}$):** Superior time resolution ($\Delta t$ small) at the cost of smeared frequency resolution ($\Delta f$ large). Resolves individual glottal pulses as vertical striations.
- **Narrowband Spectrograms ($L \approx 40\text{--}100\ \mathrm{ms}$):** Superior frequency resolution ($\Delta f$ small) at the cost of smeared temporal events ($\Delta t$ large). Resolves individual harmonic lines ($k \cdot F_0$) as horizontal bands.

Standard ASR feature pipelines utilize a balanced $25\ \mathrm{ms}$ window with a $10\ \mathrm{ms}$ frame advance.

**Appears in:** [[week-1-notes|Week 1 Notes]]

## Related
- [[discrete-fourier-transform]]
- [[source-filter-model]]
- [[mel-scale-filter-banks]]
