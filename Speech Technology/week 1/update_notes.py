import re

with open('week-1-notes.md', 'r') as f:
    text = f.read()

# 1. Equation Replacement
text = text.replace(r'\xLongleftrightarrow{\quad\mathcal{F}\quad}', r'\stackrel{\mathcal{F}}{\longleftrightarrow}')

# 2. Figure name fixes:
# Replace 02_fourier_series_decomposition.png with 03_ (and Figure 2 with Figure 3)
text = text.replace('![Fourier Decomposition](assets/02_fourier_series_decomposition.png)\n\n**Figure 2.**', '![Fourier Decomposition](assets/03_fourier_series_decomposition.png)\n\n**Figure 3.**')

# Replace 03_lti_convolution_filtering.png with 05_ (and Figure 3 with Figure 5)
text = text.replace('![LTI Convolution Filtering](assets/03_lti_convolution_filtering.png)\n\n**Figure 3.**', '![LTI Convolution Filtering](assets/05_lti_convolution_filtering.png)\n\n**Figure 5.**')

# Replace 04_pole_zero_constellation.png with 06_ (and Figure 4 with Figure 6)
text = text.replace('![Pole Zero Constellation](assets/04_pole_zero_constellation.png)\n\n**Figure 4.**', '![Pole Zero Constellation](assets/06_pole_zero_constellation.png)\n\n**Figure 6.**')

# Replace 05_autocorrelation_pitch.png with 07_ (and Figure 5 with Figure 7)
text = text.replace('![Autocorrelation Pitch Estimation](assets/05_autocorrelation_pitch.png)\n\n**Figure 5.**', '![Autocorrelation Pitch Estimation](assets/07_autocorrelation_pitch.png)\n\n**Figure 7.**')

# Replace 06_wideband_vs_narrowband_spectrogram.png with 08_ (and Figure 6 with Figure 8)
text = text.replace('![Spectrogram Trade-Off](assets/06_wideband_vs_narrowband_spectrogram.png)\n\n**Figure 6.**', '![Spectrogram Trade-Off](assets/08_wideband_vs_narrowband_spectrogram.png)\n\n**Figure 8.**')

# Replace 07_source_filter_cepstral_separation.png with 09_ (and Figure 7 with Figure 9)
text = text.replace('![Source Filter Cepstral Separation](assets/07_source_filter_cepstral_separation.png)\n\n**Figure 7.**', '![Source Filter Cepstral Separation](assets/09_source_filter_cepstral_separation.png)\n\n**Figure 9.**')

# Replace 08_mel_filterbank_frequency_response.png with 10_ (and Figure 8 with Figure 10)
text = text.replace('![Mel Filter Bank](assets/08_mel_filterbank_frequency_response.png)\n\n**Figure 8.**', '![Mel Filter Bank](assets/10_mel_filterbank_frequency_response.png)\n\n**Figure 10.**')

# 3. Add Missing Figures:
# a) Quantization (02)
quantization_text = """Every added bit of quantization precision increases dynamic range by **$6.02\\ \\mathrm{dB}$**. A 16-bit linear PCM system provides $\\approx 98\\ \\mathrm{dB}$ of dynamic range, sufficient to encompass the full dynamic range of human hearing without audible quantization hiss.

![Quantization SQNR](assets/02_quantization_sqnr.png)

**Figure 2.** *Uniform quantization mapping and the theoretical 6.02 dB per bit SQNR limit against measured error on a full-scale sinusoid.*"""

text = text.replace('Every added bit of quantization precision increases dynamic range by **$6.02\\ \\mathrm{dB}$**. A 16-bit linear PCM system provides $\\approx 98\\ \\mathrm{dB}$ of dynamic range, sufficient to encompass the full dynamic range of human hearing without audible quantization hiss.', quantization_text)

# b) DFT leakage (04)
dft_leakage_text = """All unique spectral information resides within bins $k = 0, 1, \\dots, N/2$, spanning frequencies from $0\\ \\mathrm{Hz}$ to the Nyquist folding frequency $f_s / 2$.

![DFT Windowing and Leakage](assets/04_dft_window_leakage.png)

**Figure 4.** *Spectral leakage and window function smoothing. Top: Rectangular vs. Hamming window on bin-centered and off-bin frequencies. Bottom: Sidelobe decay profiles (dB) for Rectangular, Hann, and Hamming windows.*"""

text = text.replace('All unique spectral information resides within bins $k = 0, 1, \\dots, N/2$, spanning frequencies from $0\\ \\mathrm{Hz}$ to the Nyquist folding frequency $f_s / 2$.', dft_leakage_text)


with open('week-1-notes_updated.md', 'w') as f:
    f.write(text)

