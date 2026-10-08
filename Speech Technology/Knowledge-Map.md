---
title: Speech Technology Knowledge Map
tags: [speech/moc, audio/dsp]
---

# Speech Technology — Knowledge Map & Curriculum Spine

Welcome to the central Map of Content (MOC) and navigational hub for the IIT Madras BS Degree course on **Speech Technology (BSEE4001)**.

---

## 12-Week Curriculum Spine

| Week | Notes File | Core Question | Focus Areas | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [[week-1-notes\|Week 1: Fundamentals of Speech Signals, Sampling, LTI Systems & Spectral Analysis]] | How do we bridge continuous acoustic pressure waves into discrete mathematical sequences for spectral and cepstral extraction? | Speech Chain, Nyquist-Shannon Sampling, Aliasing, Quantization, LTI Systems, Discrete Convolution, Rational Filters (Poles & Zeros), DFT, Autocorrelation Pitch Tracking, STFT, Spectrograms, Source-Filter Model, Mel Filter Banks | **complete** |
| **2** | [[week 2/week-2-notes\|Week 2: Statistical Modeling & Pattern Recognition in Speech]] | How do continuous acoustic feature vectors interface with statistical models to classify phones and decode word sequences? | MFCC 6-Stage Pipeline, DCT Decorrelation, Dynamic Deltas, K-Means Clustering, LBG Splitting, Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), Tri-states, Context-Dependent Tri-phones, Viterbi Decoding, Forward-Backward, Lexicon | **complete** |
| **3** | [[week-3-notes\|Week 3: Linear Predictive Coding (LPC) & Parametric Speech Modeling]] | How can vocal tract acoustics be modeled compactly via autoregressive all-pole filters and lattice structures? | Linear Prediction, Levinson-Durbin Recursion, Yule-Walker Equations, Reflection Coefficients, PARCOR, Prediction Error Filter | *planned* |
| **4** | [[week-4-notes\|Week 4: Pitch Tracking & Formant Estimation Algorithms]] | How do we reliably estimate vocal cord vibration rate and formant frequencies in real-time? | Time-domain (Autocorrelation, AMDF), Frequency-domain (Harmonic Product Spectrum), Cepstral Pitch Determination, Formant Tracking | *planned* |
| **5** | [[week-5-notes\|Week 5: Statistical Pattern Recognition & Vector Quantization]] | How do we compress and cluster continuous acoustic vectors into discrete codebooks? | Vector Quantization (LBG Algorithm), Distortion Measures (Itakura-Saito, Euclidean), Codebook Design | *planned* |
| **6** | [[week-6-notes\|Week 6: Dynamic Time Warping (DTW) & Template Matching]] | How do we align speech utterances with variable speaking rates and non-linear temporal durations? | Dynamic Programming, Local Distance Metrics, Global Path Constraints, Isolated Word Recognition | *planned* |
| **7** | [[week-7-notes\|Week 7: Hidden Markov Models (HMM) — Advanced Formulations]] | How do continuous observation densities and topology choices influence large-vocabulary acoustic modeling? | Continuous vs. Discrete HMMs, Topology Selection, Duration Modeling, Tied Mixtures | *planned* |
| **8** | [[week-8-notes\|Week 8: HMM Decoding & Parameter Estimation (Baum-Welch)]] | How do we optimally decode hidden states and train HMM parameters from unlabeled speech? | Viterbi Algorithm, Optimal State Sequences, Baum-Welch (EM) Re-estimation, Left-to-Right Phonetic Topologies | *planned* |
| **9** | [[week-9-notes\|Week 9: Acoustic Modeling & State Clustering (Senones)]] | How do we handle combinatorial explosion in context-dependent phone modeling? | Decision Tree State Tying, Senone Clustering, Minimum Classification Error, Discriminative Training | *planned* |
| **10** | [[week-10-notes\|Week 10: Statistical Language Modeling & ASR Search Decoding]] | How do we integrate acoustic evidence with lexical syntax and language priors to decode continuous speech? | N-gram Language Models, Perplexity, Smoothing & Backoff (Kneser-Ney), Weighted Finite-State Transducers (WFST), Beam Search | *planned* |
| **11** | [[week-11-notes\|Week 11: Deep Learning for Speech Recognition (Hybrid & End-to-End)]] | How do deep neural networks replace GMMs and integrate acoustic-lexical decoding into unified end-to-end models? | DNN-HMM Hybrids, Connectionist Temporal Classification (CTC), RNN-Transducer (RNN-T), Attention & Transformers (Conformer, Whisper) | *planned* |
| **12** | [[week-12-notes\|Week 12: Speech Synthesis (TTS) & Emerging Speech Technologies]] | How do we synthesize natural speech from text, and how do modern foundation audio models operate? | Formant Synthesis, Concatenative Synthesis, Neural Vocoders (WaveNet, HiFi-GAN), Voice Conversion, Speaker Identification | *planned* |

---

## Concept Taxonomy & Knowledge Graph Nodes

The atomic concept notes form the connective tissue across acoustic physics, digital signal processing, and machine learning:

### 1. Acoustic Foundations & Articulation
- [[speech-chain]]: The 5-stage linguistic-physiological-acoustic communication pipeline.
- [[source-filter-model]]: Mathematical factorization of speech into glottal excitation $E(\omega)$ and vocal tract resonance $V(\omega)$.

### 2. Discretization & Information Capacity
- [[nyquist-sampling-theorem]]: The bandlimited sampling threshold ($f_s \ge 2 f_{\max}$) and aliasing folding dynamics.

### 3. Linear Systems & Filter Topologies
- [[linear-time-invariant-systems]]: Superposition, shift-invariance, causality, and stability criteria in discrete speech systems.
- [[rational-filters-poles-zeros]]: Difference equations, transfer functions $H(z)$, and complex Z-plane stability regions.

### 4. Transform Analysis & Time-Frequency Representations
- [[discrete-fourier-transform]]: Finite discrete basis projections and algorithmic FFT synthesis.
- [[short-time-fourier-transform]]: Quasi-stationary windowing (25 ms frame / 10 ms shift) and wideband vs. narrowband spectrogram trade-offs.

### 5. Acoustic Feature Extraction & Perception
- [[autocorrelation-pitch-estimation]]: Even-symmetric lag analysis for voiced speech fundamental frequency ($F_0$) extraction.
- [[mel-scale-filter-banks]]: Human cochlear basilar membrane frequency compression and triangular filter weighting.
- [[cepstral-analysis-mfcc]]: Homomorphic log-spectral deconvolution, quefrency liftering, and Mel-Frequency Cepstral Coefficients (MFCCs).

### 6. Statistical Pattern Recognition & Sequential Modeling
- [[k-means-clustering]]: Deterministic hard vector quantization, WCSS minimization, and LBG centroid perturbation splitting.
- [[gaussian-mixture-models]]: Continuous multimodal probability density estimation, diagonal covariance simplification, and EM soft responsibilities.
- [[hidden-markov-models]]: Doubly stochastic sequential modeling, Bakis left-to-right acoustic topology, 3-state phoneme models, and context-dependent triphones.
- [[viterbi-algorithm]]: Dynamic programming trellis search for maximum likelihood state sequence decoding.
- [[lexicon-pronunciation-dictionary]]: Architectural mapping from sub-word phonetic emission sequences to orthographic lexical words.

---

## Central Governing Paradigm: Speech Processing Pipeline

```
           Acoustic Continuous Pressure Wave x(t)
                            │
                            ▼
            Anti-Aliasing Low-Pass Filter (f_c ≤ f_s / 2)
                            │
                            ▼
        Analog-to-Digital Conversion (Sampling @ f_s, 16-bit Quantization)
                            │
                            ▼
               Discrete Speech Sequence x[n]
                            │
            ┌───────────────┴───────────────────────────────┐
            ▼                                               ▼
    Time-Domain Analysis                       Short-Time Windowing (25 ms Frame, 10 ms Shift)
   [Autocorrelation r_xx[l]]                                │
            │                                               ▼
            ▼                                  Discrete Fourier Transform (DFT/FFT)
   Pitch (F_0) Estimation                                   │
                                                            ▼
                                             Speech Power Spectrum |S(ω)|²
                                                            │
                                                            ▼
                                               Triangular Mel Filter Bank
                                                            │
                                                            ▼
                                               Logarithmic Compression: log |S(ω)|
                                              [log S = log E + log V (Additive)]
                                                            │
                            ┌───────────────────────────────┴───────────────────────────────┐
                            ▼                                                               ▼
             Log-Mel Spectrogram Features                                    Discrete Cosine Transform (DCT)
             (Modern Deep Learning Acoustic Input)                                          │
                                                                                            ▼
                                                                                13-dimensional MFCCs
                                                                                + Deltas & Delta-Deltas (39D)
                                                                                            │
                                                                                            ▼
                                                                               Statistical Pattern Modeling
                                                                             Gaussian Mixture Models (GMMs)
                                                                           b_j(o_t) = P(Observation | State)
                                                                                            │
                                                                                            ▼
                                                                               Temporal Sequence Modeling
                                                                              Hidden Markov Models (HMMs)
                                                                             (Bakis 3-State Left-to-Right)
                                                                                            │
                                                                                            ▼
                                                                               Dynamic Programming Decoder
                                                                                    Viterbi Algorithm
                                                                                            │
                                                                                            ▼
                                                                               Sub-word Phoneme Sequence
                                                                                   /k/ - /ae/ - /t/
                                                                                            │
                                                                                            ▼
                                                                               Pronunciation Dictionary
                                                                                       (Lexicon)
                                                                                            │
                                                                                            ▼
                                                                                 Recognized Word Sequence
                                                                                          "CAT"
```
