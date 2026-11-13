---
title: Speech Technology Knowledge Map
tags: [speech/moc, audio/dsp]
---

# Speech Technology — Knowledge Map

This map records the confirmed scope of Weeks 1–5 from the supplied curriculum screenshots. Weeks 6–12 remain unassigned here until their lesson material is supplied. See [[source-catalog]] for source boundaries and evidence.

## Confirmed Curriculum

| Week | Notes File | Core Question | Confirmed Focus | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [[week-1-notes\|Week 1: Course Introduction, DSP Fundamentals, Speech Production and Waveforms]] | How do communication intent and vocal articulation become an acoustic waveform, and how do sampling and transform tools represent that signal digitally? | Course introduction; speech communication chain; phonemes and articulators; excitation, formants, pitch and acoustic waveform; sampling; Fourier series; CT/DT Fourier transforms; DFT; z-transform; systems and rational filters; revision | **complete** |
| **2** | [[week 2/week-2-notes\|Week 2: Speech Production, Perception, Frequency Analysis and Feature Extraction]] | How do auditory perception and frequency-domain processing motivate cepstral filtering and compact speech features? | Speech production and perception; frequency analysis; perceptual masking; cepstrum and filtering; feature extraction (Parts 01/02); Mel-frequency cepstral coefficients (MFCC); single Gaussian review including covariance and maximum-likelihood estimation | **complete** |
| **3** | [[week 3/week-3-notes\|Week 3: Gaussian Classification, Mixture Estimation and Markov Weather Prediction]] | How do Gaussian class models generalize to mixtures, and how does a Markov chain predict a fully observed weather sequence? | Gaussian binary classification; bivariate Gaussian density; Gaussian mixture models; EM parameter estimation; Markov models; weather prediction by Viterbi decoding | **confirmed** |
| **4** | [[week 4/week-4-notes\|Week 4: Hidden Markov Models and GMM Acoustic Modeling]] | How do hidden state sequences connect acoustic observations to sequence likelihood, decoding, and speech-unit models? | Viterbi algorithm; hidden Markov models; weather HMM review; forward algorithm; HMM/GMM speech acoustic modeling (Parts 01/02) | **confirmed** |
| **5** | [[week 5/week-5-notes\|Week 5 Notes]] | How do hidden-state acoustic models, pronunciation constraints, and a finite-context word-sequence model combine in ASR decoding? | HMM speech; continuous vs. discrete observations; HMM types/topologies; lexicon; language-model introduction; n-gram models | **confirmed** |
| **6–12** | Scope not yet supplied | Scope not yet supplied | No lesson mapping assigned | *unassigned* |

## Confirmed Concept Taxonomy

### Week 1 — Course Introduction, DSP and Speech Production
- [[speech-chain]] — communication from linguistic intent through speaker physiology and acoustic waveform to listener perception.
- [[nyquist-sampling-theorem]] — sampling criterion and aliasing.
- [[discrete-fourier-transform]] — discrete frequency representation; related course material includes Fourier series and CT/DT Fourier transforms.
- [[linear-time-invariant-systems]] — system properties, impulse response and convolution.
- [[rational-filters-poles-zeros]] — z-domain transfer functions, poles and zeros.

### Week 2 — Perception, Filtering and Features
- [[speech-perception-and-masking]] — frequency-dependent perception and masking.
- [[source-filter-model]] — excitation, vocal-tract resonances and radiated speech waveform.
- [[short-time-fourier-transform]] — localized frequency analysis of non-stationary speech.
- [[mel-scale-filter-banks]] — perceptually spaced spectral pooling.
- [[cepstral-analysis-mfcc]] — cepstral filtering and MFCC feature construction.
- [[gaussian-distribution]] — single multivariate Gaussian, covariance and maximum-likelihood parameter estimates.

### Week 3 — Gaussian Classification, Mixtures and Markov Prediction
- [[gaussian-binary-classification]] — likelihood-based binary classification with Gaussian class conditionals.
- [[gaussian-distribution]] — bivariate Gaussian geometry and the bridge from one density to a mixture.
- [[gaussian-mixture-models]] — weighted Gaussian components for multimodal density models.
- [[expectation-maximization]] — latent-component responsibilities and iterative GMM maximum-likelihood estimation.
- [[markov-chain]] — transition model and sequence prediction for fully observed weather states.
- [[viterbi-algorithm]] — maximum-product dynamic programming for weather prediction; the general algorithm is not restricted to hidden-state speech recognition.

### Week 4 — Hidden Markov Models and GMM Acoustic Modeling
- [[viterbi-algorithm]] — maximum-probability path decoding in the observed weather chain and, with emission factors, hidden-state sequence decoding.
- [[hidden-markov-models]] — latent state chain with initial distribution, transitions, and state-conditional observation emissions.
- [[forward-backward-algorithm]] — forward evidence summation and backward suffix probabilities; together they yield smoothed posteriors and Baum–Welch statistics.
- [[baum-welch-algorithm]] — HMM expectation–maximization using soft state and transition posteriors.
- [[gaussian-mixture-models]] — continuous state emission densities in classical GMM-HMM acoustic models.
- [[hmm-gmm-acoustic-modeling]] — phone-state topology, acoustic emissions, and sequence alignment in classical speech recognition.
- [[expectation-maximization]] — Week 3 GMM fitting prerequisite; HMM Baum–Welch applies EM with state and transition posteriors.
- [[triphone-state-tying]] — context-dependent phone states and parameter sharing to address coarticulation and sparse counts.

### Week 5 — HMM Speech Recognition, Lexicon and N-gram Language Models
- [[hidden-markov-models]] — discrete-state sequence model reused for speech, with explicit distinctions between discrete and continuous observation likelihoods.
- [[discrete-continuous-hmm]] — observation alphabet versus feature-space density; continuous observations do not imply continuous time.
- [[hmm-topologies]] — ergodic and left-to-right transition structures, with topology conventions separated from universal claims.
- [[lexicon-pronunciation-dictionary]] — word-to-phone alternatives that constrain pronunciations in recognition search.
- [[n-gram-language-model]] — finite-history approximation to the word-sequence chain rule, with count estimates and sparsity limits.
- [[asr-bayesian-decoding]] — joint decoding over acoustic evidence, pronunciation paths, and word-sequence priors.

## Supplementary Concepts — Course Week Not Assigned

These existing notes are retained as supplementary material. Their presence does not assert that these topics occur in the supplied Week 1–4 lessons.

- [[autocorrelation-pitch-estimation]]
- [[k-means-clustering]]

## Central Learning Path

```text
Course Introduction
        │
        ├── Speech communication chain
        │     └── Phonemes and articulators → excitation / formants / pitch → acoustic waveform
        │
        └── DSP fundamentals and revision
              └── Sampling · Fourier series · CT/DT Fourier transforms · DFT · z-transform
                    └── Systems and rational filters

Speech Production and Perception
        └── Frequency analysis → perceptual masking
              └── Cepstrum and filtering
                    └── Feature extraction (Parts 01/02)
                          └── MFCC
                                └── Single Gaussian review (covariance, MLE)
                                      └── Gaussian binary classification and bivariate density
                                            └── Gaussian mixture model
                                                  └── EM parameter estimation

Fully observed weather sequence
        └── Markov transition model
              └── Viterbi maximum-probability state sequence

Hidden-state acoustic sequence (Week 4)
        └── HMM: initial distribution π · transition matrix A · emission model B
              ├── Forward algorithm → total observation evidence P(O | λ)
              ├── Forward–backward → smoothed γ and transition ξ posteriors → Baum–Welch
              └── Viterbi → single joint-MAP state path (also usable for hard realignment)
                                      └── GMM state emissions → phone topology → triphones and state tying
                                            └── Week 5: speech HMM observation types and topology
                                                  └── pronunciation lexicon → n-gram word-sequence prior
                                                        └── joint acoustic + lexicon + language-model search
```

## Authentic exam questions

[[exam-question-index|Weeks 1–4 paper inventory, questions and worked answers]]

- [[week-1-exam-questions|Week 1: 18 source question occurrences]]
- [[week-2-exam-questions|Week 2: 21 source question occurrences]]
- [[week-3-exam-questions|Week 3: 38 source question occurrences]]
- [[week-4-exam-questions|Week 4: 21 source question occurrences]]

## Exam-focused learning route

[[exam-study-guide|Beginner learning route and evidence-based priorities for Weeks 1–4]]

- [[week-1-exam-focused-notes|Week 1: learn the concepts and practise exam mechanisms]]
- [[week-2-exam-focused-notes|Week 2: learn the concepts and practise exam mechanisms]]
- [[week-3-exam-focused-notes|Week 3: learn the concepts and practise exam mechanisms]]
- [[week-4-exam-focused-notes|Week 4: learn the concepts and practise exam mechanisms]]
