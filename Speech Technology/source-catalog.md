---
title: "Speech Technology: Curriculum Scope and Source Catalog"
status: complete
date: 2026-11-10
tags: [speech/sources, speech/curriculum]
map: "[[Knowledge-Map]]"
---

# Curriculum Scope and Source Catalog

**Scope authority:** The course-page screenshots supplied on 10 November 2026 determine the contents of [[week-1-notes|Week 1]], [[week-2-notes|Week 2]], [[week 3/week-3-notes|Week 3]], [[week 4/week-4-notes|Week 4]], and [[week 5/week-5-notes|Week 5]]. Weeks 6–12 remain unassigned. The slide decks support explanations within those boundaries; their inclusion as attachments does not assign every deck topic to a week. Page numbers below refer to physical PDF pages, counted from 1, because some printed slide numbers differ.

## Confirmed Lecture Lists

| Week | Lectures, in screenshot order | Evidence |
| :--- | :--- | :--- |
| 1 | Course Introduction; Digital Signal Processing Fundamentals; Fourier Series; Speech Production; Revision; Waveforms | [Week 1 course screenshot](sources/week-1-curriculum.png) |
| 2 | Speech Production, Perception and Frequency Analysis; Perceptual Masking, Cepstrum and Filtering; Feature Extraction for Speech Processing - Part 01; Feature Extraction for Speech Processing - Part 02; MFCC; Gaussian Review | [Week 2 course screenshot](sources/week-2-curriculum.png) |
| 3 | Gaussian Binary Classification; Bivariate Gaussian; Gaussian Mixture Models (GMM); GMM Parameter Estimation; Markov Model; Predicting Weather | [Week 3 course screenshot](sources/week-3-curriculum.png) |
| 4 | Viterbi; Hidden Markov Model; Review HMM (Weather/Grass); Forward Algorithm (HMM); Introduction to HMM/GMM for Speech (Parts 01/02) | [Week 4 course screenshot](sources/week-4-curriculum.png) |
| 5 | HMM Speech; Continuous vs. Discrete; Different Types of HMM; Lexicon; Introduction to Language Models; N-gram Models | [Week 5 course screenshot](sources/week-5-curriculum.png) |

The screenshots also list activity/practice/graded assignments. Their labels establish placement, but do not supply the assignment questions, solutions, grading scheme, or video transcripts. Notes therefore distinguish slide-supported explanations and generated practice from actual assignment content.

## Retained Sources and Relevant Ranges

| Source | Relevant range | Use and boundary |
| :--- | :--- | :--- |
| S. Umesh, *Introduction to Speech Signal Processing*, [Speech_course_slides-revamp.pdf](sources/Speech_course_slides-revamp.pdf) | PDF pp. 1–43 | Week 1 introduction, DSP recap, systems and transforms, speech production, phonemes, voiced/unvoiced speech, and waveforms. The source-filter overview is introduced only as production context. |
| Same deck | PDF pp. 44–72 | Week 2 production/frequency analysis, STFT, spectrograms, perception, masking, cepstrum/liftering, filter banks, and MFCCs. The Mel-scale slide on PDF p. 58 uses $1127\ln(1+f/700)$. |
| S. Umesh, *Gaussian Mixture Model*, [GMM_HMM_mar2026.pdf](sources/GMM_HMM_mar2026.pdf) | PDF pp. 2–36 and 47–140 | Week 2 single-Gaussian review uses pp. 2–22; Week 3 uses pp. 10–36 and 47–75 for classification, bivariate Gaussian, GMM, EM, Markov model, and weather prediction/Viterbi. Week 4 reviews Viterbi on pp. 61–75, covers HMM/forward inference on pp. 76–105, and speech HMM/GMM modeling on pp. 106–140. Week 5 uses overlapping speech-HMM material on pp. 127 and 131 for topology and discrete-vs-continuous observations. At p. 131 the lecture informally defines ergodic HMM by all $a_{ij}>0$; formal finite-chain ergodicity does not require every one-step matrix entry to be positive. PDF pp. 37–46 are optional GMM-initialization background, not a separate Week 3 headline. The weather-prediction transition matrix is distinct from the matrix estimated from a training sequence on PDF p. 49. |
| S. Umesh, [GMM_HMM_final.pdf](sources/GMM_HMM_final.pdf) | PDF pp. 25–54 and 65–154 | Corroborating Week 3 reference for EM parameter estimation and Markov models. For Week 4, pp. 75–89 review Viterbi, pp. 90–119 cover HMM/forward inference, and pp. 120–154 cover speech HMM/GMM modeling; this latter range also supports Week 5's overlapping HMM-speech fundamentals. The March 2026 deck's fully observed weather Viterbi lecture remains direct Week 3 scope evidence. |
| *End-to-End ASR*, supplied course deck, [End-to-End-ASR-VW.pdf](sources/End-to-End-ASR-VW.pdf) | PDF pp. 6, 10, and 12 | Supporting visual explanation of the lexicon as the link between acoustic/phoneme sequence hypotheses and words. It is an end-to-end ASR deck, not evidence that its other topics belong to Week 5; the Week 5 screenshot establishes the lexicon lecture placement. No n-gram content was found in searchable text. |
| S. Umesh, *Neural Network Based Language Models*, [Neural Network LM.pdf](sources/Neural%20Network%20LM.pdf) | supplied deck | Not a Week 5 n-gram source: the deck treats one-hot word encoding, embeddings, and neural next-word prediction (including Word2Vec), and does not teach count-based bigram/trigram estimation. It is not used as a count-based N-gram lesson source. |
| *Practice Question Bank: Numerical Problems in Speech Technology*, May 2026, [numerials_BS_ST_qn_bank.pdf](sources/numerials_BS_ST_qn_bank.pdf) | Section 1, Q1–Q3, PDF p. 1 | Sampling and frame arithmetic only. This is a supplied practice bank, not authenticated previous-year examination evidence. The remaining sections are not used to establish Weeks 1–5 scope. |

The supplied Week 5 screenshot confirms the six topic titles and their order, but no dedicated transcript, lecture slides for n-gram estimation, or official question text were found. Searchable attached PDFs included neural-network language-model material and lexicon diagrams in an end-to-end ASR deck; neither supplies Week 5 n-gram lecture content or authentic question wording. Accordingly, count-based n-gram mathematics in Week 5 is identified as an authored explanation derived from the chain rule and maximum-likelihood counts, not as a transcript or attributed lecture derivation. The supplied screenshots establish lecture titles and order; they do not authenticate video transcripts or official AQ/practice/graded-assignment question text. A user-supplied draft labels its generated questions as mock/AI-generated, not authentic previous-year questions; preserve that caveat and do not reconstruct official questions. Numerical examples in notes must be identified as slide examples or generated practice according to their evidence.

## Week 4 Source Checks

- The weather HMM example starts with **Day 0 Cloudy** before the first modeled transition; Day 1 is therefore the Cloudy row of the forecast transition matrix, not an emission-weighted posterior.
- Weather forecast matrix rows/columns are Sunny, Cloudy, Rainy: $A=[[.5,.375,.125],[.25,.125,.625],[.25,.075,.675]]$. The HMM review emission matrix has rows Sunny, Cloudy, Rainy and columns Dry, Dryish, Damp, Soggy: $B=[[.6,.2,.15,.05],[.25,.25,.25,.25],[.05,.1,.35,.5]]$. Keep forecast and observation-conditioned inference distinct.
- PDF pp. 133–134 in the March 2026 deck label a fixed-path quantity as $P(O\mid Q)$, although the displayed product includes the initial-state and transition factors and is the joint path-observation probability $P(Q,O)$ (or path score under the stated model). The printed score `.0471` differs from the product of the displayed factors, `.04826304`; note the correction when using this example.

## Retained Earlier Material

The former Week 1 PDF export is preserved at [archived earlier PDF](archive/week-1-notes-before-scope-correction.pdf). It predates the screenshot-based scope correction and is **not the current Week 1 note**. The current authoritative notes are the linked Markdown documents above.

Earlier generated images not embedded in the revised notes may remain on disk. Only the assets generated by the revised weekly scripts and listed in their current manifests belong to the active weekly material. Unassigned atomic notes are listed separately in [[Knowledge-Map]] rather than presented as assigned weekly coverage unless listed in the Knowledge Map.
