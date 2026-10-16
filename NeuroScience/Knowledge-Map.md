---
title: Neuroscience Knowledge Map
tags: [ns/moc]
---

# AI + Neuroscience — Knowledge Map

## The 4-week spine

| Week | Report | Core Question | Status |
| --- | --- | --- | --- |
| **1** | [[week-1-report\|Week 1]] | What is neural data? | complete |
| **2** | [[week-2-report\|Week 2]] | What happens to a signal before AI uses it? | complete |
| **3** | [[week-3-report\|Week 3]] | How does ML turn neural data into predictions? | complete |
| **4** | [[week-4-report\|Week 4]] | How do modern deep architectures learn spatiotemporal representations from neural data? | complete |


### Week 2 Daily Notes
- [[w2d1|W2D1]]: Signals, Time Series & Sampling Fundamentals (incl. Week 1 Diagnostic Verification)
- [[w2d2|W2D2]]: The Nyquist Intuition & Artifact Contamination
- [[w2d3|W2D3]]: Basic Filtering — The Mathematics of Signal Rescue
- [[w2d4|W2D4]]: Feature Extraction — From Matrices to Meaning
- [[w2d5|W2D5]]: The End-to-End Pipeline & Final Review (incl. Week 2 Mastery Test)

### Week 3 Daily Notes
- [[w3d1|W3D1]]: From Neural Features to Supervised Datasets (incl. Week 2 Diagnostic Verification)
- [[w3d2|W3D2]]: Logistic Regression & Held-Out Predictions
- [[w3d3|W3D3]]: Decision Trees & Support Vector Machines
- [[w3d4|W3D4]]: Evaluation, Metrics & The Overfitting Hazard
- [[w3d5|W3D5]]: Subject Generalization & Week 3 Synthesis (incl. Comprehensive Mastery Evaluation)

### Week 4 Daily Notes
- [[w4d1|W4D1]]: Neural Networks — Artificial Neurons, Forward Propagation, and Backpropagation
- [[w4d2|W4D2]]: Convolutional Neural Networks — Learned Local EEG Features and Spatiotemporal Filtering
- [[w4d3|W4D3]]: Sequential Data, Recurrent Neural Networks, and LSTMs
- [[w4d4|W4D4]]: Attention Mechanisms, Self-Attention, and Transformers for EEG
- [[w4d5|W4D5]]: Applications in Neuroscience, Empirical Synthesis, and Month 1 Capstone

## Concept spine

- **Data & measurement** — [[eeg-sampling-rate]] · [[nyquist-criterion]] · [[artifact-contamination]]
- **Representation** — [[band-pass-filtering]] · [[time-domain-features]] · [[band-power-features]]
- **Learning (Supervised)** — [[supervised-learning]] · [[feature-matrix]] · [[experimental-labels]] · [[logistic-regression]] · [[decision-tree]] · [[support-vector-machine]] · [[classification-metrics]] · [[overfitting]] · [[data-leakage]] · [[subject-generalization]]
- **Deep Representation Learning** — [[artificial-neuron]] · [[activation-function]] · [[forward-propagation]] · [[backpropagation]] · [[convolution]] · [[pooling]] · [[recurrent-neural-network]] · [[long-short-term-memory]] · [[self-attention]] · [[transformer]] · [[representation-learning]]

## Central paradigm

**Brain → Neural Activity → Neural Data → AI → Prediction / Interpretation**

Defined in [[week-1-report|Week 1]] §7; refined into a sampling, preprocessing, and feature-extraction pipeline in [[week-2-report|Week 2]] §§2–7; grounded into supervised decoding ($X \in \mathbb{R}^{N \times P} \to y \in \{0, 1\}^N$), decision boundaries, metric auditing, and cross-subject generalization in [[week-3-report|Week 3]]; extended to CNN representation learning, recurrent/attention mechanisms, and scientific application formulations in [[week-4-report|Week 4]].

## The handoff to Week 4 and Month 2

Week 3 established handcrafted feature representations and subject-aware evaluation. Week 4 extends this to trainable representations: an explicit neuron update, exact CNN tensor/receptive-field analysis, and untrained recurrent/attention mechanism demonstrations. In the 450-trial, 10-subject LOSO benchmark, the CNN achieves 55.3% pooled accuracy versus 52.7% for logistic regression, with heterogeneous subject outcomes. The small descriptive gain does not establish general superiority or identify anatomical causes. Month 2 can investigate whether spatial and latent representations improve transfer under the same evaluation discipline.

### Forward Handoff to Month 2 / Week 5
Month 2 expands beyond single sensor pairs to high-density spatial arrays:
- **Spatial Filtering:** Common Spatial Patterns (CSP), spatial Laplacian rereferencing, and Independent Component Analysis (ICA).
- **Unsupervised Representation Learning & Manifolds:** Principal Component Analysis (PCA), t-SNE, UMAP, and autoencoders for discovery of low-dimensional latent neural manifolds.
- **Foundation Models:** Self-supervised masked autoencoding on massive public multi-cohort EEG archives.
