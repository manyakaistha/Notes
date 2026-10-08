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
| **3** | Week 3 — report not yet created | How does ML turn neural data into predictions? | planned |
| **4** | Week 4 — report not yet created | How do modern architectures fit this data? | planned |

Planned weeks use plain text until their report files exist, so the graph contains no unresolved report links.

### Week 2 Daily Notes
- [[w2d1|W2D1]]: Signals, Time Series & Sampling Fundamentals (incl. Week 1 Diagnostic Verification)
- [[w2d2|W2D2]]: The Nyquist Intuition & Artifact Contamination
- [[w2d3|W2D3]]: Basic Filtering — The Mathematics of Signal Rescue
- [[w2d4|W2D4]]: Feature Extraction — From Matrices to Meaning
- [[w2d5|W2D5]]: The End-to-End Pipeline & Final Review (incl. Week 2 Mastery Test)

## Concept spine

- **Data & measurement** — [[eeg-sampling-rate]] · [[nyquist-criterion]] · [[artifact-contamination]]
- **Representation** — [[band-pass-filtering]] · [[time-domain-features]] · [[band-power-features]]
- **Learning (Week 3, planned)** — feature matrices, experimental labels, held-out evaluation, subject/session generalization.

## Central paradigm

**Brain → Neural Activity → Neural Data → AI → Prediction / Interpretation**

Defined in [[week-1-report|Week 1]] §7; refined into a sampling, preprocessing, and feature-extraction pipeline in [[week-2-report|Week 2]] §§2–7.

## The handoff to Week 3

Week 2 supplies timestamped windows and fixed feature columns. A window is a training observation, while its label comes from the experimental design. Week 3 must test whether the relationship generalizes across recordings and subjects instead of exploiting artifacts or shared windows.
