---
title: Artifact Contamination
slug: artifact-contamination
tags: [ns/concept, ns/eeg]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-2-report|Week 2]]"]
---

# Artifact Contamination

An EEG artifact is a non-neural contribution to recorded voltage, arising from ocular activity, muscle activity, electrode contact changes, or the environment. Sources coexist in the same channel and can overlap genuine neural frequency ranges. Eye blinks can inflate low-frequency power; jaw and neck EMG can inflate Beta or Gamma; mains interference contributes a narrow spectral peak. The label “noise” depends on the scientific question, whereas source attribution requires evidence beyond waveform appearance. Filters can attenuate selected frequencies but cannot identify overlapping sources. Inspect channels, timing, annotations, and residuals, and reject severe contaminated windows when appropriate.

**Why it matters here:** A classifier may predict labels accurately by exploiting artifacts that correlate with the experimental conditions.

**Appears in:** [[week-1-report|Week 1]] · [[week-2-report|Week 2]]

## Related
- [[band-pass-filtering]]
- [[band-power-features]]
