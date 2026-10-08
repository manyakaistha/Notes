---
title: Nyquist Criterion
slug: nyquist-criterion
tags: [ns/concept, ns/sampling]
introduced: "[[week-2-report|Week 2]]"
---

# Nyquist Criterion

A band-limited signal must have its highest retained frequency strictly below half the sampling rate to avoid ambiguity at the Nyquist boundary. For 70 Hz content, the boundary rate is 140 Hz; acquisition requires a rate above that boundary and an anti-aliasing filter with a practical transition margin. A 10 Hz sine sampled at 12 Hz shares its samples with a phase-reversed 2 Hz sine. At exactly two samples per cycle, sample phase can even place every observation at a zero crossing. Frequencies above the new Nyquist limit must also be attenuated before downsampling; later software filtering cannot reliably undo aliasing.

**Why it matters here:** False low-frequency patterns can otherwise enter the feature matrix before learning begins.

**Appears in:** [[week-2-report|Week 2]]

## Related
- [[eeg-sampling-rate]]
- [[band-pass-filtering]]
