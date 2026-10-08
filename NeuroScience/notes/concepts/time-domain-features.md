---
title: Time-Domain Features
slug: time-domain-features
tags: [ns/concept, ns/features]
introduced: "[[week-2-report|Week 2]]"
---

# Time-Domain Features

Mean, variance, and peak-to-peak amplitude summarize an EEG window without reconstructing its temporal sequence. Mean is the average level in µV; population variance is the mean squared deviation in µV²; peak-to-peak amplitude is the difference between maximum and minimum voltage in µV. Sorting a window preserves these statistics while destroying its waveform ordering. High-pass filtering suppresses DC, often making mean a weak feature, although short windows can retain nonzero means through edge effects and transients. Large variance or extrema may reflect artifacts as well as neural fields. Timestamped windows preserve coarse localization, but whole-record summaries contain no explicit event onset.

**Why it matters here:** Choose summaries that preserve the information required by the prediction target.

**Appears in:** [[week-2-report|Week 2]]

## Related
- [[band-power-features]]
- [[artifact-contamination]]
