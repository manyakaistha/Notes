---
title: Time-Domain Features
slug: time-domain-features
tags: [ns/concept, ns/features]
introduced: "[[week-2-report|Week 2]]"
reinforced: ["[[week-3-report|Week 3]]"]
---

# Time-Domain Features

Mean, variance, and peak-to-peak amplitude summarize an EEG window without reconstructing its temporal sequence. Mean is the average level in µV; population variance is the mean squared deviation in µV²; peak-to-peak amplitude is the difference between maximum and minimum voltage in µV. Sorting a window preserves these statistics while destroying its waveform ordering. High-pass filtering suppresses DC, often making mean a weak feature, although short windows can retain nonzero means through edge effects and transients. Large variance or extrema may reflect artifacts as well as neural fields. Timestamped windows preserve coarse localization, but whole-record summaries contain no explicit event onset.

**Why it matters here:** Time-domain features combine with frequency-domain features to form the columns of the supervised feature matrix, but their disparate numerical scales require mandatory standardization.

**Appears in:** [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[w3d1|W3D1]]

## Related
- [[band-power-features]]
- [[artifact-contamination]]
- [[feature-matrix]]
