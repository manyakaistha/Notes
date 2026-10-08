---
title: EEG Sampling Rate
slug: eeg-sampling-rate
tags: [ns/concept, ns/sampling]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-2-report|Week 2]]"]
---

# EEG Sampling Rate

Sampling rate specifies how many voltage measurements a recording system takes each second. At 250 Hz, consecutive samples are separated by `1/250 = 0.004 s = 4 ms`. A recording of `C` channels for `D` s has shape `(C, sampling_rate × D)` when the sample count is integral. Signal frequency instead specifies how fast an oscillatory component repeats. These are different quantities even though both use Hz. Timing depends on preserved sample order, rate metadata, and acquisition quality. A higher rate permits a wider retained spectrum but cannot reverse spatial mixing or establish that a voltage source is neural.

**Why it matters here:** Correct time axes, window durations, and resampling depend on recording-rate metadata.

**Appears in:** [[week-1-report|Week 1]] · [[week-2-report|Week 2]]

## Related
- [[nyquist-criterion]]
- [[artifact-contamination]]
