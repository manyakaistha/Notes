---
title: Band-Pass Filtering
slug: band-pass-filtering
tags: [ns/concept, ns/signal-processing]
introduced: "[[week-2-report|Week 2]]"
---

# Band-Pass Filtering

A band-pass filter largely retains frequencies between two selected cutoffs while attenuating slower and faster components. It combines high-pass and low-pass goals, but a single band-pass design need not match a separately cascaded pair exactly. Real responses have transition bands rather than sharp walls. Forward-backward filtering avoids phase delay offline, squares the amplitude response, and changes boundary behavior; it needs future samples and is unsuitable as written for streaming inference. A 1–30 Hz choice is useful for a restricted Alpha/Beta demonstration, but damages Delta near the lower cutoff and suppresses Gamma. Frequency filtering cannot distinguish neural and artifact sources within the same band.

**Why it matters here:** Cutoff choices determine which information remains available to the model.

**Appears in:** [[week-2-report|Week 2]]

## Related
- [[artifact-contamination]]
- [[band-power-features]]
