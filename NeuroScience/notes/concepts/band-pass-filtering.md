---
title: Band-Pass Filtering
slug: band-pass-filtering
tags: [ns/concept, ns/signal-processing]
introduced: "[[week-2-report|Week 2]]"
reinforced: ["[[week-3-report|Week 3]]"]
---

# Band-Pass Filtering

A band-pass filter largely retains frequencies between two selected cutoffs while attenuating slower and faster components. It combines high-pass and low-pass goals, but a single band-pass design need not match a separately cascaded pair exactly. Real responses have transition bands rather than sharp walls. Forward-backward filtering avoids phase delay offline, squares the amplitude response, and changes boundary behavior; it needs future samples and is unsuitable as written for streaming inference. A 1–35 Hz choice is standard for isolating sensorimotor mu and beta rhythms, but damages slow cortical potentials near the lower cutoff and suppresses Gamma. Frequency filtering cannot distinguish neural and artifact sources within the same band.

**Why it matters here:** Cutoff choices determine which information remains available to downstream machine learning classifiers, and must be fit without leaking cross-run boundaries.

**Appears in:** [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[w3d1|W3D1]]

## Related
- [[artifact-contamination]]
- [[band-power-features]]
- [[data-leakage]]
