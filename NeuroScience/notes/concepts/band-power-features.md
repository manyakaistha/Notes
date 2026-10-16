---
title: Band-Power Features
slug: band-power-features
tags: [ns/concept, ns/features]
introduced: "[[week-2-report|Week 2]]"
reinforced: ["[[week-3-report|Week 3]]"]
---

# Band-Power Features

Band power measures mean-square signal strength over a specified frequency range. Integrating a PSD in µV²/Hz over frequency in Hz yields power in µV². For a zero-mean signal, variance equals mean-square power; variance after band-pass filtering estimates the power of that filtered output. Transition weighting, finite-window edges, and limited frequency resolution make that estimate differ from ideal rectangular-band power. A complete-cycle sine with peak amplitude `A` has power `A²/2`. Alpha power is not a universal relaxation label, and EMG or blinks can contribute to apparent band strength. Always specify band boundaries, channel order, units, and window duration.

**Why it matters here:** Band-power columns hand a biologically motivated representation (such as sensorimotor mu and beta power) to supervised classifiers, but exhibit extensive inter-trial overlap that necessitates regularized modeling.

**Appears in:** [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[w3d1|W3D1]] · [[w3d2|W3D2]]

## Related
- [[band-pass-filtering]]
- [[time-domain-features]]
- [[feature-matrix]]
- [[logistic-regression]]
