---
title: Band-Power Features
slug: band-power-features
tags: [ns/concept, ns/features]
introduced: "[[week-2-report|Week 2]]"
---

# Band-Power Features

Band power measures mean-square signal strength over a specified frequency range. Integrating a PSD in µV²/Hz over frequency in Hz yields power in µV². For a zero-mean signal, variance equals mean-square power; variance after band-pass filtering estimates the power of that filtered output. Transition weighting, finite-window edges, and limited frequency resolution make that estimate differ from ideal rectangular-band power. A complete-cycle sine with peak amplitude `A` has power `A²/2`. Alpha power is not a universal relaxation label, and EMG or blinks can contribute to apparent band strength. Always specify band boundaries, channel order, units, and window duration.

**Why it matters here:** Band-power columns hand a biologically motivated but imperfect representation to Week 3 classifiers.

**Appears in:** [[week-2-report|Week 2]]

## Related
- [[band-pass-filtering]]
- [[time-domain-features]]
