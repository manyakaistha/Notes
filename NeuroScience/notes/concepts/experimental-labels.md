---
title: Experimental Labels
slug: experimental-labels
tags: [ns/concept, ns/data-representation]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Experimental Labels

Experimental labels constitute the ground-truth target vector $y \in \mathcal{Y}^N$ in a supervised neurotechnology pipeline. In electrophysiology, labels are derived from synchronized external event markers, stimulus presentation timestamps, behavioral response triggers, or clinical annotations (e.g., expert-scored seizure intervals or sleep stages). In motor imagery classification, labels are typically encoded as binary integers (e.g., $y_i = 0$ for left fist imagery, $y_i = 1$ for right fist imagery). An experimental label indicates the instruction given to or the state observed from the participant; it does not guarantee that the participant faithfully executed the mental task on that specific trial.

**Why it matters here:** Without ground truth labels, supervised models cannot optimize discriminative boundaries. Imperfect mental compliance or misaligned event timestamps introduce label noise that degrades model decodability.

**Appears in:** [[week-3-report|Week 3]] · [[w3d1|W3D1]] · [[w3d4|W3D4]]

## Related
- [[supervised-learning]]
- [[feature-matrix]]
- [[classification-metrics]]
