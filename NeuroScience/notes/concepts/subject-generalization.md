---
title: Subject Generalization
slug: subject-generalization
tags: [ns/concept, ns/evaluation]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Subject Generalization

Subject generalization refers to the ability of a predictive model trained on a cohort of individuals to accurately decode neural activity from an entirely new, unseen subject without subject-specific retraining or calibration. In electrophysiology, zero-shot subject generalization is notoriously difficult: while within-subject models (evaluated via Leave-One-Run-Out, LORO) often achieve robust decodability (e.g., $70\text{--}80\%$ accuracy in responsive subjects), subject-independent models (evaluated via Leave-One-Subject-Out, LOSO) frequently collapse toward the chance baseline ($51\%$). This generalization gap is driven by **biological heterogeneity**: individual differences in cortical gyral folding, skull thickness and conductivity, resting alpha peak frequencies, and BCI illiteracy.

**Why it matters here:** Evaluating models across unseen subjects is the ultimate test of clinical and technological viability for plug-and-play neuroprosthetics. High subject-dependent accuracy must never be confused with generalizable cohort transfer.

**Appears in:** [[week-3-report|Week 3]] · [[w3d5|W3D5]]

## Related
- [[supervised-learning]]
- [[data-leakage]]
- [[support-vector-machine]]
- [[classification-metrics]]
