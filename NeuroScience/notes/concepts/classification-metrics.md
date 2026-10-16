---
title: Classification Metrics
slug: classification-metrics
tags: [ns/concept, ns/evaluation]
introduced: "[[week-3-report|Week 3]]"
reinforced: []
---

# Classification Metrics

Classification metrics quantify the predictive performance of a model evaluated against ground-truth experimental labels. In binary classification, all predictions partition into a $2 \times 2$ **confusion matrix** comprising True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). From these counts, key metrics are derived:
- **Accuracy:** $(\text{TP} + \text{TN}) / (\text{TP} + \text{TN} + \text{FP} + \text{FN})$
- **Precision:** $\text{TP} / (\text{TP} + \text{FP})$
- **Recall (Sensitivity):** $\text{TP} / (\text{TP} + \text{FN})$
- **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

Under severe class imbalance (e.g., clinical seizure detection, where events occupy $< 1\%$ of recording time), accuracy suffers from the **accuracy paradox**: a degenerate model predicting the negative class for every sample achieves $> 99\%$ accuracy while exhibiting $0\%$ recall, missing every clinical event. In electrophysiology, reporting precision, recall, F1-scores, and explicit confusion matrices is essential.

**Why it matters here:** Metrics determine whether an algorithm provides true biological discriminative capability or merely reflects baseline class prevalence.

**Appears in:** [[week-3-report|Week 3]] · [[w3d2|W3D2]] · [[w3d4|W3D4]] · [[w3d5|W3D5]]

## Related
- [[supervised-learning]]
- [[experimental-labels]]
- [[data-leakage]]
