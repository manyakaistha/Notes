---
title: Scientific Machine Learning (SciML)
slug: scientific-machine-learning
tags: [pinn/concept, pinn/sciml, pinn/foundations]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Scientific Machine Learning (SciML)

Scientific Machine Learning (SciML) is the convergence of mechanistic physical modeling (differential equations, conservation laws, thermodynamics) and modern data-driven statistical learning (neural networks, kernel methods, automatic differentiation). Classical physics modeling provides interpretability, physical consistency, and generalizability, but struggles with high dimensionality, ill-posed inverse problems, missing parameters, and noisy sensor data. Conversely, standard deep learning excels at finding statistical representations in high-dimensional datasets, but acts as an uninterpretable "black box" that requires massive training corpora, violates fundamental conservation principles (energy, mass, momentum), and fails catastrophically when extrapolating outside the support of training data. SciML embeds prior physical laws directly into the learning framework—via network architectures, loss penalties, or differentiable simulators—achieving high data efficiency, physical reliability, and robustness to noise.

**Why it matters here:** SciML defines the overarching paradigm of modern computational science, within which PINNs constitute the foundational methodology.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d1|W3D1]] · [[w4d1|W4D1]]

## Related
- [[physics-residual]]
- [[collocation-points]]
- [[spectral-bias]]
