---
title: Inverse Parameter Discovery
slug: inverse-parameter-discovery
tags: [pinn/concept, pinn/inverse, pinn/system-identification]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Inverse Parameter Discovery

Inverse Parameter Discovery is the scientific machine learning methodology of simultaneously inferring latent spatiotemporal state fields $u(\mathbf{x}, t)$ and discovering unknown, uncalibrated physical coefficients $\boldsymbol{\lambda} \in \mathbb{R}^p$ (e.g., kinematic viscosity $\nu$, thermal conductivity $\alpha$, Reynolds stress coefficients, or spring-damping parameters) from sparse, noisy empirical sensor observations. In a PINN formulation, the unknown physical constants are instantiated as learnable PyTorch parameters `nn.Parameter` and embedded directly into the differential physics residual operator: $\mathcal{R}(u_\theta; \boldsymbol{\lambda}) = \mathcal{N}_{\mathbf{x}}[u_\theta; \boldsymbol{\lambda}] - f$. The composite loss functional simultaneously minimizes the data regression error over sensor probes and the differential residual over unmonitored interior collocation points. As the network aligns with observation points, backpropagation through the secondary computational graph drives $\boldsymbol{\lambda}$ toward true physical values with high robustness to measurement noise up to $10\%$.

**Why it matters here:** Parameter discovery transforms PINNs from forward differential equation solvers into automated scientific discovery and system identification engines.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d5|W3D5]] · [[w4d4|W4D4]]

## Related
- [[physics-residual]]
- [[scientific-machine-learning]]
- [[loss-formulation-pinn]]
- [[two-phase-optimization]]
