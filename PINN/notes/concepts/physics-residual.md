---
title: Physics Residual
slug: physics-residual
tags: [pinn/concept, pinn/physics, pinn/differential-operators]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Physics Residual

The physics residual $\mathcal{R}(\mathbf{x}; \theta)$ is the quantitative algebraic deficit obtained when a candidate surrogate solution $u_\theta(\mathbf{x})$ is substituted directly into a governing differential operator equation $\mathcal{N}[u] - f = 0$. For an exact physical solution, $\mathcal{R}(\mathbf{x}) \equiv 0$ everywhere in the continuous domain $\Omega$. For a parameterized neural network, $\mathcal{R}(\mathbf{x}; \theta) = \mathcal{N}[u_\theta(\mathbf{x})] - f(\mathbf{x}) \ne 0$. Evaluating this residual requires spatial and temporal derivatives of the network's output with respect to its continuous coordinate inputs, computed analytically to machine precision via automatic differentiation. The mean squared norm of this residual vector across all collocation points forms the core PDE loss functional:

$$\mathcal{L}_{\mathrm{pde}}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} \|\mathcal{R}(\mathbf{x}_i; \theta)\|^2$$

**Why it matters here:** The physics residual acts as a self-supervised inductive prior that allows neural networks to solve differential equations with zero labeled training data.

**Appears in:** [[week-1-report|Week 1]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w1d5|W1D5]] · [[w3d4|W3D4]] · [[w4d2|W4D2]]

## Related
- [[collocation-points]]
- [[automatic-differentiation]]
- [[loss-formulation-pinn]]
