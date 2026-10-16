---
title: Two-Phase Optimization
slug: two-phase-optimization
tags: [pinn/concept, pinn/optimization, pinn/lbfgs]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Two-Phase Optimization

Two-Phase Optimization is the authoritative training strategy in scientific machine learning that synergistically pairs first-order stochastic gradient methods with second-order quasi-Newton algorithms. In **Phase 1**, the network is trained for several hundred to thousands of epochs using the Adam optimizer with a moderate learning rate ($\eta \in [10^{-3}, 10^{-2}]$) and momentum. Adam's stochastic robustness navigates complex, non-convex loss landscapes, evades sharp local minima, and brings parameter vectors into the quadratic basin of attraction. Once first-order loss gradients plateau, the solver transitions to **Phase 2**, deploying full-batch Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimization with Strong Wolfe line searches. L-BFGS constructs low-rank approximations of the inverse Hessian matrix, exhibiting superlinear convergence rates that drive physics residuals down from $10^{-3}$ to near machine precision ($10^{-7}$ to $10^{-9}$).

**Why it matters here:** Two-phase optimization overcomes first-order gradient stagnation and enables PINNs to achieve engineering-grade accuracy comparable to classical spectral and finite element methods.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d5|W3D5]] · [[w4d3|W4D3]]

## Related
- [[gradient-stiffness]]
- [[learning-rate-annealing]]
- [[loss-formulation-pinn]]
