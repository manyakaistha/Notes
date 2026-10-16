---
title: Gradient Stiffness
slug: gradient-stiffness
tags: [pinn/concept, pinn/optimization, pinn/pathologies]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Gradient Stiffness

Gradient Stiffness is a severe optimization pathology endemic to physics-informed neural networks arising from dramatic disparities in the eigenvalues and condition numbers of the Hessian matrices associated with different loss components. In composite PINN objectives, the total loss $\mathcal{L} = \lambda_{\mathrm{pde}} \mathcal{L}_{\mathrm{pde}} + \lambda_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}} + \lambda_{\mathrm{ic}} \mathcal{L}_{\mathrm{ic}}$ requires backpropagating through higher-order coordinate derivative graphs for the PDE residual while computing zero-order point regressions for boundary and initial conditions. Consequently, the backpropagated gradient norms $\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|$ often exceed the interior residual gradients $\|\nabla_\theta \mathcal{L}_{\mathrm{pde}}\|$ by two to four orders of magnitude. Under standard gradient descent or unweighted Adam optimization, the network parameters align almost exclusively with boundary satisfaction while the interior PDE residual stagnates in sub-optimal local minima, causing the solver to generate physically degenerate solutions.

**Why it matters here:** Diagnosing gradient stiffness explains why naïve neural network solvers fail on simple differential equations and motivates dynamic multi-objective balancing algorithms.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d4|W3D4]] · [[w4d3|W4D3]]

## Related
- [[learning-rate-annealing]]
- [[loss-formulation-pinn]]
- [[spectral-bias]]
