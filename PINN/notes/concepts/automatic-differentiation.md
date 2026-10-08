---
title: Automatic Differentiation
slug: automatic-differentiation
tags: [pinn/concept, pinn/calculus, pinn/autodiff]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-2-report|Week 2]]", "[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Automatic Differentiation

Automatic Differentiation (autodiff, AD) is an algorithmic technique for evaluating exact derivatives of numeric computer programs using the chain rule of differential calculus. Unlike numerical finite differences, which suffer from truncation error $\mathcal{O}(h)$ and catastrophic floating-point cancelation $\mathcal{O}(\epsilon/h)$, and unlike symbolic differentiation, which triggers exponential expression swell, automatic differentiation computes exact derivative evaluations down to machine floating-point epsilon ($\sim 10^{-16}$ in double precision) with linear time complexity. In scientific machine learning and PINNs, autodiff is executed in two directions: reverse-mode autodiff with respect to spatial-temporal coordinates $\mathbf{x} = (t, x)$ to compute differential operators ($\partial_t u, \nabla^2 u$), and reverse-mode autodiff with respect to network parameters $\theta$ (backpropagation) to compute optimization update gradients $\nabla_\theta \mathcal{L}$.

**Why it matters here:** Autodiff allows neural networks to evaluate continuous differential equations without constructing discrete numerical grids or approximating derivative stencils.

**Appears in:** [[week-1-report|Week 1]] · [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w1d2|W1D2]] · [[w3d2|W3D2]] · [[w4d2|W4D2]]

## Related
- [[physics-residual]]
- [[discretization-error]]
- [[backpropagation]]
