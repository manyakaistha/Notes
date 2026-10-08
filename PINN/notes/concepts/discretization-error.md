---
title: Discretization Error
slug: discretization-error
tags: [pinn/concept, pinn/numerics, pinn/error-analysis]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Discretization Error

Discretization error is the discrepancy between a continuous mathematical model (an ODE or PDE defined on an uncountable continuum $\Omega$) and its discrete numerical approximation generated on a finite mesh of step size $h$ (Finite Difference, Finite Volume, or Finite Element methods). Discretization error is governed by **truncation error** (the error introduced by approximating continuous differential operators with truncated Taylor series stencils) and **roundoff error** (accumulation of finite IEEE 754 floating-point errors). In classical mesh-based solvers, truncation error scales polynomially with grid spacing $\mathcal{O}(h^p)$, creating a severe trade-off known as the **curse of dimensionality**: resolving a $d$-dimensional domain with $N$ points per axis requires $N^d$ degrees of freedom, rendering 4D+ physical simulations intractable.

**Why it matters here:** PINNs operate as continuous, mesh-free function approximators, bypassing classical spatial discretization grids and opening computational avenues in high-dimensional physics.

**Appears in:** [[week-1-report|Week 1]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w1d5|W1D5]] · [[w3d2|W3D2]] · [[w4d4|W4D4]]

## Related
- [[collocation-points]]
- [[automatic-differentiation]]
- [[physics-residual]]
