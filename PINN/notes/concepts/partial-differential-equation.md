---
title: Partial Differential Equations
slug: partial-differential-equation
tags: [pinn/concept, pinn/calculus, pinn/pde]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Partial Differential Equations

A Partial Differential Equation (PDE) is a mathematical relation involving an unknown multivariable field $u(t, x_1, \dots, x_d)$ and its partial derivatives across two or more independent variables. PDEs model continuous physical transport, field distributions, and wave propagation throughout space-time. Canonical linear and non-linear second-order PDEs are classified by their discriminant into three fundamental physical regimes:
1. **Elliptic:** (e.g., Laplace / Poisson $\nabla^2 u = f$) describing static equilibrium, electrostatic potentials, and steady-state thermal distributions.
2. **Parabolic:** (e.g., Heat / Diffusion $\partial_t u = \alpha \nabla^2 u$) describing irreversible dissipation, thermal conduction, and brownian dispersion.
3. **Hyperbolic:** (e.g., Wave $\partial_{tt} u = c^2 \nabla^2 u$) describing finite-speed reversible wave propagation and acoustic vibrations.
Non-linear PDEs like the Navier-Stokes equations governing fluid dynamics and the Burgers' equation $\partial_t u + u \partial_x u = \nu \partial_{xx} u$ introduce advection, shocks, and turbulence.

**Why it matters here:** PDEs are the universal mathematical language of continuum mechanics, fluid dynamics, electromagnetism, and thermal transport that PINNs are constructed to solve.

**Appears in:** [[week-1-report|Week 1]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w1d4|W1D4]] · [[w3d1|W3D1]] · [[w4d3|W4D3]]

## Related
- [[ordinary-differential-equation]]
- [[boundary-initial-conditions]]
- [[physics-residual]]
