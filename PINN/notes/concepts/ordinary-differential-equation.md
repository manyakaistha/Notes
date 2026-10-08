---
title: Ordinary Differential Equations
slug: ordinary-differential-equation
tags: [pinn/concept, pinn/calculus, pinn/ode]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Ordinary Differential Equations

An Ordinary Differential Equation (ODE) is an equation containing an unknown function of a single independent variable (typically time $t$) along with its derivatives:

$$\mathcal{F}\left(t, u(t), \frac{du}{dt}, \frac{d^2u}{dt^2}, \dots, \frac{d^n u}{dt^n}\right) = 0$$

ODEs model dynamic physical systems with lumped spatial parameters, such as the classical damped harmonic oscillator ($m \ddot{u} + c \dot{u} + k u = 0$), chemical kinetics, biological neural firing dynamics (FitzHugh-Nagumo), and planetary mechanics. The order of an ODE is the highest derivative present. To yield a well-posed Initial Value Problem (IVP), an $n$-th order ODE requires $n$ independent initial conditions at $t=t_0$. In PINNs, an ODE is approximated by a neural network $u_\theta(t): \mathbb{R} \to \mathbb{R}$, where derivatives with respect to $t$ are calculated via automatic differentiation and compared against the differential operator.

**Why it matters here:** ODEs form the fundamental benchmark sandbox for developing, debugging, and proving physics-informed learning architectures before tackling multi-dimensional PDEs.

**Appears in:** [[week-1-report|Week 1]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w1d3|W1D3]] · [[w4d1|W4D1]]

## Related
- [[partial-differential-equation]]
- [[boundary-initial-conditions]]
- [[physics-residual]]
