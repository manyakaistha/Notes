---
title: Boundary and Initial Conditions
slug: boundary-initial-conditions
tags: [pinn/concept, pinn/calculus, pinn/pde]
introduced: "[[week-1-report|Week 1]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Boundary and Initial Conditions

Boundary Conditions (BCs) and Initial Conditions (ICs) provide the auxiliary mathematical constraints required to isolate a unique, physically admissible solution from an infinite family of general differential equation solutions. Initial conditions specify the state of the dynamical system $u(0, \mathbf{x}) = u_0(\mathbf{x})$ and velocity $\partial_t u(0, \mathbf{x}) = v_0(\mathbf{x})$ at time $t=0$. Boundary conditions define constraints on the domain boundary $\partial\Omega$ across time:
1. **Dirichlet BC:** Prescribes the field value $u(t, \mathbf{x}) = g(\mathbf{x})$ on $\partial\Omega$ (e.g., fixed wall temperature).
2. **Neumann BC:** Prescribes the normal flux $\nabla u \cdot \hat{\mathbf{n}} = h(\mathbf{x})$ on $\partial\Omega$ (e.g., insulated adiabatic boundary).
3. **Robin BC:** Prescribes a linear combination $a u + b (\nabla u \cdot \hat{\mathbf{n}}) = c$.
In standard PINNs, these conditions are enforced as "soft constraints" via mean-squared boundary penalties in the objective loss functional, or as "hard constraints" via exact mathematical ansatz architectures (e.g., $u_\theta(\mathbf{x}) = g(\mathbf{x}) + D(\mathbf{x}) \cdot \mathcal{N}_\theta(\mathbf{x})$ where $D(\mathbf{x})$ vanishes identically on $\partial\Omega$).

**Why it matters here:** Without rigorously enforcing boundary and initial conditions, minimizing the PDE residual yields trivial or physically meaningless null solutions.

**Appears in:** [[week-1-report|Week 1]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w1d4|W1D4]] · [[w3d1|W3D1]] · [[w4d1|W4D1]]

## Related
- [[ordinary-differential-equation]]
- [[partial-differential-equation]]
- [[loss-formulation-pinn]]
