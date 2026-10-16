---
title: Differential Residual Operator
tags:
  - pinn/concept
  - mathematics/calculus
  - sciml/residual
---

# Differential Residual Operator

The **differential residual operator** $\mathcal{R}[u](\mathbf{x})$ quantifies the degree to which a candidate function $u(\mathbf{x})$ violates a governing differential equation $\mathcal{N}[u](\mathbf{x}) = f(\mathbf{x})$. Formally, it is defined continuously across the domain $\Omega \subset \mathbb{R}^d$ as:

$$\mathcal{R}[u](\mathbf{x}) \equiv \mathcal{N}[u](\mathbf{x}) - f(\mathbf{x})$$

When $u(\mathbf{x})$ is an exact classical solution, $\mathcal{R}[u](\mathbf{x}) \equiv 0$ for all $\mathbf{x} \in \Omega$. Unlike discrete algebraic residuals derived from finite difference stencils (which are defined only at mesh vertices), the differential residual is a continuous scalar or vector field that can be queried at any coordinate point in space-time.

### Why it matters here
In Physics-Informed Neural Networks (PINNs), the differential residual transforms an unconstrained neural surrogate $u_\theta(\mathbf{x})$ into a physics-constrained solver. By evaluating $\mathcal{R}[u_\theta](\mathbf{x})$ using automatic differentiation and minimizing the mean squared residual over collocation points, the network learns to satisfy the underlying physical conservation laws without requiring discretization grids.

**Appears in:** [[w1d5]], [[week-1-report]], [[w3d4]], [[week-3-report]], [[w4d1]], [[week-4-report]]

## Related links
- [[physics-residual]]
- [[collocation-points]]
- [[partial-differential-equation]]
- [[loss-formulation-pinn]]
