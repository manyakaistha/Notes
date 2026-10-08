---
title: Collocation Points
slug: collocation-points
tags: [pinn/concept, pinn/sampling, pinn/meshfree]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Collocation Points

Collocation points are discrete spatial and temporal evaluation coordinates $\mathbf{x}_c = (t_i, x_i) \in \Omega$ sampled across the continuous problem domain where the neural network surrogate is required to satisfy the governing differential equation. Unlike traditional numerical grids (e.g., Finite Difference or Finite Element meshes), collocation points require no topological connectivity, element triangulation, or neighbor graph. They serve purely as evaluation sites for automatic differentiation to evaluate the physics residual $\mathcal{R}(u_\theta(\mathbf{x}_c))$. Sampling strategies include uniform Cartesian lattices, Monte Carlo random uniform draws, Latin Hypercube Sampling (LHS), and residual-based adaptive distribution (RAD) that focuses points in regions of high physical gradients or shocks.

**Why it matters here:** Collocation points liberate scientific computing from mesh generation, enabling seamless solution of PDEs in high dimensions and complex, irregular geometries.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d3|W3D3]] · [[w4d1|W4D1]]

## Related
- [[physics-residual]]
- [[discretization-error]]
- [[loss-formulation-pinn]]
