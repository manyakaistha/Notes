---
title: Quasi-Newton Optimization & L-BFGS
tags:
  - pinn/concept
  - optimization/algorithms
---

# Quasi-Newton Optimization & L-BFGS

**Quasi-Newton optimization** algorithms minimize non-linear objective functionals $\mathcal{L}(\theta)$ by iteratively approximating the inverse Hessian matrix $\mathbf{H}^{-1} = \left( \nabla_\theta^2 \mathcal{L} \right)^{-1}$ without explicitly computing or storing second-order parameter derivatives.

The **Limited-Memory BFGS (L-BFGS)** algorithm maintains a rolling history of the $m$ most recent parameter displacements $\mathbf{s}_k = \theta_{k+1} - \theta_k$ and gradient differences $\mathbf{y}_k = \nabla \mathcal{L}_{k+1} - \nabla \mathcal{L}_k$. Utilizing the two-loop recursion algorithm, L-BFGS computes the search direction $\mathbf{p}_k = -\mathbf{H}_k^{-1} \nabla \mathcal{L}_k$ in $\mathcal{O}(m P)$ operations, where $P$ is parameter count. Line searches are governed by the Strong Wolfe conditions:

$$\mathcal{L}(\theta_k + \alpha_k \mathbf{p}_k) \le \mathcal{L}(\theta_k) + c_1 \alpha_k \nabla \mathcal{L}_k^T \mathbf{p}_k$$
$$|\nabla \mathcal{L}(\theta_k + \alpha_k \mathbf{p}_k)^T \mathbf{p}_k| \le c_2 |\nabla \mathcal{L}_k^T \mathbf{p}_k|$$

### Why it matters here
In physics-informed learning, loss landscapes are characterized by ill-conditioned parabolic ravines where first-order gradient descent (SGD/Adam) stagnates. L-BFGS captures curvature information to achieve superlinear convergence to machine precision, serving as the refinement engine in two-phase optimization.

**Appears in:** [[w2d5]], [[week-2-report]], [[w3d5]], [[week-3-report]], [[w4d3]], [[week-4-report]]

## Related links
- [[two-phase-optimization]]
- [[backpropagation]]
- [[loss-formulation-pinn]]
- [[gradient-stiffness]]
