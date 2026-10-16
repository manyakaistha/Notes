---
title: Smooth Differentiability ($C^k$ Classes) in SciML
tags:
  - pinn/concept
  - mathematics/calculus
  - sciml/activations
---

# Smooth Differentiability ($C^k$ Classes) in SciML

A function $f: \mathbb{R} \to \mathbb{R}$ belongs to the **smoothness class** $C^k$ if its derivatives up to order $k$ exist and are continuous. In particular:
- $C^0$: Continuous functions with discontinuous first derivatives (e.g., $\text{ReLU}(z) = \max(0, z)$).
- $C^1$: Continuously differentiable functions whose second derivatives may be discontinuous.
- $C^2$: Functions with continuous second derivatives, required for evaluating spatial Laplacians $\nabla^2 u$.
- $C^\infty$: Smooth functions with continuous derivatives of all orders (e.g., $\tanh(z)$, $\sin(z)$, Gaussian RBFs).

### Why it matters here
The choice of activation function determines the smoothness class of the neural network surrogate $u_\theta(\mathbf{x})$. For second-order partial differential equations (such as the heat, wave, and Navier-Stokes equations), the PDE residual depends directly on second spatial derivatives $\nabla^2 u_\theta$. If a $C^0$ activation like ReLU is used, $\frac{d^2}{dz^2}\text{ReLU}(z) \equiv 0$ almost everywhere, causing higher-order physics residuals to evaluate to zero and completely halting gradient backpropagation.

**Appears in:** [[w2d2]], [[week-2-report]], [[w3d2]], [[week-3-report]]

## Related links
- [[activation-functions]]
- [[automatic-differentiation]]
- [[partial-differential-equation]]
- [[physics-residual]]
