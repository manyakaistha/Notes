---
title: Loss Formulation in Physics-Informed Learning
slug: loss-formulation-pinn
tags: [pinn/concept, pinn/optimization, pinn/loss]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Loss Formulation in Physics-Informed Learning

Unlike conventional supervised machine learning, which minimizes empirical risk over discrete labeled input-target pairs ($\mathcal{L}_{\mathrm{data}} = \frac{1}{N}\sum \|y_i - \hat{y}_i\|^2$), a Physics-Informed Neural Network minimizes a composite multi-objective functional encoding physical conservation laws:

$$\mathcal{L}_{\mathrm{total}}(\theta) = \lambda_{\mathrm{pde}} \mathcal{L}_{\mathrm{pde}}(\theta) + \lambda_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}}(\theta) + \lambda_{\mathrm{ic}} \mathcal{L}_{\mathrm{ic}}(\theta) + \lambda_{\mathrm{data}} \mathcal{L}_{\mathrm{data}}(\theta)$$

where:
- $\mathcal{L}_{\mathrm{pde}}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} \|\mathcal{R}(t_i, x_i; \theta)\|^2$ penalizes non-zero physics residuals on interior collocation points.
- $\mathcal{L}_{\mathrm{bc}}(\theta) = \frac{1}{N_b} \sum_{i=1}^{N_b} \|\mathcal{B}[u_\theta](t_i, x_i) - g(t_i, x_i)\|^2$ enforces boundary constraints.
- $\mathcal{L}_{\mathrm{ic}}(\theta) = \frac{1}{N_0} \sum_{i=1}^{N_0} \|u_\theta(0, x_i) - u_0(x_i)\|^2$ enforces initial temporal conditions.
- $\lambda_{\mathrm{pde}}, \lambda_{\mathrm{bc}}, \lambda_{\mathrm{ic}}$ are penalty hyper-parameters balancing the competing objectives.

Improper weighting produces **gradient stiffness pathologies**, where the optimizer satisfies the easy objective (e.g., initial condition) while completely stalling on the stiff differential operator. Adaptive weighting techniques (such as Learning Rate Annealing or Soft-Attention PINNs) dynamically rescale $\lambda$ to equalize gradient backpropagation magnitudes.

**Why it matters here:** The composite loss functional is the heart of a PINN, translating a physical boundary value problem into an unconstrained non-linear optimization landscape.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d4|W3D4]] · [[w4d3|W4D3]]

## Related
- [[physics-residual]]
- [[boundary-initial-conditions]]
- [[collocation-points]]
