---
title: Loss Functionals in Scientific Regression
tags:
  - pinn/concept
  - optimization/loss
---

# Loss Functionals in Scientific Regression

A **loss functional** $\mathcal{L}(\theta)$ maps model predictions and ground-truth observations to a scalar penalty driving gradient-based parameter updates. In scientific regression and PDE solving, common formulations include:
- **Mean Squared Error (MSE / $L_2$ norm):**
  $$\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N} \sum_{i=1}^N \|u_\theta(\mathbf{x}_i) - y_i\|^2, \quad \frac{\partial \mathcal{L}}{\partial e_i} = 2 e_i$$
  Penalizes large errors quadratically, ensuring strict adherence to physical conservation laws.
- **Mean Absolute Error (MAE / $L_1$ norm):**
  $$\mathcal{L}_{\mathrm{MAE}} = \frac{1}{N} \sum_{i=1}^N |u_\theta(\mathbf{x}_i) - y_i|, \quad \frac{\partial \mathcal{L}}{\partial e_i} = \text{sign}(e_i)$$
  Provides constant gradient magnitudes, conferring robust resistance to sensor outliers.
- **Huber Loss:** Smoothly interpolates between quadratic behavior for small errors and linear behavior for large errors.

### Why it matters here
In PINNs, composite loss functionals combine data fitting terms with differential PDE residuals: $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{pde}} + \lambda_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}} + \lambda_d \mathcal{L}_{\mathrm{data}}$. The mathematical properties of the loss functional dictate error gradient scaling and optimization conditioning.

**Appears in:** [[w2d4]], [[week-2-report]], [[w3d4]], [[week-3-report]], [[w4d3]], [[week-4-report]]

## Related links
- [[loss-formulation-pinn]]
- [[gradient-stiffness]]
- [[learning-rate-annealing]]
- [[two-phase-optimization]]
