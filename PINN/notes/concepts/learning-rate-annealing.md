---
title: Learning Rate Annealing
slug: learning-rate-annealing
tags: [pinn/concept, pinn/optimization, pinn/loss-balancing]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Learning Rate Annealing

Learning Rate Annealing (Wang et al., 2021) is an adaptive multi-objective gradient balancing algorithm designed to eliminate gradient stiffness and pathological scale disparities in physics-informed neural networks. Rather than maintaining static penalty multipliers $\lambda_i$, Learning Rate Annealing dynamically tracks the maximum and mean Frobenius norms of the parameter gradients produced by each individual loss functional during training: $\hat{\lambda}_k = \frac{\max_{\theta} \|\nabla_\theta \mathcal{L}_{\mathrm{pde}}\|}{\overline{\|\nabla_\theta \mathcal{L}_k\|}}$. The operational weights are then updated using an exponential moving average: $\lambda_k^{(t)} = (1 - \alpha) \lambda_k^{(t-1)} + \alpha \hat{\lambda}_k$, where $\alpha \in [0.1, 0.9]$. By dynamically scaling boundary and data loss gradients to match the maximum magnitude of the interior physics residual gradient, Learning Rate Annealing enforces equitable parameter updates across both primary domain physics and boundary constraints, unlocking robust convergence on stiff, advection-dominated problems.

**Why it matters here:** Learning Rate Annealing provides the standard algorithmic remedy for multi-objective optimization collapse in stiff PDE systems.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d4|W3D4]] · [[w4d3|W4D3]]

## Related
- [[gradient-stiffness]]
- [[loss-formulation-pinn]]
- [[two-phase-optimization]]
