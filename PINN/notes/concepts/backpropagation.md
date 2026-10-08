---
title: Backpropagation and Computational Graphs
slug: backpropagation
tags: [pinn/concept, pinn/deep-learning, pinn/optimization]
introduced: "[[week-2-report|Week 2]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Backpropagation and Computational Graphs

Backpropagation is an algorithmic application of reverse-mode automatic differentiation applied to the computational directed acyclic graph (DAG) of a parameterized machine learning model. During the forward pass, intermediate activations are computed and stored. During the backward pass, the chain rule of calculus traverses backward from the scalar scalar loss functional $\mathcal{L}$ to compute exact vector gradients $\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}$ with respect to all network weights $\mathbf{W}$ and biases $\mathbf{b}$ in time proportional to a single forward evaluation. In PINNs, backpropagation exhibits a unique two-tier structure:
1. **The Physics Operator Pass:** Reverse-mode autodiff with respect to coordinate inputs $\mathbf{x} = (t, x)$ to construct the physical differential operator graph.
2. **The Optimization Pass:** Backpropagation with respect to network parameters $\theta$ through the physical residual DAG to update weights.

**Why it matters here:** Backpropagation bridges the differential operator residual back into parameter space, enabling gradient-based optimizers to drive the network toward physically consistent states.

**Appears in:** [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w2d3|W2D3]] · [[w3d2|W3D2]] · [[w4d2|W4D2]]

## Related
- [[automatic-differentiation]]
- [[loss-formulation-pinn]]
- [[activation-functions]]
