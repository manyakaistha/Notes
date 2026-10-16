---
title: Dual Computational Graph
slug: dual-computational-graph
tags: [pinn/concept, pinn/autograd, pinn/architecture]
introduced: "[[week-3-report|Week 3]]"
reinforced: ["[[week-4-report|Week 4]]"]
---

# Dual Computational Graph

The Dual Computational Graph is the foundational operational mechanism of Physics-Informed Neural Networks that differentiates them from conventional deep learning pipelines. In standard neural networks, a single forward graph maps inputs to outputs, followed by a reverse pass that computes parameter gradients $\nabla_\theta \mathcal{L}$. In contrast, a PINN constructs two interlinked directed acyclic graphs (DAGs). The **Primary Field Graph** evaluates the surrogate field $u_\theta(\mathbf{x}, t)$ from spatial and temporal coordinates via feedforward linear-activation layers. The **Secondary Differential Graph** executes reverse-mode automatic differentiation on the output node with respect to the input coordinate leaf tensors (`create_graph=True`), yielding exact spatial and temporal partial derivatives ($\partial_t u_\theta, \partial_x u_\theta, \partial_{xx} u_\theta$) without numerical grid discretization. These differential operators assemble the physics residual node $\mathcal{R}(u_\theta)$, creating a compound graph through which the **Tertiary Parameter Backpropagation** traverses during loss minimization.

**Why it matters here:** The dual computational graph enables exact differential physics operators to become intrinsic components of the optimization landscape without numerical grid truncation.

**Appears in:** [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w3d2|W3D2]] · [[w4d1|W4D1]]

## Related
- [[automatic-differentiation]]
- [[physics-residual]]
- [[backpropagation]]
- [[scientific-machine-learning]]
