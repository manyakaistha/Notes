---
title: Forward Propagation & Coordinate Tensor Flow
tags:
  - pinn/concept
  - deep-learning/architecture
---

# Forward Propagation & Coordinate Tensor Flow

**Forward propagation** is the deterministic sequential evaluation of a multi-layer neural network from input coordinates to output predictions. For an $L$-layer multi-layer perceptron (MLP) receiving a batch of spatial-temporal coordinates $\mathbf{X}^{(0)} \in \mathbb{R}^{B \times d_{\mathrm{in}}}$, forward propagation proceeds as:

$$\mathbf{Z}^{(l)} = \mathbf{X}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)}, \quad \mathbf{X}^{(l)} = \sigma\left(\mathbf{Z}^{(l)}\right), \quad l = 1, \dots, L-1$$
$$\mathbf{u}_\theta(\mathbf{X}^{(0)}) = \mathbf{X}^{(L-1)} \mathbf{W}^{(L)} + \mathbf{b}^{(L)}$$

where $\mathbf{W}^{(l)} \in \mathbb{R}^{d_{l-1} \times d_l}$ are weight matrices, $\mathbf{b}^{(l)} \in \mathbb{R}^{1 \times d_l}$ are broadcast bias vectors, and the final layer is linear.

### Why it matters here
In coordinate-based SciML, the input batch shape is typically `[Batch, Coordinate_Dim]` (e.g., `[N, 2]` for $(x, t)$) rather than image grids. Forward propagation constructs the Primary Computational Graph, which must be retained in memory with higher-order graph flags to enable coordinate autodiff.

**Appears in:** [[w2d3]], [[week-2-report]], [[w3d2]], [[week-3-report]]

## Related links
- [[artificial-neuron]]
- [[activation-functions]]
- [[backpropagation]]
- [[dual-computational-graph]]
