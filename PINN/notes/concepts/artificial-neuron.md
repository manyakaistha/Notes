---
title: The Artificial Neuron & Perceptron Geometry
tags:
  - pinn/concept
  - deep-learning/architecture
---

# The Artificial Neuron & Perceptron Geometry

An **artificial neuron** is the elementary computational atom of deep learning, parameterized by a weight vector $\mathbf{w} \in \mathbb{R}^d$ and scalar bias $b \in \mathbb{R}$. It maps an input vector $\mathbf{x} \in \mathbb{R}^d$ to a scalar activation $a \in \mathbb{R}$ via an affine transformation followed by a scalar non-linear activation function $\sigma: \mathbb{R} \to \mathbb{R}$:

$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b, \quad a = \sigma(z)$$

Geometrically, the equation $\mathbf{w}^T \mathbf{x} + b = 0$ defines a $(d-1)$-dimensional hyperplane in the input coordinate space $\mathbb{R}^d$. The weight vector $\mathbf{w}$ represents the normal vector determining hyperplane orientation, while $b / \|\mathbf{w}\|$ dictates its signed orthogonal distance from the origin.

### Why it matters here
In scientific machine learning, inputs to artificial neurons are physical continuous space-time coordinates $(t, x, y, z)$. Individual neurons partition the physical continuum into oriented half-spaces, providing localized activation ridges that assemble into continuous physical field approximations.

**Appears in:** [[w2d1]], [[week-2-report]], [[w2d3]]

## Related links
- [[activation-functions]]
- [[forward-propagation]]
- [[backpropagation]]
- [[scientific-machine-learning]]
