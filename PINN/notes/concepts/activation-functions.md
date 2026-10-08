---
title: Activation Functions in Scientific Machine Learning
slug: activation-functions
tags: [pinn/concept, pinn/deep-learning, pinn/architectures]
introduced: "[[week-2-report|Week 2]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Activation Functions in Scientific Machine Learning

In standard computer vision and natural language processing, the Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$) is the standard activation function due to its computational simplicity and constant gradient for positive inputs. However, in Scientific Machine Learning and Physics-Informed Neural Networks, **ReLU is completely forbidden** for solving second-order or higher differential equations because its second derivative is identically zero everywhere ($d^2\text{ReLU}/dz^2 \equiv 0$) and its first derivative is discontinuous. PINNs require continuously differentiable ($C^k, k \ge 2$) non-linearities:
1. **Hyperbolic Tangent ($\tanh$):** The gold standard for PINNs. Smooth, symmetric, with non-zero higher-order derivatives that decay smoothly.
2. **Sinusoidal Activations (SIREN):** Periodic activations ($\sin(\omega_0 z)$) whose derivatives are shifted sines and cosines, exceptionally suited for capturing high-frequency physical oscillations and wave phenomena without gradient attenuation.
3. **Gaussian / Radial Basis:** Smooth and localized, ideal for boundary layer and compact support problems.

**Why it matters here:** The choice of activation function directly dictates whether higher-order differential operators can exist and propagate non-zero backpropagation gradients.

**Appears in:** [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w2d2|W2D2]] · [[w3d2|W3D2]] · [[w4d1|W4D1]]

## Related
- [[automatic-differentiation]]
- [[backpropagation]]
- [[spectral-bias]]
