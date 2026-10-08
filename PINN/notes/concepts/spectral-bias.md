---
title: Spectral Bias and Multi-Scale Physics
slug: spectral-bias
tags: [pinn/concept, pinn/deep-learning, pinn/fourier]
introduced: "[[week-2-report|Week 2]]"
reinforced: ["[[week-3-report|Week 3]]", "[[week-4-report|Week 4]]"]
---

# Spectral Bias and Multi-Scale Physics

Spectral Bias (also known as the Frequency Principle) is an intrinsic property of standard coordinate-based Multi-Layer Perceptrons trained via gradient descent: neural networks learn low-frequency components of target functions first and exponentially fast, while high-frequency components converge at an exponentially slower rate. Analyzed through the lens of the Neural Tangent Kernel (NTK), the eigenvalues of the kernel decay rapidly with frequency $\lambda_k \sim k^{-2d}$. In scientific machine learning and PINNs, this manifests as a critical pathology: when solving differential equations with steep spatial gradients, boundary layers, shock waves (e.g., Burgers' or Navier-Stokes equations), or high-frequency acoustic waves, the network quickly captures the smooth background flow but completely stalls on sharp physical features. Overcoming spectral bias requires specialized architectures such as Fourier Feature Networks, SIREN sinusoidal activations, or multi-scale coordinate embeddings.

**Why it matters here:** Spectral bias explains why standard neural networks struggle to resolve shock fronts and turbulent eddies in fluid and continuum mechanics.

**Appears in:** [[week-2-report|Week 2]] · [[week-3-report|Week 3]] · [[week-4-report|Week 4]] · [[w2d4|W2D4]] · [[w4d5|W4D5]]

## Related
- [[activation-functions]]
- [[loss-formulation-pinn]]
- [[scientific-machine-learning]]
