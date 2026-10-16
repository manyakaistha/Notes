---
title: Physics-Informed Neural Networks Knowledge Map
tags: [pinn/moc]
---

# Physics-Informed Neural Networks (PINN) — Knowledge Map

Welcome to the central Map of Content (MOC) and navigational hub for the intensive curriculum on **Physics-Informed Neural Networks (PINNs) and Scientific Machine Learning (SciML)**.

---

## The 4-Week Curriculum Spine

| Week | Report | Core Question | Focus Areas | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | [[week-1-report\|Week 1: Mathematical Foundations]] | How do continuous calculus, differential equations, and discretization errors lay the groundwork for physics constraints? | Functions, Multivariable Calculus, ODEs, PDEs, Boundary Conditions, Numerical Discretization, Truncation Residuals | **complete** |
| **2** | [[week-2-report\|Week 2: Neural Network Foundations]] | How do multi-layer perceptrons, forward-backward propagation, and gradient-based optimizers approximate continuous functions? | Perceptrons, Smooth Activations, Backpropagation, Computational Graphs, Loss Landscapes, Optimizer Dynamics, Spectral Bias | **complete** |
| **3** | [[week-3-report\|Week 3: Concepts, Formulation & Learning Framework]] | How does embedding governing differential equations into the loss functional transmute deep learning into a gridless scientific solver? | SciML Paradigm, Mesh-free Collocation, Dual-Graph Autodiff, Physics Residuals, Soft vs. Hard Constraints, Gradient Stiffness, Learning Rate Annealing | **complete** |
| **4** | [[week-4-report\|Week 4: Basic Implementation & Applications]] | How do we implement, train, balance, and validate a full PINN solver from scratch on canonical ODEs and PDEs? | 10-Stage Pipeline, Damped Oscillator IVP, 1D Heat / Burgers PDE, Chebfun Spectral Grounding, Two-Phase Optimization, Inverse Parameter Discovery | **complete** |

---

## Daily Working Notes Index

### Week 1 Daily Notes — Mathematical Foundations
- [[w1d1\|W1D1]]: Functions, Variables, Domain Mappings & Coordinate Graphs
- [[w1d2\|W1D2]]: Differential Calculus — Limits, Derivatives, Chain Rule & Gradient Vector Fields
- [[w1d3\|W1D3]]: Ordinary Differential Equations (ODEs) — Order, Linearity & Initial Value Problems
- [[w1d4\|W1D4]]: Partial Differential Equations (PDEs) — Elliptic, Parabolic, Hyperbolic & Boundary Operators
- [[w1d5\|W1D5]]: Numerical Fundamentals — Discretization, Finite Differences & Truncation Residuals

### Week 2 Daily Notes — Neural Network Foundations
- [[w2d1\|W2D1]]: The Artificial Neuron, Perceptron Geometry, Weights, Biases & Affine Projections
- [[w2d2\|W2D2]]: Non-linear Activation Functions — The Imperative for Smooth $C^2$ Differentiability in SciML
- [[w2d3\|W2D3]]: Multi-Layer Perceptrons, Forward Propagation & Computational Graph Topologies
- [[w2d4\|W2D4]]: Loss Functionals (MSE vs. MAE), Spectral Bias & the Frequency Principle
- [[w2d5\|W2D5]]: Backpropagation, Exact Chain Rule Autograd & First/Second-Order Optimization Dynamics

### Week 3 Daily Notes — Physics-Informed Learning Framework
- [[w3d1\|W3D1]]: The Scientific Machine Learning Paradigm, Inductive Bias & Extrapolation Catastrophe
- [[w3d2\|W3D2]]: Automatic Differentiation, Vector-Jacobian Products & The Dual Computational Graph
- [[w3d3\|W3D3]]: Spatiotemporal Collocation, Low-Discrepancy Sampling & Continuous Residuals
- [[w3d4\|W3D4]]: Loss Landscape Geometry, Soft vs. Hard Boundary Constraints & Gradient Stiffness
- [[w3d5\|W3D5]]: Two-Phase Optimization Dynamics, Inverse Parameter Discovery & SciML Taxonomy

### Week 4 Daily Notes — Basic PINN Implementation & Applications
- [[w4d1\|W4D1]]: The 10-Stage PINN Implementation Lifecycle, Data Grounding & Ingestion
- [[w4d2\|W4D2]]: Dynamic Initial Value Problems: The Damped Harmonic Oscillator & Phase Space Tracking
- [[w4d3\|W4D3]]: Two-Phase Optimization Dynamics: Adam Exploration to L-BFGS Quasi-Newton Descent
- [[w4d4\|W4D4]]: Continuum Mechanics PDE Solvers: 1D Transient Heat & Viscous Burgers' Shock
- [[w4d5\|W4D5]]: System Identification, Sensor Noise Robustness & Month 1 Cumulative Mastery Examination

---

## Concept Spine & Taxonomy

The knowledge graph's conceptual nodes are categorized across four pillars:

- **1. Mathematical Foundations & Differential Systems**
  - [[ordinary-differential-equation]] · [[partial-differential-equation]] · [[boundary-initial-conditions]] · [[discretization-error]] · [[differential-residual]]
- **2. Neural Network Machinery & Computational Calculus**
  - [[artificial-neuron]] · [[activation-functions]] · [[smooth-differentiability]] · [[forward-propagation]] · [[backpropagation]] · [[automatic-differentiation]] · [[spectral-bias]]
- **3. Physics-Informed Formulation & Graph Architecture**
  - [[scientific-machine-learning]] · [[collocation-points]] · [[physics-residual]] · [[dual-computational-graph]] · [[latin-hypercube-sampling]]
- **4. Optimization Dynamics & Parameter Inversion**
  - [[loss-functionals]] · [[loss-formulation-pinn]] · [[gradient-stiffness]] · [[learning-rate-annealing]] · [[quasi-newton-optimization]] · [[two-phase-optimization]] · [[inverse-parameter-discovery]]

---

## Central Governing Paradigm

```
Physical Continuous Domain (x, t) in Omega
                │
                ▼
      Collocation Sampling (LHS / Uniform / RAD)
                │
                ▼
   Neural Network Surrogate: u_theta(x, t)
                │
                ├────────────────────────────────────────┐
                ▼                                        ▼
   Automatic Differentiation (w.r.t. x, t)       Boundary / Initial Points
   [d_t(u), d_x(u), d_xx(u), ...]                u_theta(0, x), u_theta(t, dOmega)
                │                                        │
                ▼                                        ▼
   Physics Residual Operator                     Boundary Loss Functional
   R(x, t) = N[u_theta] - f                      L_bc + L_ic
                │                                        │
                └───────────────────┬────────────────────┘
                                    │
                                    ▼
                      Total Multi-Objective Loss Functional:
                      L_total = lambda_pde*L_pde + lambda_bc*L_bc + lambda_ic*L_ic
                                    │
                                    ▼
                      Backpropagation w.r.t. theta & lambda
                      [Adam Exploration ──> L-BFGS Quasi-Newton Convergence]
                                    │
                                    ▼
                       Physical Continuous Solution u_theta & Discovered Parameters lambda*
```

Mathematically formalized as:

$$\mathbf{x} \in \Omega \xrightarrow{\quad u_\theta(\mathbf{x})\quad} u \xrightarrow{\quad\text{Autodiff}\quad} \left\{ \frac{\partial u}{\partial t}, \nabla u, \Delta u, \dots \right\} \xrightarrow{\quad\mathcal{N}[u; \boldsymbol{\lambda}] - f\quad} \mathcal{R}_{\text{physics}}(\mathbf{x}) \longrightarrow \mathcal{L}_{\text{total}}(\theta, \boldsymbol{\lambda})$$

---

## Continuity & Inter-Week Handoffs

- **Week 1 $\to$ Week 2:** Week 1 establishes continuous calculus, differential operators, boundary conditions, and numerical residuals. Week 2 replaces discrete numerical approximations with multi-layer neural networks, providing the parameterized functional form $u_\theta(\mathbf{x})$ and smooth $C^\infty$ activation functions.
- **Week 2 $\to$ Week 3:** Week 2 provides the forward-backward autodiff engine and backpropagation mechanics. Week 3 embeds the differential operators from Week 1 into the loss function via coordinate automatic differentiation and the dual computational graph, establishing the mesh-free PINN learning paradigm, Latin Hypercube collocation, and dynamic Learning Rate Annealing.
- **Week 3 $\to$ Week 4:** Week 3 establishes the theoretical formulation, collocation theory, and loss landscape dynamics. Week 4 synthesizes this into an end-to-end, runnable, publication-verified computational codebase realizing the 10-stage lifecycle across canonical ODEs (damped harmonic oscillator), PDEs (1D heat equation, viscous Burgers' shock with Chebfun spectral validation), and multi-domain inverse parameter discovery under sensor noise.
