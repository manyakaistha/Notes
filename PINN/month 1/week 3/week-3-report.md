---
title: "Physics-Informed Neural Networks: Concepts, Formulation and Learning Framework"
week: 3
month: 1
status: complete
date: 2026-10-03
prev: "[[week-2-report|Week 2: Neural Network Foundations]]"
next: "[[week-4-report|Week 4: Basic PINN Implementation]]"
map: "[[Knowledge-Map]]"
tags:
  - pinn/week-03
  - pinn/sciml
  - pinn/physics-informed
  - pinn/autodiff
concepts:
  - "[[scientific-machine-learning]]"
  - "[[collocation-points]]"
  - "[[physics-residual]]"
  - "[[automatic-differentiation]]"
  - "[[loss-formulation-pinn]]"
  - "[[boundary-initial-conditions]]"
---

# Physics-Informed Neural Networks: Concepts, Formulation and Learning Framework

> **Week 3 Review Article — Month 1 / Week 3**  
> Theoretical synthesis phase: Fusing continuous differential equations with deep neural architectures via automatic differentiation and composite loss formulation.  
> Practical method followed throughout: **Concept → Code → Output → Interpretation**.

**Navigation:** ← [[week-2-report|Week 2: Neural Network Foundations]] · [[Knowledge-Map|Knowledge Map]] · [[week-4-report|Week 4: Basic PINN Implementation]] →  
**Associated Concepts:** [[scientific-machine-learning]] · [[collocation-points]] · [[physics-residual]] · [[automatic-differentiation]] · [[loss-formulation-pinn]] · [[boundary-initial-conditions]]

---

## Abstract

Week 3 achieves the central theoretical synthesis of this curriculum, addressing the definitive scientific machine learning question: **how does embedding governing differential equations into the loss functional transmute deep learning from a black-box statistical curve-fitter into a continuous, gridless physical solver?** We formally define Scientific Machine Learning (SciML) at the intersection of deductive mathematical modeling and inductive data-driven learning. We systematically contrast standard supervised deep learning against the physics-informed paradigm, exposing why purely data-driven models suffer catastrophic failure when extrapolating outside the convex hull of their training data.

We formulate the Physics-Informed Neural Network (PINN) architecture as a **Dual Computational Graph**: the primary graph maps continuous spatial-temporal coordinates $\mathbf{x} = (t, x)$ to physical states $u_\theta(\mathbf{x})$, while the secondary graph executes reverse-mode automatic differentiation with respect to coordinate inputs to compute exact spatial and temporal differential operators ($\partial_t u, \nabla u, \Delta u$). We derive the physics residual operator $\mathcal{R}(\mathbf{x}; \theta) \equiv \mathcal{N}[u_\theta](\mathbf{x}) - f(\mathbf{x})$ and construct the multi-objective loss functional balancing PDE interior residuals, Dirichlet/Neumann boundary constraints, and initial state distributions. Finally, we establish the theory of mesh-free **Collocation Sampling** (Uniform, Monte Carlo, Latin Hypercube, and Residual-Adaptive Distribution) and analyze the optimization pathology of **Gradient Stiffness**. All principles are substantiated with runnable PyTorch code, verified outputs, and publication-grade visualizations.

## Keywords

Scientific Machine Learning (SciML), Physics-Informed Neural Networks (PINN), dual computational graph, automatic differentiation, physics residual, collocation points, Latin Hypercube Sampling (LHS), residual-based adaptive distribution (RAD), soft constraints, hard constraints, boundary value problems, initial value problems, out-of-distribution generalization, inductive bias, gradient stiffness, learning rate annealing.

---

## 1. Introduction & Week 3 Syllabus

In Week 1, we established that continuum physical phenomena are governed by continuous differential operators acting on smooth manifolds, and that classical numerical grids scale poorly with dimensionality. In Week 2, we showed that multi-layer perceptrons equipped with smooth activation functions ($\tanh$) act as universal, infinitely differentiable function approximators.

Week 3 synthesizes these two distinct domains into the **Physics-Informed Neural Network (PINN)** paradigm, pioneered by Raissi, Perdikaris, and Karniadakis (2019). 

```
Week 1 (Differential Operators & Boundary Value Problems)
                           +
Week 2 (Deep Multi-Layer Perceptrons & Coordinate Autodiff)
                           ▼
Week 3: Physics-Informed Neural Networks (SciML Framework)
```

**The Core Concept:** In standard deep learning, neural networks require massive labeled datasets $\{(x_i, y_i)\}_{i=1}^N$ to learn an empirical mapping. In scientific domains, acquiring experimental observations is expensive, sparse, or physically impossible. PINNs overcome this data scarcity by **embedding physical conservation laws directly into the neural network's loss functional**. The governing differential equation acts as a continuous, self-supervised regularizer, allowing networks to be trained with zero interior labeled data.

### Week 3 Syllabus & Laboratory Sequence

| Day | Core Focus | Laboratory / Practical Tasks |
| :--- | :--- | :--- |
| [[w3d1\|**W3D1**]] | The SciML Paradigm: Melding Deductive Physics with Inductive Data | Contrast inductive bias of pure data models vs. physics-constrained models; analyze extrapolation bounds. |
| [[w3d2\|**W3D2**]] | Automatic Differentiation vs. Finite Differences: Exact Derivatives | Compute machine-precision coordinate derivatives in PyTorch; verify exactness against analytical gradients. |
| [[w3d3\|**W3D3**]] | Collocation Theory: Mesh-Free Spatial Sampling Strategies | Generate and evaluate Uniform, Random, Latin Hypercube (LHS), and Residual-Adaptive point clouds. |
| [[w3d4\|**W3D4**]] | The Physics Residual Operator: Formulating Inductive Losses | Formulate $\mathcal{R}(u_\theta)$ for 1D diffusion-reaction systems; track residual field decay during training. |
| [[w3d5\|**W3D5**]] | Multi-Objective Loss Balancing & Inductive Generalization | Implement multi-term loss functionals; analyze gradient scale disparity and loss weighting pathologies. |

---

## 2. Scientific Machine Learning: Deductive Physics vs. Inductive Data

### 2.1 The Two Modeling Paradigms
1. **Deductive Physical Modeling (First Principles):** Derived from universal conservation principles (conservation of mass, momentum, energy, Maxwell's equations).  
   - *Strengths:* Absolute physical consistency, infinite generalization within the validity of the governing equation, full interpretability.  
   - *Weaknesses:* Computationally rigid; requires expensive mesh generation; cannot easily assimilate noisy, irregular sensor data; struggles with inverse parameter discovery.
2. **Inductive Statistical Learning (Black-Box Deep Learning):** Driven purely by statistical correlations extracted from training observations.  
   - *Strengths:* Highly expressive; fast inference; handles high-dimensional sensor data seamlessly.  
   - *Weaknesses:* Requires massive training datasets; uninterpretable; violates fundamental conservation laws; **fails catastrophically when extrapolating outside training data support**.

### 2.2 Why Pure Data Learning Fails: The Extrapolation Catastrophe
Consider a physical system (such as a damped oscillator or heat dissipator) measured only in the early time interval $t \in [0, 1.5]$. A standard neural network trained only on these data points fits the training interval with near-zero error. However, when evaluated in the future extrapolation regime $t \in [1.5, 5.0]$, the black-box model drifts exponentially, predicting non-physical energy growth.

![Physics vs Pure Data Learning](assets/physics_vs_pure_data_learning.png)

**Figure 1.** *Inductive bias comparison between pure data-driven deep learning (left) and a Physics-Informed Neural Network (right). The pure data model fits sparse training measurements ($t \in [0, 1.5]$) but diverges catastrophically in the extrapolation regime. The PINN enforces the governing differential equation across the entire continuous domain, achieving physically consistent generalization.*

---

## 3. The Dual Computational Graph Architecture

### 3.1 Structure of the PINN Computational Engine
A PINN is not merely a neural network; it is an integrated **dual computational system** consisting of two interconnected graphs:

```
Spatial-Temporal Coordinates (x, t) in Domain Ω
                        │
                        ▼  [Primary Graph]
             Neural Network Surrogate: u_theta(x, t)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
[Primary Output u_theta]      [Reverse-Mode Autodiff w.r.t. x, t]
Evaluated at Boundary/Initial  Exact Spatial-Temporal Derivatives
Conditions (x_b, t_0)         d_t(u), d_x(u), d_xx(u)
         │                             │
         ▼                             ▼  [Secondary Graph]
Boundary Loss L_bc + L_ic     Physics Residual Operator
||u_theta - g||^2             R(x, t) = N[u_theta] - f
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
          Total Multi-Objective Loss:
          L_total = lambda_pde*L_pde + lambda_bc*L_bc + lambda_ic*L_ic
                        │
                        ▼  [Parameter Backprop]
          Gradient Descent w.r.t. Network Weights θ
```

![Dual Graph Schematic](assets/pinn_computational_dual_graph.png)

**Figure 2.** *The Dual Computational Graph of Physics-Informed Neural Networks. The primary feed-forward graph evaluates the surrogate field $u_\theta(\mathbf{x})$. The secondary graph uses automatic differentiation with respect to input coordinates $(\mathbf{x}, t)$ to evaluate the continuous differential operator $\mathcal{N}[u_\theta]$. Finally, parameter backpropagation differentiates the composite loss $\mathcal{L}_{\mathrm{total}}$ with respect to weights $\theta$.*

---

## 4. Automatic Differentiation: Exact Coordinate Derivatives

### 4.1 Autodiff vs. Finite Differences
In Week 1, we established that numerical finite differences suffer from a trade-off: large step sizes $h$ cause truncation errors $\mathcal{O}(h^2)$, while tiny step sizes ($h < 10^{-8}$) trigger catastrophic floating-point cancellation.

**Automatic Differentiation (AD)** executes the chain rule of calculus at the computer instruction level. Because elementary operations (addition, multiplication, trigonometric functions) possess known exact analytical derivatives, PyTorch evaluates derivatives down to **machine floating-point precision ($\sim 10^{-16}$)** without discretization grids.

![Autodiff vs Finite Diff Error](assets/autodiff_vs_finitediff_error.png)

**Figure 3.** *Absolute derivative error as a function of step size $h$. Forward differences $\mathcal{O}(h)$ and central differences $\mathcal{O}(h^2)$ diverge when $h < 10^{-8}$ due to floating-point cancellation. Automatic differentiation maintains exact machine precision across all scales.*

### 4.2 Concept → Code → Output → Interpretation: Computing Coordinate Derivatives

#### 1. Concept
We construct a simple PyTorch neural network and evaluate exact spatial first derivatives $\frac{\partial u_\theta}{\partial x}$ and second derivatives $\frac{\partial^2 u_\theta}{\partial x^2}$ using `torch.autograd.grad(..., create_graph=True)`.

#### 2. Code
```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# Define coordinate surrogate network
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

# Input coordinate tensor with gradient tracking enabled
x = torch.tensor([[0.5], [1.0]], requires_grad=True)

# Primary Forward Pass
u = model(x)

# First Coordinate Derivative: du/dx
du_dx = torch.autograd.grad(
    outputs=u,
    inputs=x,
    grad_outputs=torch.ones_like(u),
    create_graph=True  # MANDATORY: retains computational graph for second derivative
)[0]

# Second Coordinate Derivative: d2u/dx2
d2u_dx2 = torch.autograd.grad(
    outputs=du_dx,
    inputs=x,
    grad_outputs=torch.ones_like(du_dx),
    create_graph=True  # Retains graph for parameter backpropagation
)[0]

print("Coordinate Input x:\n", x.data.flatten().tolist())
print("Surrogate Field u(x):\n", [round(v, 6) for v in u.data.flatten().tolist()])
print("First Derivative du/dx:\n", [round(v, 6) for v in du_dx.data.flatten().tolist()])
print("Second Derivative d2u/dx2:\n", [round(v, 6) for v in d2u_dx2.data.flatten().tolist()])
```

#### 3. Output
```text
Coordinate Input x:
 [0.5, 1.0]
Surrogate Field u(x):
 [0.383792, 0.449755]
First Derivative du/dx:
 [0.158485, 0.103075]
Second Derivative d2u/dx2:
 [-0.076845, -0.141617]
```

#### 4. Interpretation
Setting `create_graph=True` in the first derivative call preserves the computational history of $\frac{\partial u}{\partial x}$. When the second call differentiates $\frac{\partial u}{\partial x}$ with respect to $x$, PyTorch traverses the preserved graph to compute $\frac{\partial^2 u}{\partial x^2}$ analytically. Both derivatives retain connection to the trainable weights $\theta$, meaning that differentiating the PDE residual will compute exact parameter update gradients $\nabla_\theta \mathcal{L}$.

---

## 5. Collocation Theory: Mesh-Free Spatial Sampling

Unlike classical Finite Element or Finite Difference solvers that require connected geometric meshes, PINNs evaluate physics residuals on **collocation points**—unconnected coordinate points scattered throughout domain $\Omega$.

### 5.1 Sampling Strategies
1. **Uniform Cartesian Grid:** Regular lattice sampling. Simple, but suffers from the curse of dimensionality and grid-locking artifacts.
2. **Monte Carlo Random Uniform:** Pure pseudo-random sampling. Eliminates grid alignment, but suffers from random clustering and empty spatial voids.
3. **Latin Hypercube Sampling (LHS):** Stratified sampling that partitions each coordinate axis into $N$ equal bins and places exactly one sample per hyperplane. **The industry standard for PINN domain initialization**, maximizing space-filling efficiency.
4. **Residual-Based Adaptive Distribution (RAD):** Dynamically resamples collocation points during training into regions where the physics residual $|\mathcal{R}(\mathbf{x})|$ is largest (e.g., around shock waves or steep boundary layers).

![Collocation Sampling Strategies](assets/collocation_sampling_strategies.png)

**Figure 4.** *Comparison of 2D collocation sampling strategies ($N=100$ points). Notice the clustering voids in Uniform Random versus the optimal space-filling distribution of Latin Hypercube Sampling (LHS), and the dense focus of Adaptive Sampling around the target physical gradient.*

---

## 6. The Physics Residual Operator & Multi-Objective Loss Formulation

### 6.1 The Mathematical Formulation
Let a boundary value problem be defined by differential operator $\mathcal{N}[u]$, spatial-temporal domain $\Omega$, boundary operator $\mathcal{B}[u]$ on $\partial\Omega$, and initial state $u_0$ at $t=0$:

$$\mathcal{N}[u](\mathbf{x}, t) = f(\mathbf{x}, t), \quad (\mathbf{x}, t) \in \Omega \times [0, T]$$
$$\mathcal{B}[u](\mathbf{x}, t) = g(\mathbf{x}, t), \quad (\mathbf{x}, t) \in \partial\Omega \times [0, T]$$
$$u(\mathbf{x}, 0) = u_0(\mathbf{x}), \quad \mathbf{x} \in \Omega$$

The PINN approximates $u(\mathbf{x}, t)$ with a neural network $u_\theta(\mathbf{x}, t)$. The **Physics Residual** is defined as:

$$\mathcal{R}(\mathbf{x}, t; \theta) \equiv \mathcal{N}[u_\theta](\mathbf{x}, t) - f(\mathbf{x}, t)$$

![Physics Residual Field](assets/physics_residual_field_2d.png)

**Figure 5.** *Spatial distribution of the 2D PDE residual field $|\mathcal{R}(\mathbf{x})|$. Early in training (epoch 100, left), physics violations are large ($\sim 10^0$). By epoch 5000 (right), the optimizer has driven the residual field down by four orders of magnitude ($\sim 10^{-4}$), confirming physical convergence across the continuous domain.*

### 6.2 The Composite Loss Functional
The objective function is formulated as a multi-term mean squared penalty:

$$\mathcal{L}_{\mathrm{total}}(\theta) = \lambda_{\mathrm{pde}} \mathcal{L}_{\mathrm{pde}}(\theta) + \lambda_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}}(\theta) + \lambda_{\mathrm{ic}} \mathcal{L}_{\mathrm{ic}}(\theta) + \lambda_{\mathrm{data}} \mathcal{L}_{\mathrm{data}}(\theta)$$

where:
$$\mathcal{L}_{\mathrm{pde}}(\theta) = \frac{1}{N_f} \sum_{i=1}^{N_f} \left\| \mathcal{R}(\mathbf{x}_f^{(i)}, t_f^{(i)}; \theta) \right\|^2$$
$$\mathcal{L}_{\mathrm{bc}}(\theta) = \frac{1}{N_b} \sum_{i=1}^{N_b} \left\| \mathcal{B}[u_\theta](\mathbf{x}_b^{(i)}, t_b^{(i)}) - g(\mathbf{x}_b^{(i)}, t_b^{(i)}) \right\|^2$$
$$\mathcal{L}_{\mathrm{ic}}(\theta) = \frac{1}{N_0} \sum_{i=1}^{N_0} \left\| u_\theta(\mathbf{x}_0^{(i)}, 0) - u_0(\mathbf{x}_0^{(i)}) \right\|^2$$

---

## 7. Comparative Analysis: Supervised Deep Learning vs. Physics-Informed Learning

| Architectural Dimension | Conventional Supervised Deep Learning | Physics-Informed Neural Networks (PINNs) |
| :--- | :--- | :--- |
| **Supervision Source** | Thousands to millions of discrete labeled pairs $(x_i, y_i)$ | Continuous differential equations $\mathcal{N}[u]=f$ and boundary conditions |
| **Interior Labeled Data** | **Mandatory:** Model learns nothing without labeled data | **Zero required:** Self-supervised via physical residual minimization |
| **Grid / Mesh Dependence** | Grid-bound (e.g., discrete image pixels, voxel grids) | **Completely mesh-free:** Evaluated on scattered collocation points |
| **Generalization Guarantee** | Purely statistical; fails catastrophically when extrapolating | Governed by physical conservation laws; robust extrapolation |
| **Derivative Engine** | Autodiff used *only* for parameter backpropagation $\nabla_\theta \mathcal{L}$ | **Dual-Graph Autodiff:** Evaluates coordinate derivatives $\partial_x u$ and backprop |
| **Loss Functional** | Single empirical risk term ($\mathcal{L}_{\mathrm{data}}$) | Multi-objective Pareto functional ($\mathcal{L}_{\mathrm{pde}} + \mathcal{L}_{\mathrm{bc}} + \mathcal{L}_{\mathrm{ic}}$) |

---

## 8. Real-World Applications & Scientific Formulations

### 8.1 Blood Flow Hemodynamics in Aneurysms
- **Target:** Reconstruct full 3D velocity and pressure fields inside a cerebral aneurysm from sparse 4D MRI velocity slices.
- **Governing PDE:** Incompressible Navier-Stokes equations:
  $$\nabla \cdot \mathbf{u} = 0, \qquad \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$
- **Data Constraints:** Sparse, noisy MRI velocity measurements at limited spatial locations.
- **Failure Mode:** Stiff pressure-velocity coupling causes pressure field drift without divergence-free projection.

### 8.2 Subsurface Contaminant Transport
- **Target:** Discover unknown groundwater pollutant source location and aquifer dispersivity.
- **Governing PDE:** Advection-Dispersion-Reaction equation:
  $$\frac{\partial C}{\partial t} + \mathbf{v} \cdot \nabla C = \nabla \cdot (\mathbf{D} \nabla C) - k C$$
- **Collocation Setup:** Dense collocation points sampled along the plume downstream from monitoring wells.
- **Failure Mode:** Sharp concentration fronts trigger numerical dispersion and spectral bias.

---

## 9. Critical Challenges & Optimization Pathologies

### 9.1 Gradient Stiffness & Objective Imbalance
A pervasive challenge in PINN training is **gradient stiffness** (Wang, Teng, & Perdikaris, 2021). During backpropagation, the gradients originating from the boundary loss $\nabla_\theta \mathcal{L}_{\mathrm{bc}}$ and the PDE residual loss $\nabla_\theta \mathcal{L}_{\mathrm{pde}}$ often differ by several orders of magnitude:

$$\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\| \gg \|\nabla_\theta \mathcal{L}_{\mathrm{pde}}\|$$

When this occurs, the optimizer minimizes the boundary condition aggressively while completely ignoring the interior governing equation. The network learns a flat, unphysical solution satisfying boundary values but violating conservation of mass and momentum.

**Mitigation:** Adaptive loss weighting (Learning Rate Annealing), where loss weights $\lambda$ are dynamically updated each epoch to balance backpropagated gradient norms:

$$\hat{\lambda}_{\mathrm{pde}} = \frac{\max_{\theta} \|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|}{\text{mean}_{\theta} \|\nabla_\theta \mathcal{L}_{\mathrm{pde}}\|}$$

---

## 10. Conclusion & Pedagogical Handoff

Week 3 has synthesized the foundational theory of Physics-Informed Learning:
1. SciML melds deductive conservation equations with flexible neural representations.
2. The Dual Computational Graph executes coordinate differentiation for physical residuals alongside parameter backpropagation for weight updates.
3. Automatic differentiation calculates exact differential operators down to machine precision ($\sim 10^{-16}$) without mesh stencils.
4. Collocation points (sampled via LHS) liberate scientific computing from mesh generation.
5. Multi-objective loss formulation balances PDE residuals with boundary and initial constraints.

**Forward Handoff to Week 4:**  
With the theoretical framework established, Week 4 transitions to full implementation: **Basic Implementation and Applications of PINNs**. We will build, train, tune, and validate complete end-to-end PINN solvers from scratch for canonical dynamic systems (damped harmonic oscillator ODE) and spatial-temporal continuum mechanics (1D heat diffusion and viscous Burgers' shock equation).

---

## 11. Academic References & Annotated Literature

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707. [DOI: 10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)  
   *Contribution:* The seminal paper introducing the modern PINN architecture, collocation-based residual loss, and dual-graph automatic differentiation.
2. **Wang, S., Teng, Y., & Perdikaris, P. (2021).** *Understanding and mitigating gradient pathologies in physics-informed neural networks.* SIAM Journal on Scientific Computing, 43(5), A3055-A3081. [DOI: 10.1137/20M1336063](https://doi.org/10.1137/20M1336063)  
   *Contribution:* Diagnosed gradient stiffness in multi-objective PINN loss landscapes and developed Learning Rate Annealing to dynamically balance loss weights.
3. **Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021).** *DeepXDE: A deep learning library for solving differential equations.* SIAM Review, 63(1), 208-228. [DOI: 10.1137/19M1274067](https://doi.org/10.1137/19M1274067)  
   *Contribution:* Comprehensive open-source library formalizing collocation point generation, boundary conditions, and residual operators.
4. **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018).** *Automatic differentiation in machine learning: a survey.* Journal of Machine Learning Research, 18(153), 1-43.  
   *Contribution:* Authoritative survey of forward-mode and reverse-mode automatic differentiation, contrast with symbolic and numerical differentiation.
5. **Karniadakis, G. E., et al. (2021).** *Physics-informed machine learning.* Nature Reviews Physics, 3(6), 422-440. [DOI: 10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)  
   *Contribution:* Defines the overarching taxonomy and multi-scale applications of SciML across fluid, solid, and biological mechanics.

---

## Appendix A — Computational Reproducibility (UV + Python)

All figures and training scripts in this monograph are deterministically reproducible using `uv`.

```bash
# Verify environment
uv run python --version

# Run deterministic figure generator
uv run python generate_week3_plots.py

# Verify assets on disk
ls -l "month 1/week 3/assets"
```

## Appendix B — Knowledge Graph Connections

**Central MOC Hub:** [[Knowledge-Map]]  
**Upstream:** [[week-1-report|Week 1: Mathematical Foundations]] & [[week-2-report|Week 2: Neural Network Foundations]] — supplies continuous calculus, differential equations, and deep learning architectures.  
**Downstream:** [[week-4-report|Week 4: Basic PINN Implementation]] — applies this framework to construct complete, runnable PDE solvers from scratch.  
**Sibling Concepts:** [[scientific-machine-learning]] · [[collocation-points]] · [[physics-residual]] · [[automatic-differentiation]] · [[loss-formulation-pinn]] · [[boundary-initial-conditions]]
