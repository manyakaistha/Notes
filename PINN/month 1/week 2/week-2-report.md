---
title: "Fundamentals of Neural Networks and Optimization for Scientific Machine Learning"
week: 2
month: 1
status: complete
date: 2026-10-03
prev: "[[week-1-report|Week 1: Mathematical Foundations]]"
next: "[[week-3-report|Week 3: Physics-Informed Learning]]"
map: "[[Knowledge-Map]]"
tags:
  - pinn/week-02
  - pinn/neural-networks
  - pinn/optimization
  - pinn/sciml
concepts:
  - "[[activation-functions]]"
  - "[[backpropagation]]"
  - "[[automatic-differentiation]]"
  - "[[spectral-bias]]"
  - "[[loss-formulation-pinn]]"
---

# Fundamentals of Neural Networks and Optimization for Scientific Machine Learning

> **Week 2 Review Article — Month 1 / Week 2**  
> Foundation learning phase: establishing deep learning architectures, smooth activation dynamics, computational graphs, and optimization landscapes for scientific computing.  
> Practical method followed throughout: **Concept → Code → Output → Interpretation**.

**Navigation:** ← [[week-1-report|Week 1: Mathematical Foundations]] · [[Knowledge-Map|Knowledge Map]] · [[week-3-report|Week 3: Physics-Informed Learning]] →  
**Associated Concepts:** [[activation-functions]] · [[backpropagation]] · [[automatic-differentiation]] · [[spectral-bias]] · [[loss-formulation-pinn]]

---

## Abstract

In Week 1, we established that physical conservation laws are governed by continuous differential operators and that classical numerical grids suffer from the curse of dimensionality. Week 2 resolves the fundamental machine learning question: **how do multi-layer perceptrons, forward-backward propagation, and gradient-based optimizers approximate continuous physical functions?** We dismantle the artificial neuron into its fundamental linear-affine transformation $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$ and non-linear activation $\sigma(\cdot)$, analyzing the Universal Approximation Theorem. We rigorously examine activation functions through the lens of scientific computing, proving why standard piece-wise linear activations (ReLU) fail catastrophically for higher-order differential equations and establishing the absolute necessity of $C^k$-smooth activations ($\tanh$, SIREN sinusoidal).

We formalize forward propagation through deep architectures and analyze empirical loss functionals, contrasting the quadratic sensitivity of Mean Squared Error (MSE) against the robust $L_1$ properties of Mean Absolute Error (MAE). We unpack reverse-mode automatic differentiation as executed in backpropagation, constructing the computational Directed Acyclic Graph (DAG) and deriving exact weight gradient updates $\nabla_\theta \mathcal{L}$. Finally, we evaluate first-order and second-order optimization dynamics (SGD, Adam, L-BFGS) across ill-conditioned, anisotropic loss surfaces and expose **Spectral Bias**—the intrinsic tendency of neural networks to prioritize low-frequency modes over high-frequency spatial gradients. All concepts are demonstrated with runnable PyTorch implementations, exact console logs, and publication-grade visualizations.

## Keywords

Artificial neuron, multi-layer perceptron, affine projection, activation functions, $C^2$ differentiability, hyperbolic tangent, SIREN, universal approximation theorem, forward propagation, Mean Squared Error (MSE), Mean Absolute Error (MAE), computational graph, backpropagation, automatic differentiation, gradient descent, learning rate, Adam optimizer, L-BFGS, spectral bias, `Tensor [Batch, Hidden]`.

---

## 1. Introduction & Week 2 Syllabus

Traditional deep learning was developed primarily for perceptual tasks: image classification, object detection, and speech recognition. In these domains, data is discrete, noisy, and plentiful, and the underlying mathematical mapping is completely unknown. In **Scientific Machine Learning (SciML)**, the problem setting is inverted: data is often extremely sparse, noisy, or expensive to acquire, but we possess profound deductive mathematical prior knowledge in the form of differential equations (Week 1).

**The Architectural Bridge:** To solve differential equations using deep learning, the neural network cannot be treated as a discrete black-box classifier. It must function as a **continuous, differentiable surrogate function** $u_\theta(\mathbf{x}): \mathbb{R}^d \to \mathbb{R}^m$. The network must be capable of being differentiated with respect to its continuous coordinate inputs $(t, x)$ to evaluate the differential operators established in Week 1.

> [!IMPORTANT]
> **The Smoothness Requirement Axiom:** In computer vision, non-smooth activations like $\text{ReLU}(z) = \max(0, z)$ are dominant because their first derivative is simple ($1$ or $0$). In physics-informed learning, **ReLU is completely forbidden** for solving second-order PDEs (such as the Heat or Navier-Stokes equations) because its second derivative is identically zero everywhere ($d^2\text{ReLU}/dz^2 \equiv 0$). Without smooth, continuously differentiable activations, differential operators cannot propagate physical information through the network.

### Week 2 Syllabus & Laboratory Sequence

| Day | Core Focus | Laboratory / Practical Tasks |
| :--- | :--- | :--- |
| [[w2d1\|**W2D1**]] | The Artificial Neuron, Perceptron Geometry, Weights & Biases | Implement single-neuron forward projections; visualize affine hyperplanes; formalize tensor batch operations. |
| [[w2d2\|**W2D2**]] | Activation Functions: The Imperative for Smooth $C^2$ Differentiability | Derive higher-order derivatives of ReLU, Sigmoid, Tanh, and SIREN; prove second-derivative vanishing of ReLU. |
| [[w2d3\|**W2D3**]] | Multi-Layer Perceptrons, Forward Propagation & Graph Topologies | Construct multi-layer PyTorch modules; trace forward tensor flows `[Batch, 1] -> [Batch, 32] -> [Batch, 1]`. |
| [[w2d4\|**W2D4**]] | Loss Functionals (MSE vs. MAE), Spectral Bias & Optimization | Analyze outlier sensitivity of MSE vs. MAE; visualize frequency-dependent convergence (Spectral Bias). |
| [[w2d5\|**W2D5**]] | Backpropagation, Exact Chain Rule Autograd & Optimizer Dynamics | Derive analytical DAG backprop gradients; benchmark SGD vs. Adam on stiff anisotropic quadratic loss surfaces. |

---

## 2. The Artificial Neuron & Perceptron Geometry

### 2.1 The Mathematical Anatomy of a Neuron
The artificial neuron (McCulloch & Pitts, 1943; Rosenblatt, 1958) is the elementary mathematical computing unit of deep learning. It executes two consecutive mathematical operations:

1. **Affine Transformation (Linear Projection):** Computes the weighted inner product of an input coordinate vector $\mathbf{x} \in \mathbb{R}^{d_{in}}$ with a weight vector $\mathbf{w} \in \mathbb{R}^{d_{in}}$, shifted by a scalar bias $b \in \mathbb{R}$:
   $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d_{in}} w_j x_j + b$$
2. **Non-linear Activation:** Passes the scalar pre-activation potential $z$ through a non-linear activation function $\sigma: \mathbb{R} \to \mathbb{R}$:
   $$a = \sigma(z) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

```
Input x1 ───(w1)───┐
Input x2 ───(w2)───┼───> Sum: z = w^T x + b ───> Activation: a = σ(z) ───> Output
Input xd ───(wd)───┘
Bias b   ──────────┘
```

- **Weights ($\mathbf{w}$):** Govern the orientation, spatial scale, and frequency of the linear projection.
- **Bias ($b$):** Translates the separating hyperplane along the normal direction, controlling the spatial activation threshold.
- **Linear Layer Vectorization:** For a layer mapping from $d_{in}$ inputs to $d_{out}$ hidden neurons across a batch of $N$ coordinate points, the affine transformation is expressed compactly in matrix notation:
   $$\mathbf{Z} = \mathbf{X} \mathbf{W}^T + \mathbf{b}^T, \quad \mathbf{X} \in \mathbb{R}^{N \times d_{in}}, \; \mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}, \; \mathbf{b} \in \mathbb{R}^{d_{out}}$$

---

## 3. Activation Functions in Scientific Machine Learning

### 3.1 The Failure of ReLU in Scientific Computing
In standard deep learning, the Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$) is widely celebrated. However, in scientific machine learning, its mathematical properties create severe pathologies:

1. **Discontinuous First Derivative:**
   $$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & z > 0 \\ \text{undefined} & z = 0 \\ 0 & z < 0 \end{cases}$$
2. **Vanishing Second Derivative:**
   $$\frac{d^2}{dz^2}\text{ReLU}(z) \equiv 0 \quad \forall z \ne 0$$

> [!WARNING]
> **The Dangerous Misconception: "ReLU Networks Can Solve General PDEs."**  
> **Why this is physically false:** Any second-order differential equation (e.g., Heat Equation $\partial_t u = \alpha \partial_{xx} u$, Laplace Equation $\nabla^2 u = 0$, Wave Equation $\partial_{tt} u = c^2 \nabla^2 u$) requires non-zero second-order spatial or temporal derivatives. If a neural network $u_\theta$ uses ReLU activations, its second derivative with respect to input coordinates is identically zero ($\frac{\partial^2 u_\theta}{\partial x^2} \equiv 0$). The computed PDE residual evaluates to nonsense, and the gradient of the PDE loss with respect to network parameters vanishes completely ($\nabla_\theta \mathcal{L}_{\mathrm{pde}} = \mathbf{0}$). The network is physically incapable of learning the differential equation.

### 3.2 Smooth Activations: Tanh and SIREN
For physics-informed scientific computing, the hypothesis space must reside in $C^k(\Omega)$ with $k \ge 2$.

1. **Hyperbolic Tangent ($\tanh$):** The canonical activation function for PINNs:
   $$\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
   - Smooth, continuously differentiable to infinite order ($C^\infty$).
   - Symmetric around the origin ($\tanh(0) = 0$).
   - First derivative: $\sigma'(z) = 1 - \tanh^2(z)$.
   - Second derivative: $\sigma''(z) = -2\tanh(z)(1 - \tanh^2(z))$.
2. **Sinusoidal Representation Networks (SIREN):** Periodic activation $\sigma(z) = \sin(\omega_0 z)$ (Sitzmann et al., 2020):
   - First derivative: $\sigma'(z) = \omega_0 \cos(\omega_0 z)$.
   - Second derivative: $\sigma''(z) = -\omega_0^2 \sin(\omega_0 z)$.
   - Since the derivative of a sine is a phase-shifted cosine of identical frequency, derivatives of SIREN networks do not attenuate or vanish across deep architectures, making them extraordinarily effective for wave equations and high-frequency physical fields.

![Activation Functions and Derivatives](assets/activation_functions_and_derivatives.png)

**Figure 1.** *Comparison of activation functions $\sigma(z)$, first derivatives $\sigma'(z)$, and second derivatives $\sigma''(z)$. Notice in the right panel that the second derivative of ReLU is identically zero everywhere, whereas $\tanh$ and Sine provide smooth, non-vanishing higher-order derivatives essential for PDE Laplacians.*

---

## 4. Multi-Layer Perceptrons & Forward Propagation

### 4.1 Architecture and Universal Approximation
A Multi-Layer Perceptron (MLP) consists of an input layer, $L-1$ hidden layers, and an output layer:

$$\mathbf{h}^{(0)} = \mathbf{x} \in \mathbb{R}^{d_{in}}$$
$$\mathbf{h}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1, \dots, L-1$$
$$u_\theta(\mathbf{x}) = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)} \in \mathbb{R}^{d_{out}}$$

where $\theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}_{l=1}^L$ represents the full set of trainable network parameters.

> [!NOTE]
> **Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991):** Let $\sigma$ be any non-constant, bounded, continuous activation function. Then, any continuous function $f \in C(\Omega)$ on a compact domain $\Omega \subset \mathbb{R}^d$ can be approximated to arbitrary precision $\epsilon > 0$ by a single-hidden-layer neural network with a finite number of neurons:
> $$\sup_{\mathbf{x} \in \Omega} |f(\mathbf{x}) - u_\theta(\mathbf{x})| < \epsilon$$
> In Pinkus (1999), this theorem was extended to show that if $\sigma \in C^k$, the network and its derivatives simultaneously approximate $f$ and all its partial derivatives up to order $k$.

---

## 5. Loss Functionals: MSE vs. MAE

### 5.1 Mathematical Formulations
In scientific regression and physics residual minimization, the objective functional measures the discrepancy between model predictions and target conditions:

1. **Mean Squared Error (MSE / $L_2$ Loss):**
   $$\mathcal{L}_{\mathrm{MSE}}(\theta) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   - Gradient: $\frac{\partial \mathcal{L}_{\mathrm{MSE}}}{\partial e_i} = -2(y_i - \hat{y}_i)$. The gradient is proportional to the error magnitude, driving rapid convergence when errors are large.
2. **Mean Absolute Error (MAE / $L_1$ Loss):**
   $$\mathcal{L}_{\mathrm{MAE}}(\theta) = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   - Gradient: $\frac{\partial \mathcal{L}_{\mathrm{MAE}}}{\partial e_i} = -\text{sign}(y_i - \hat{y}_i)$. The gradient magnitude is constant ($\pm 1$), providing robustness against extreme sensor outliers.

![MSE vs MAE Sensitivity](assets/mse_vs_mae_sensitivity.png)

**Figure 2.** *Loss value (left panel) and gradient magnitude (right panel) as a function of residual error $e = y - \hat{y}$. MSE applies a quadratic penalty that magnifies large deviations, making it ideal for enforcing physical conservation laws where large localized violations cannot be tolerated.*

---

## 6. Backpropagation & Optimization Dynamics

### 6.1 Reverse-Mode Automatic Differentiation on Computational Graphs
Backpropagation (Rumelhart et al., 1986) is the reverse-mode accumulation of chain rule gradients across a Directed Acyclic Graph (DAG). For a network with loss $\mathcal{L}$, the gradient with respect to weight matrix $\mathbf{W}^{(l)}$ is computed as:

$$\boldsymbol{\delta}^{(l)} \equiv \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} = \begin{cases} \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \odot \sigma'(\mathbf{z}^{(L)}) & l = L \\ \left( (\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \right) \odot \sigma'(\mathbf{z}^{(l)}) & l = 1, \dots, L-1 \end{cases}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{h}^{(l-1)})^T, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

where $\odot$ denotes element-wise Hadamard multiplication.

### 6.2 Optimizer Dynamics: SGD, Adam, and L-BFGS
1. **Stochastic Gradient Descent (SGD):**
   $$\theta_{k+1} = \theta_k - \eta \nabla_\theta \mathcal{L}(\theta_k)$$
   In ill-conditioned loss valleys with high condition number $\kappa = \lambda_{\max}/\lambda_{\min} \gg 1$, standard SGD oscillates violently across the steep direction while making negligible progress along the slow physical manifold.
2. **Adam (Adaptive Moment Estimation; Kingma & Ba, 2014):**
   Maintains exponentially decaying running averages of past gradients ($m_k$) and squared gradients ($v_k$):
   $$m_k = \beta_1 m_{k-1} + (1 - \beta_1) g_k, \qquad v_k = \beta_2 v_{k-1} + (1 - \beta_2) g_k^2$$
   Adapts effective learning rates on a per-parameter basis, navigating complex, non-convex landscapes reliably.
3. **L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno):**
   A quasi-Newton second-order optimizer that approximates the inverse Hessian matrix $\mathbf{H}^{-1}$ using recent gradient history:
   $$\theta_{k+1} = \theta_k - \eta_k \mathbf{H}_k^{-1} \nabla_\theta \mathcal{L}(\theta_k)$$
   In scientific machine learning, **L-BFGS is the optimizer of choice for fine-tuning**, driving residuals down to machine precision ($\sim 10^{-7}$) where first-order methods stagnate.

![Loss Landscape and Optimizers](assets/loss_landscape_and_optimizers.png)

**Figure 3.** *Optimization trajectories on an anisotropic loss surface ($\kappa = 10$). Standard SGD displays violent orthogonal oscillations in the stiff direction, whereas curvature-adaptive algorithms (Adam / L-BFGS) navigate smoothly along the physical valley toward the global minimum.*

---

## 7. Concept → Code → Output → Interpretation: Building a Differentiable MLP

### 7.1 Practical Implementation in PyTorch

#### 1. Concept
We build a fully differentiable multi-layer perceptron in PyTorch with $\tanh$ activations to approximate a continuous 1D physical target field $f(x) = \sin(\pi x)$ across domain $x \in [-1, 1]$. We train the network via Adam optimization using MSE loss, logging convergence across 100 epochs.

#### 2. Code
```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)

# Define Scientific Multi-Layer Perceptron
class ScientificMLP(nn.Module):
    def __init__(self, in_features=1, hidden_dim=16, out_features=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, x):
        return self.network(x)

# Instantiate model, optimizer, and loss functional
model = ScientificMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

# Generate training coordinate tensors
x_train = torch.linspace(-1.0, 1.0, 21).unsqueeze(1)
y_train = torch.sin(np.pi * x_train)

print(f"Model Architecture:\n{model}")
print(f"Training Batch Tensor Shape: {x_train.shape}")
print("-" * 50)

# Optimization loop
for epoch in range(1, 101):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch in [1, 25, 50, 75, 100]:
        print(f"Epoch {epoch:03d} | MSE Loss: {loss.item():.6f}")

# Verification at test points
x_test = torch.tensor([[-0.5], [0.0], [0.5]])
y_exact = torch.sin(np.pi * x_test)
y_eval = model(x_test)
print("-" * 50)
print("Test Coordinates:   ", x_test.flatten().tolist())
print("Exact Target Field: ", [round(v, 4) for v in y_exact.flatten().tolist()])
print("Model Prediction:   ", [round(v, 4) for v in y_eval.flatten().tolist()])
```

#### 3. Output
```text
Model Architecture:
ScientificMLP(
  (network): Sequential(
    (0): Linear(in_features=1, out_features=16, bias=True)
    (1): Tanh()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): Tanh()
    (4): Linear(in_features=16, out_features=1, bias=True)
  )
)
Training Batch Tensor Shape: torch.Size([21, 1])
--------------------------------------------------
Epoch 001 | MSE Loss: 0.635832
Epoch 025 | MSE Loss: 0.058319
Epoch 050 | MSE Loss: 0.012543
Epoch 075 | MSE Loss: 0.007621
Epoch 100 | MSE Loss: 0.006301
--------------------------------------------------
Test Coordinates:    [-0.5, 0.0, 0.5]
Exact Target Field:  [-1.0, 0.0, 1.0]
Model Prediction:    [-0.9632, -0.0035, 0.9681]
```

#### 4. Interpretation
In 100 epochs, the network reduces the MSE loss from $0.635832$ down to $0.006301$. At the test coordinates, the network predicts $[-0.9632, -0.0035, 0.9681]$ compared to the analytical targets $[-1.0, 0.0, 1.0]$. The network has learned a continuous, differentiable representation of the sinusoidal function. Crucially, because all activation functions are $\tanh$, this trained PyTorch model can now be differentiated with respect to its input coordinates $x$ to evaluate spatial derivatives.

---

## 8. Spectral Bias & The Frequency Principle

A fundamental theoretical limitation of coordinate-based neural networks trained with gradient descent is **Spectral Bias** (Rahaman et al., 2019):

> [!CAUTION]
> **The Frequency Principle:** Neural networks learn low-frequency components of physical target fields first and exponentially fast, whereas high-frequency components converge at an exponentially slower rate.

![Spectral Bias](assets/mlp_function_approximation.png)

**Figure 4.** *Progression of function approximation over training epochs on a multi-scale target field $f(x) = \sin(2\pi x) + 0.4\sin(8\pi x)$. At epoch 50, the low-frequency mode ($2\text{ Hz}$) is completely resolved, while the high-frequency mode ($8\text{ Hz}$) remains unlearned until epoch 1000.*

In scientific computing, spectral bias explains why PINNs struggle to capture shock waves (e.g., in Burgers' equation or compressible aerodynamics) and turbulent boundary layers without specialized architectural modifications (such as Fourier feature embeddings or multi-scale coordinate inputs).

---

## 9. Real-World Applications & Scientific Formulations

### 9.1 Non-linear Stress-Strain Modeling in Solid Mechanics
- **Target:** Surrogate approximation of non-linear hyperelastic material constitutive laws.
- **Formulation:** Map 3D strain tensor $\boldsymbol{\varepsilon} \in \mathbb{R}^6$ to stress tensor $\boldsymbol{\sigma} \in \mathbb{R}^6$.
- **Failure Mode:** Violations of thermodynamically consistent positive-definite strain energy density functionals.

### 9.2 Chemical Reaction Kinetics Surrogates
- **Target:** Emulate stiff chemical kinetic combustion networks (e.g., 50+ chemical species in methane ignition).
- **Formulation:** Map species concentrations $\mathbf{c}(t) \in \mathbb{R}^K$ across microsecond reaction steps.
- **Failure Mode:** Negative concentration predictions violating conservation of mass; extreme stiffness spanning 8 orders of magnitude.

### 9.3 Electromagnetic Scattering Field Approximation
- **Target:** Predict high-frequency radar cross-section scattered electric fields $\mathbf{E}(\mathbf{x})$.
- **Formulation:** Coordinate MLP taking $(x, y, z)$ to complex amplitude $(E_r, E_i)$.
- **Failure Mode:** Catastrophic spectral bias failure due to high wave numbers $k = 2\pi/\lambda \gg 1$.

---

## 10. Conclusion & Pedagogical Handoff

Week 2 has established the core computational and machine learning machinery:
1. Artificial neurons combine affine projections $\mathbf{W}\mathbf{x} + \mathbf{b}$ with non-linear activations $\sigma(\cdot)$.
2. $C^2$-smooth activation functions ($\tanh$, SIREN) are mandatory in scientific machine learning to prevent higher-order differential operators from vanishing.
3. Multi-layer perceptrons act as universal function approximators across continuous coordinate spaces.
4. Backpropagation applies the multivariate chain rule across computational graphs to compute exact parameter update gradients $\nabla_\theta \mathcal{L}$.
5. Optimizers navigate non-convex landscapes, with L-BFGS providing second-order fine-tuning precision.
6. Spectral bias naturally favors low-frequency physical modes over high-frequency spatial gradients.

**Forward Handoff to Week 3:**  
In Week 3, we unite Week 1 (Differential Equations & Physics Residuals) with Week 2 (Neural Networks & Autodiff). We formally introduce **Physics-Informed Neural Networks (PINNs)**: defining collocation points, building the dual computational graph, formulating composite physics-informed loss functionals, and proving why scientific machine learning generalizes beyond training data.

---

## 11. Academic References & Annotated Literature

1. **Cybenko, G. (1989).** *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems, 2(4), 303-314. [DOI: 10.1007/BF02551274](https://doi.org/10.1007/BF02551274)  
   *Contribution:* Proved the Universal Approximation Theorem for arbitrary continuous functions using single-hidden-layer sigmoidal neural networks.
2. **Hornik, K. (1991).** *Approximation capabilities of multilayer feedforward networks.* Neural Networks, 4(2), 251-257. [DOI: 10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T)  
   *Contribution:* Established that multi-layer perceptrons simultaneously approximate continuous functions and their partial derivatives.
3. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** *Learning representations by back-propagating errors.* Nature, 323(6088), 533-536. [DOI: 10.1038/323533a0](https://doi.org/10.1038/323533a0)  
   *Contribution:* Popularized reverse-mode automatic differentiation on directed graphs for multi-layer neural network optimization.
4. **Kingma, D. P., & Ba, J. (2014).** *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980. [DOI: 10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)  
   *Contribution:* Designed the adaptive moment estimation (Adam) algorithm, standardizing deep learning optimization on ill-conditioned non-convex surfaces.
5. **Rahaman, N., et al. (2019).** *On the Spectral Bias of Neural Networks.* International Conference on Machine Learning (ICML). [PMLR 97:5301-5310](https://proceedings.mlr.press/v97/rahaman19a.html)  
   *Contribution:* Mathematically formalized the Frequency Principle, proving that neural networks learn low frequencies before high frequencies.
6. **Sitzmann, V., et al. (2020).** *Implicit Neural Representations with Periodic Activation Functions.* Advances in Neural Information Processing Systems (NeurIPS), 33, 7462-7473.  
   *Contribution:* Introduced SIREN sinusoidal activations, demonstrating that periodic functions retain informative higher-order derivatives for physical modeling.

---

## Appendix A — Computational Reproducibility (UV + Python)

All figures and training scripts in this monograph are deterministically reproducible using `uv`.

```bash
# Verify environment
uv run python --version

# Run deterministic figure generator
uv run python generate_week2_plots.py

# Verify assets on disk
ls -l "month 1/week 2/assets"
```

## Appendix B — Knowledge Graph Connections

**Central MOC Hub:** [[Knowledge-Map]]  
**Upstream:** [[week-1-report|Week 1: Mathematical Foundations]] — supplies continuous differential calculus, differential operators, and boundary conditions.  
**Downstream:** [[week-3-report|Week 3: Physics-Informed Learning]] — embeds governing equations directly into the neural network loss functional via coordinate automatic differentiation.  
**Sibling Concepts:** [[activation-functions]] · [[backpropagation]] · [[automatic-differentiation]] · [[spectral-bias]] · [[loss-formulation-pinn]]
