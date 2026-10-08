"""Generate all publication-grade figures for PINN Week 2."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

OUT = Path("month 1/week 2/assets")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. Activations & Derivatives
z = np.linspace(-4, 4, 400)

# Functions
relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
sine = np.sin(z)

# Derivatives
d_relu = np.where(z > 0, 1.0, 0.0)
d_sigmoid = sigmoid * (1 - sigmoid)
d_tanh = 1 - tanh**2
d_sine = np.cos(z)

# Second derivatives (crucial for 2nd order PDEs!)
d2_relu = np.zeros_like(z)
d2_tanh = -2 * tanh * d_tanh
d2_sine = -np.sin(z)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

axes[0].plot(z, relu, label="ReLU", lw=2)
axes[0].plot(z, sigmoid, label="Sigmoid", lw=2)
axes[0].plot(z, tanh, label="Tanh", lw=2)
axes[0].plot(z, sine, label="Sine (SIREN)", lw=2)
axes[0].set_title(r"Activation Function $\sigma(z)$", fontsize=14)
axes[0].set_ylabel(r"$\sigma(z)$")
axes[0].legend(fontsize=10)

axes[1].plot(z, d_relu, label="d(ReLU)/dz", lw=2)
axes[1].plot(z, d_sigmoid, label="d(Sigmoid)/dz", lw=2)
axes[1].plot(z, d_tanh, label="d(Tanh)/dz", lw=2)
axes[1].plot(z, d_sine, label="d(Sine)/dz", lw=2)
axes[1].set_title(r"First Derivative $\sigma'(z)$", fontsize=14)
axes[1].set_ylabel(r"$\sigma'(z)$")
axes[1].legend(fontsize=10)

axes[2].plot(z, d2_relu, label=r"$d^2(\mathrm{ReLU})/dz^2 \equiv 0$", lw=2, ls='--')
axes[2].plot(z, d2_tanh, label=r"$d^2(\mathrm{Tanh})/dz^2$", lw=2)
axes[2].plot(z, d2_sine, label=r"$d^2(\mathrm{Sine})/dz^2$", lw=2)
axes[2].set_title(r"Second Derivative $\sigma''(z)$ (PDE Laplacian)", fontsize=14)
axes[2].set_ylabel(r"$\sigma''(z)$")
axes[2].legend(fontsize=10)

for ax in axes:
    ax.set_xlabel("Input Activation Potential z")

fig.suptitle("Activation Functions for Scientific ML: Non-vanishing Higher-Order Derivatives", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "activation_functions_and_derivatives.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. Loss Landscape & Optimizers
w1 = np.linspace(-3.5, 3.5, 100)
w2 = np.linspace(-3.5, 3.5, 100)
W1, W2 = np.meshgrid(w1, w2)
# Anisotropic quadratic bowl (condition number kappa = 10)
Loss = 0.1 * W1**2 + 1.0 * W2**2

# Simulate SGD trajectory (oscillatory in stiff direction)
curr = np.array([3.0, 2.5])
lr = 0.8
sgd_path = [curr.copy()]
for _ in range(12):
    grad = np.array([0.2 * curr[0], 2.0 * curr[1]])
    curr = curr - lr * grad
    sgd_path.append(curr.copy())
sgd_path = np.array(sgd_path)

# Simulate Adam / Momentum path
curr_m = np.array([3.0, 2.5])
v = np.zeros(2)
adam_path = [curr_m.copy()]
for _ in range(12):
    grad = np.array([0.2 * curr_m[0], 2.0 * curr_m[1]])
    v = 0.7 * v + 0.3 * grad
    curr_m = curr_m - 0.9 * v
    adam_path.append(curr_m.copy())
adam_path = np.array(adam_path)

fig, ax = plt.subplots(figsize=(8, 7))
cp = ax.contour(W1, W2, Loss, levels=18, cmap="coolwarm")
ax.clabel(cp, inline=True, fontsize=9)
ax.plot(sgd_path[:, 0], sgd_path[:, 1], 'ro-', lw=2, markersize=6, label="Standard SGD (Stiff Oscillations)")
ax.plot(adam_path[:, 0], adam_path[:, 1], 'go-', lw=2.5, markersize=6, label="Momentum / Adam (Curvature Adaptive)")
ax.plot(0, 0, 'k*', markersize=14, label="Global Minimum $(0,0)$")
ax.set_title("Optimization Dynamics on Anisotropic Loss Surface", fontsize=15)
ax.set_xlabel(r"Weight $w_1$ (Soft Eigen-direction)")
ax.set_ylabel(r"Weight $w_2$ (Stiff Eigen-direction)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "loss_landscape_and_optimizers.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. Learning Rate Dynamics
epochs = np.arange(1, 151)
loss_optimal = 2.0 * np.exp(-0.06 * epochs) + 0.005 + rng.normal(0, 0.001, len(epochs))
loss_too_small = 2.0 * np.exp(-0.01 * epochs)
loss_too_large = 1.8 * np.exp(-0.05 * epochs) + 0.3 * np.abs(np.sin(0.4 * epochs)) + rng.normal(0, 0.02, len(epochs))
loss_diverge = 0.5 * np.exp(0.04 * epochs[:45])

fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(epochs, loss_optimal, 'g-', lw=2.5, label="Optimal Learning Rate ($\eta^*$)")
ax.semilogy(epochs, loss_too_small, 'b--', lw=2, label="Too Small Learning Rate ($\eta \ll \eta^*$)")
ax.semilogy(epochs, loss_too_large, 'm-.', lw=2, label="High Learning Rate: Limit Cycles / Oscillations")
ax.semilogy(epochs[:45], loss_diverge, 'r-', lw=2.5, label="Divergent Learning Rate ($\eta > 2/\lambda_{max}$)")
ax.set_title("Empirical Loss Trajectories Across Learning Rate Regimes", fontsize=14)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Squared Error Loss (Log Scale)")
ax.set_ylim(1e-3, 1e2)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "learning_rate_dynamics.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. MSE vs MAE Sensitivity
error = np.linspace(-3, 3, 300)
mse = error**2
mae = np.abs(error)
d_mse = 2 * error
d_mae = np.sign(error)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(error, mse, 'r-', lw=2.5, label=r"MSE: $\mathcal{L} = (y - \hat{y})^2$")
ax1.plot(error, mae, 'b--', lw=2.5, label=r"MAE: $\mathcal{L} = |y - \hat{y}|$")
ax1.axvline(2.5, color='gray', ls=':', label="Outlier Residual ($e = 2.5$)")
ax1.set_title("Loss Value vs. Prediction Error", fontsize=14)
ax1.set_xlabel(r"Residual Error $e = y - \hat{y}$")
ax1.set_ylabel("Loss Magnitude")
ax1.legend(fontsize=11)

ax2.plot(error, d_mse, 'r-', lw=2.5, label=r"$\partial \mathrm{MSE}/\partial e = 2e$ (Linear Gradient)")
ax2.plot(error, d_mae, 'b--', lw=2.5, label=r"$\partial \mathrm{MAE}/\partial e = \mathrm{sign}(e)$ (Constant Gradient)")
ax2.set_title("Gradient Magnitude (Driving Force of Optimization)", fontsize=14)
ax2.set_xlabel(r"Residual Error $e = y - \hat{y}$")
ax2.set_ylabel("Gradient Magnitude")
ax2.legend(fontsize=11)

fig.suptitle("Loss Function Comparison: Sensitivity to Scientific Outliers", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "mse_vs_mae_sensitivity.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. Spectral Bias in MLP Approximation
x_dense = np.linspace(0, 1, 300)
y_target = np.sin(2 * np.pi * x_dense) + 0.4 * np.sin(8 * np.pi * x_dense)

# Simulate low-to-high frequency learning progression
pred_ep10 = 0.3 * np.sin(2 * np.pi * x_dense)
pred_ep50 = 0.95 * np.sin(2 * np.pi * x_dense)
pred_ep200 = 1.0 * np.sin(2 * np.pi * x_dense) + 0.15 * np.sin(8 * np.pi * x_dense)
pred_ep1000 = y_target.copy()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_dense, y_target, 'k-', lw=3, label=r"True Target: $\sin(2\pi x) + 0.4\sin(8\pi x)$")
ax.plot(x_dense, pred_ep10, 'r--', lw=1.8, label="Epoch 10: Initializing global mean")
ax.plot(x_dense, pred_ep50, 'g-.', lw=2, label="Epoch 50: Low frequency (2 Hz) learned")
ax.plot(x_dense, pred_ep200, 'b:', lw=2.2, label="Epoch 200: High frequency (8 Hz) resolving")
ax.plot(x_dense, pred_ep1000, 'm-', alpha=0.8, lw=2, label="Epoch 1000: Full multi-scale convergence")
ax.set_title("Spectral Bias of Neural Networks: Low Frequencies Converge Prior to High Frequencies", fontsize=14)
ax.set_xlabel("Spatial Domain x")
ax.set_ylabel("Field u(x)")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "mlp_function_approximation.png")
plt.close(fig)

print("Week 2 plots generated successfully:")
for p in OUT.glob("*.png"):
    print(f"  - {p}")
