import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from pathlib import Path

# Styling configuration
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.alpha": 0.22,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

assets_dir = Path("month 1/week 2/assets")
assets_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path("data")

print("=== Generating Week 2 Publication Figures (12 Figures) ===")

# -------------------------------------------------------------
# Figure 1: artificial_neuron_hyperplane_geometry.png
# -------------------------------------------------------------
print("Generating Figure 1: artificial_neuron_hyperplane_geometry.png...")
fig, ax = plt.subplots(figsize=(8, 6.5))
x1 = np.linspace(-2.5, 2.5, 300)
x2 = np.linspace(-2.5, 2.5, 300)
X1, X2 = np.meshgrid(x1, x2)

# Affine transformation z = w1*x1 + w2*x2 + b
w1, w2, b = 1.2, 0.8, -0.4
Z = w1 * X1 + w2 * X2 + b
Sigma = 1.0 / (1.0 + np.exp(-Z))

cs = ax.contourf(X1, X2, Sigma, levels=20, cmap='RdBu_r', alpha=0.85)
cb = fig.colorbar(cs, ax=ax)
cb.set_label(r"Neuron Activation $\sigma(\mathbf{w}^T \mathbf{x} + b)$")

# Decision boundary line: w1*x1 + w2*x2 + b = 0 -> x2 = (-w1*x1 - b)/w2
line_x2 = (-w1 * x1 - b) / w2
ax.plot(x1, line_x2, color='black', lw=2.5, label=r"Separating Hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$")

# Normal vector arrow w = [w1, w2]
origin_pt = [0.0, -b/w2]
ax.quiver(origin_pt[0], origin_pt[1], w1, w2, angles='xy', scale_units='xy', scale=1.5,
          color='#d62728', width=0.008, headwidth=4, headlength=5, label=r"Weight Vector $\mathbf{w} = [1.2, 0.8]^T$")

ax.set_title(r"Artificial Neuron Hyperplane Geometry & Activation Manifold")
ax.set_xlabel(r"Coordinate $x_1$")
ax.set_ylabel(r"Coordinate $x_2$")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.legend(loc="upper left", framealpha=0.9)
plt.tight_layout()
fig.savefig(assets_dir / "artificial_neuron_hyperplane_geometry.png")
plt.close()

# -------------------------------------------------------------
# Figure 2: activation_functions_and_derivatives.png
# -------------------------------------------------------------
print("Generating Figure 2: activation_functions_and_derivatives.png...")
df_acts = pd.read_csv(data_dir / "week2_activation_derivatives.csv")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: Value
ax1.plot(df_acts["z"], df_acts["relu_d0"], label="ReLU", lw=2, color='#1f77b4')
ax1.plot(df_acts["z"], df_acts["gelu_d0"], label="GELU", lw=2, color='#ff7f0e')
ax1.plot(df_acts["z"], df_acts["tanh_d0"], label="Tanh", lw=2.2, color='#2ca02c')
ax1.plot(df_acts["z"], df_acts["siren_d0"], label="SIREN (sin)", lw=2, color='#d62728')
ax1.set_title(r"Activation Function $\sigma(z)$")
ax1.set_xlabel("Input $z$")
ax1.set_ylabel(r"$\sigma(z)$")
ax1.set_ylim(-1.5, 3.0)
ax1.legend(loc="upper left", fontsize=9)

# Panel 2: First Derivative
ax2.plot(df_acts["z"], df_acts["relu_d1"], label="ReLU'", lw=2, color='#1f77b4')
ax2.plot(df_acts["z"], df_acts["gelu_d1"], label="GELU'", lw=2, color='#ff7f0e')
ax2.plot(df_acts["z"], df_acts["tanh_d1"], label="Tanh'", lw=2.2, color='#2ca02c')
ax2.plot(df_acts["z"], df_acts["siren_d1"], label="SIREN' (cos)", lw=2, color='#d62728')
ax2.set_title(r"First Derivative $\sigma'(z)$")
ax2.set_xlabel("Input $z$")
ax2.set_ylabel(r"$\sigma'(z)$")
ax2.set_ylim(-1.2, 1.4)
ax2.legend(loc="upper left", fontsize=9)

# Panel 3: Second Derivative (Laplacian viability)
ax3.plot(df_acts["z"], df_acts["relu_d2"], label="ReLU'' (Zero everywhere)", lw=2.5, color='#1f77b4', linestyle='--')
ax3.plot(df_acts["z"], df_acts["gelu_d2"], label="GELU''", lw=2, color='#ff7f0e')
ax3.plot(df_acts["z"], df_acts["tanh_d2"], label="Tanh''", lw=2.2, color='#2ca02c')
ax3.plot(df_acts["z"], df_acts["siren_d2"], label="SIREN'' (-sin)", lw=2, color='#d62728')
ax3.set_title(r"Second Derivative $\sigma''(z)$ (Laplacian)")
ax3.set_xlabel("Input $z$")
ax3.set_ylabel(r"$\sigma''(z)$")
ax3.set_ylim(-1.2, 1.2)
ax3.legend(loc="lower left", fontsize=9)

plt.tight_layout()
fig.savefig(assets_dir / "activation_functions_and_derivatives.png")
plt.close()

# -------------------------------------------------------------
# Figure 3: relu_laplacian_failure.png
# -------------------------------------------------------------
print("Generating Figure 3: relu_laplacian_failure.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
x_t = torch.linspace(-1.0, 1.0, 300, requires_grad=True).unsqueeze(1)

# Fit or evaluate synthetic wave approximations
# Tanh network vs ReLU network
torch.manual_seed(42)
net_tanh = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
net_relu = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

y_tanh = net_tanh(x_t)
y_relu = net_relu(x_t)

# First and second derivatives
dy_tanh = torch.autograd.grad(y_tanh.sum(), x_t, create_graph=True)[0]
d2y_tanh = torch.autograd.grad(dy_tanh.sum(), x_t)[0]

dy_relu = torch.autograd.grad(y_relu.sum(), x_t, create_graph=True)[0]
d2y_relu = torch.autograd.grad(dy_relu.sum(), x_t)[0]

ax1.plot(x_t.detach().numpy(), y_tanh.detach().numpy(), label="Tanh Network $u_\\theta(x)$", color='#2ca02c', lw=2.2)
ax1.plot(x_t.detach().numpy(), y_relu.detach().numpy(), label="ReLU Network $u_\\theta(x)$", color='#1f77b4', lw=2.2, linestyle='--')
ax1.set_title("Coordinate Network Outputs $u_\\theta(x)$")
ax1.set_xlabel("Coordinate $x$")
ax1.set_ylabel("$u(x)$")
ax1.legend(loc="upper right")

ax2.plot(x_t.detach().numpy(), d2y_tanh.detach().numpy(), label="Tanh $\\partial^2 u / \\partial x^2$ (Smooth Continuous)", color='#2ca02c', lw=2.2)
ax2.plot(x_t.detach().numpy(), d2y_relu.detach().numpy(), label="ReLU $\\partial^2 u / \\partial x^2 \\equiv 0$ (Dead Residuals)", color='#d62728', lw=2.5, linestyle=':')
ax2.set_title("Second Derivative (Laplacian) Residual")
ax2.set_xlabel("Coordinate $x$")
ax2.set_ylabel(r"$\partial^2 u / \partial x^2$")
ax2.legend(loc="lower right")

plt.tight_layout()
fig.savefig(assets_dir / "relu_laplacian_failure.png")
plt.close()

# -------------------------------------------------------------
# Figure 4: siren_high_frequency_derivatives.png
# -------------------------------------------------------------
print("Generating Figure 4: siren_high_frequency_derivatives.png...")
fig, ax = plt.subplots(figsize=(8.5, 5.2))
z_s = np.linspace(-3, 3, 400)
omega_0 = 3.0
s0 = np.sin(omega_0 * z_s)
s1 = omega_0 * np.cos(omega_0 * z_s)
s2 = -(omega_0**2) * np.sin(omega_0 * z_s)
s3 = -(omega_0**3) * np.cos(omega_0 * z_s)

ax.plot(z_s, s0, label=r"$\sigma(z) = \sin(\omega_0 z)$", color='#1f77b4', lw=2)
ax.plot(z_s, s1 / omega_0, label=r"Normalized $\sigma'(z) / \omega_0$", color='#ff7f0e', lw=2, linestyle='--')
ax.plot(z_s, s2 / (omega_0**2), label=r"Normalized $\sigma''(z) / \omega_0^2$", color='#2ca02c', lw=2, linestyle='-.')
ax.plot(z_s, s3 / (omega_0**3), label=r"Normalized $\sigma'''(z) / \omega_0^3$", color='#d62728', lw=2, linestyle=':')

ax.set_title(r"SIREN Derivative Preservation ($\omega_0 = 3.0$)")
ax.set_xlabel("Coordinate $z$")
ax.set_ylabel("Normalized Derivative Amplitude")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
plt.tight_layout()
fig.savefig(assets_dir / "siren_high_frequency_derivatives.png")
plt.close()

# -------------------------------------------------------------
# Figure 5: mlp_architecture_tensor_flow.png
# -------------------------------------------------------------
print("Generating Figure 5: mlp_architecture_tensor_flow.png...")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('off')

# Diagram blocks
boxes = [
    ("Input Batch\n`[N, 1]`", 0.08, 0.5, '#aec7e8'),
    ("Linear 1\n`W_1: [1, 32]`\n`b_1: [32]`", 0.28, 0.5, '#c7e9c0'),
    ("Activation\n`Tanh`\n`[N, 32]`", 0.48, 0.5, '#bcbddc'),
    ("Linear 2\n`W_2: [32, 32]`\n`b_2: [32]`", 0.68, 0.5, '#c7e9c0'),
    ("Output State\n`[N, 1]`", 0.88, 0.5, '#ffbb78')
]

for name, x, y, col in boxes:
    ax.text(x, y, name, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.8', facecolor=col, edgecolor='black', lw=1.5), fontsize=10, weight='bold')

# Connect with arrows
for i in range(len(boxes) - 1):
    x_start = boxes[i][1] + 0.08
    x_end = boxes[i+1][1] - 0.08
    ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8, shrink=0.05))

ax.set_title("Coordinate Multi-Layer Perceptron (MLP) Tensor Flow", fontsize=13, weight='bold', pad=20)
ax.set_xlim(0, 1)
ax.set_ylim(0.2, 0.8)
plt.tight_layout()
fig.savefig(assets_dir / "mlp_architecture_tensor_flow.png")
plt.close()

# -------------------------------------------------------------
# Figure 6: mse_vs_mae_sensitivity.png
# -------------------------------------------------------------
print("Generating Figure 6: mse_vs_mae_sensitivity.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
err = np.linspace(-3.0, 3.0, 300)
mse = err**2
mae = np.abs(err)
huber = np.where(np.abs(err) <= 1.0, 0.5 * err**2, np.abs(err) - 0.5)

grad_mse = 2.0 * err
grad_mae = np.sign(err)
grad_huber = np.where(np.abs(err) <= 1.0, err, np.sign(err))

ax1.plot(err, mse, color='#d62728', lw=2.5, label=r"MSE: $L_2 = e^2$ (Outlier Sensitive)")
ax1.plot(err, mae, color='#1f77b4', lw=2.2, label=r"MAE: $L_1 = |e|$ (Outlier Robust)")
ax1.plot(err, huber, color='#2ca02c', lw=2.0, linestyle='--', label=r"Huber Loss ($\delta=1.0$)")
ax1.set_title("Loss Penalty Functional Comparison")
ax1.set_xlabel("Residual Error $e = y - \hat{y}$")
ax1.set_ylabel("Loss Magnitude $\mathcal{L}(e)$")
ax1.set_ylim(-0.2, 9.0)
ax1.legend(loc="upper center", framealpha=0.9)

ax2.plot(err, grad_mse, color='#d62728', lw=2.5, label=r"$\partial \text{MSE}/\partial e = 2e$ (Linear Gradient)")
ax2.plot(err, grad_mae, color='#1f77b4', lw=2.2, label=r"$\partial \text{MAE}/\partial e = \text{sign}(e)$ (Constant $\pm 1$)")
ax2.plot(err, grad_huber, color='#2ca02c', lw=2.0, linestyle='--', label=r"$\partial \text{Huber}/\partial e$ (Capped Gradient)")
ax2.set_title("Error Gradient Magnitude")
ax2.set_xlabel("Residual Error $e = y - \hat{y}$")
ax2.set_ylabel(r"Loss Gradient $\partial \mathcal{L} / \partial e$")
ax2.set_ylim(-6.5, 6.5)
ax2.legend(loc="upper left", framealpha=0.9)

plt.tight_layout()
fig.savefig(assets_dir / "mse_vs_mae_sensitivity.png")
plt.close()

# -------------------------------------------------------------
# Figure 7: loss_landscape_and_optimizers.png
# -------------------------------------------------------------
print("Generating Figure 7: loss_landscape_and_optimizers.png...")
with open(data_dir / "week2_optimizer_benchmarks.json") as f:
    bench_data = json.load(f)

fig, ax = plt.subplots(figsize=(9, 6.5))
x_grid = np.linspace(-0.5, 2.5, 200)
y_grid = np.linspace(-1.2, 1.2, 200)
XG, YG = np.meshgrid(x_grid, y_grid)
ZG = 0.5 * (XG**2 + 100.0 * YG**2)

cs = ax.contour(XG, YG, ZG, levels=np.logspace(-2, 2.5, 18), cmap='magma', alpha=0.6)
ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

traj_sgd = bench_data["trajectories"]["sgd_momentum"]
traj_adam = bench_data["trajectories"]["adam"]
traj_lbfgs = bench_data["trajectories"]["lbfgs"]

xs_sgd = [p["x"] for p in traj_sgd]
ys_sgd = [p["y"] for p in traj_sgd]
xs_adam = [p["x"] for p in traj_adam]
ys_adam = [p["y"] for p in traj_adam]
xs_lbfgs = [p["x"] for p in traj_lbfgs]
ys_lbfgs = [p["y"] for p in traj_lbfgs]

ax.plot(xs_sgd, ys_sgd, 'o-', color='#d62728', lw=1.8, markersize=3, label="SGD + Momentum (Oscillates across valley)")
ax.plot(xs_adam, ys_adam, 's-', color='#1f77b4', lw=2.0, markersize=3, label="Adam (Adaptive per-coordinate scaling)")
ax.plot(xs_lbfgs, ys_lbfgs, 'd-', color='#2ca02c', lw=2.5, markersize=5, label="L-BFGS (Second-order quasi-Newton step)")

ax.scatter([2.0], [1.0], color='black', s=80, zorder=6, label="Start Point $(2.0, 1.0)$")
ax.scatter([0.0], [0.0], color='gold', edgecolor='black', s=100, zorder=6, label="Global Minimum $(0, 0)$")

ax.set_title(r"Optimization Dynamics on Anisotropic Quadratic Bowl ($\kappa = 100$)")
ax.set_xlabel("Coordinate $x$ (Slow Eigendirection $\lambda_1 = 1$)")
ax.set_ylabel("Coordinate $y$ (Stiff Eigendirection $\lambda_2 = 100$)")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
plt.tight_layout()
fig.savefig(assets_dir / "loss_landscape_and_optimizers.png")
plt.close()

# -------------------------------------------------------------
# Figure 8: adam_moment_dynamics.png
# -------------------------------------------------------------
print("Generating Figure 8: adam_moment_dynamics.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
steps = np.arange(len(traj_adam))
losses_adam = [p["loss"] for p in traj_adam]
losses_sgd = [p["loss"] for p in traj_sgd]
losses_lbfgs = [p["loss"] for p in traj_lbfgs]

ax1.plot(steps, losses_sgd, color='#d62728', lw=2.0, label="SGD Momentum")
ax1.plot(steps, losses_adam, color='#1f77b4', lw=2.2, label="Adam")
ax1.plot(range(len(losses_lbfgs)), losses_lbfgs, color='#2ca02c', lw=2.5, label="L-BFGS")
ax1.set_yscale('log')
ax1.set_title("Loss Convergence Comparison")
ax1.set_xlabel("Iteration Step")
ax1.set_ylabel("Loss $\mathcal{L}$ (Log Scale)")
ax1.legend(loc="upper right")

# Synthetic Adam moment tracking
beta1, beta2 = 0.9, 0.999
m_track = []
v_track = []
m_curr, v_curr = 0.0, 0.0
for p in traj_adam:
    g = p["grad_norm"]
    m_curr = beta1 * m_curr + (1 - beta1) * g
    v_curr = beta2 * v_curr + (1 - beta2) * (g**2)
    m_track.append(m_curr)
    v_track.append(np.sqrt(v_curr))

ax2.plot(steps, m_track, color='#1f77b4', lw=2.2, label=r"First Moment $m_t$ (Directional Velocity)")
ax2.plot(steps, v_track, color='#ff7f0e', lw=2.2, label=r"Second Moment $\sqrt{v_t}$ (RMS Gradient Scale)")
ax2.set_title("Adam Internal Moment Estimates")
ax2.set_xlabel("Iteration Step")
ax2.set_ylabel("Moment Magnitude")
ax2.legend(loc="upper right")

plt.tight_layout()
fig.savefig(assets_dir / "adam_moment_dynamics.png")
plt.close()

# -------------------------------------------------------------
# Figure 9: mlp_function_approximation.png
# -------------------------------------------------------------
print("Generating Figure 9: mlp_function_approximation.png...")
df_target = pd.read_csv(data_dir / "week2_multiscale_target.csv")
fig, ax = plt.subplots(figsize=(9, 5.5))

x_val = df_target["x"].values
y_val = df_target["target_signal"].values

# Synthetic fitting progression across epochs 10, 50, 200, 1000
# Epoch 10 captures low freq sin(2*pi*x)
# Epoch 50 captures carrier + mid freq
# Epoch 1000 captures full multiscale
y_ep10 = 0.8 * np.sin(2.0 * np.pi * x_val)
y_ep50 = np.sin(2.0 * np.pi * x_val) + 0.35 * np.sin(8.0 * np.pi * x_val)
y_ep1000 = y_val - 0.03 * np.sin(32.0 * np.pi * x_val)

ax.plot(x_val, y_val, color='black', lw=2.5, label="Ground Truth Multi-Scale Field $f(x)$")
ax.plot(x_val, y_ep10, color='#1f77b4', lw=1.8, linestyle='--', label="Epoch 10: Carrier frequency resolved ($2\pi$)")
ax.plot(x_val, y_ep50, color='#ff7f0e', lw=2.0, linestyle='-.', label="Epoch 50: Mid frequency emerging ($8\pi$)")
ax.plot(x_val, y_ep1000, color='#2ca02c', lw=2.2, label="Epoch 1000: High frequency resolved ($32\pi$)")

ax.set_title(r"Multi-Scale Function Fitting: $\sin(2\pi x) + 0.4\sin(8\pi x) + 0.15\sin(32\pi x)$")
ax.set_xlabel("Coordinate $x$")
ax.set_ylabel("Field Value $f(x)$")
ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
plt.tight_layout()
fig.savefig(assets_dir / "mlp_function_approximation.png")
plt.close()

# -------------------------------------------------------------
# Figure 10: spectral_bias_fourier_decay.png
# -------------------------------------------------------------
print("Generating Figure 10: spectral_bias_fourier_decay.png...")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
epochs = np.linspace(0, 1000, 100)
# Frequency Principle: Error at frequency k decays as exp(-2 * lambda_k * t) where lambda_k decreases with k
err_low = np.exp(-epochs / 50.0) # low freq k=1 (2*pi)
err_mid = np.exp(-epochs / 250.0) # mid freq k=4 (8*pi)
err_high = np.exp(-epochs / 1200.0) # high freq k=16 (32*pi)

ax.plot(epochs, err_low, color='#2ca02c', lw=2.5, label=r"Low Frequency $k = 1$ ($2\pi$): Rapid Convergence $\tau \approx 50$")
ax.plot(epochs, err_mid, color='#ff7f0e', lw=2.5, label=r"Mid Frequency $k = 4$ ($8\pi$): Moderate Decay $\tau \approx 250$")
ax.plot(epochs, err_high, color='#d62728', lw=2.5, label=r"High Frequency $k = 16$ ($32\pi$): Severe Stagnation $\tau \approx 1200$")

ax.set_yscale('log')
ax.set_title("Spectral Bias / Frequency Principle: Modal Error Decay Rates")
ax.set_xlabel("Training Epochs")
ax.set_ylabel("Relative Modal Fourier Error (Log Scale)")
ax.set_ylim(1e-4, 1.2)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
fig.savefig(assets_dir / "spectral_bias_fourier_decay.png")
plt.close()

# -------------------------------------------------------------
# Figure 11: computational_graph_backprop_dag.png
# -------------------------------------------------------------
print("Generating Figure 11: computational_graph_backprop_dag.png...")
fig, ax = plt.subplots(figsize=(10, 5.0))
ax.axis('off')

# Forward and backward graph schematic
nodes = [
    (r"$x$", 0.1, 0.6, '#aec7e8'),
    (r"$W_1, b_1$", 0.28, 0.8, '#ffbb78'),
    (r"$z_1 = W_1 x + b_1$", 0.35, 0.6, '#c7e9c0'),
    (r"$h_1 = \tanh(z_1)$", 0.55, 0.6, '#bcbddc'),
    (r"$W_2, b_2$", 0.70, 0.8, '#ffbb78'),
    (r"$\hat{y} = W_2 h_1 + b_2$", 0.75, 0.6, '#c7e9c0'),
    (r"$\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$", 0.92, 0.6, '#ff9896')
]

for label, x, y, col in nodes:
    ax.text(x, y, label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=col, edgecolor='black', lw=1.2), fontsize=9.5, weight='bold')

# Forward arrows (black)
fwd_links = [(0.15, 0.30), (0.42, 0.50), (0.60, 0.70), (0.80, 0.88)]
for x1, x2 in fwd_links:
    ax.annotate("", xy=(x2, 0.6), xytext=(x1, 0.6), arrowprops=dict(facecolor='black', width=1.5, headwidth=6, shrink=0.08))

# Backward arrows (red)
bwd_links = [(0.88, 0.80), (0.70, 0.60), (0.50, 0.42), (0.30, 0.15)]
for x1, x2 in bwd_links:
    ax.annotate("", xy=(x2, 0.42), xytext=(x1, 0.42), arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', width=1.5, headwidth=6, shrink=0.08))

ax.text(0.5, 0.65, "Forward Forward Evaluation Pass $\longrightarrow$", color='black', fontsize=10, weight='bold', ha='center')
ax.text(0.5, 0.35, r"$\longleftarrow$ Reverse-Mode Backpropagation ($\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$)", color='#d62728', fontsize=10, weight='bold', ha='center')

ax.set_title("Directed Acyclic Computational Graph (Forward Pass & Reverse Autodiff)", fontsize=13, weight='bold')
ax.set_xlim(0.02, 0.98)
ax.set_ylim(0.25, 0.95)
plt.tight_layout()
fig.savefig(assets_dir / "computational_graph_backprop_dag.png")
plt.close()

# -------------------------------------------------------------
# Figure 12: activation_derivative_saturation.png
# -------------------------------------------------------------
print("Generating Figure 12: activation_derivative_saturation.png...")
fig, ax = plt.subplots(figsize=(8.5, 5.2))
z_span = np.linspace(-6, 6, 400)
sig = 1.0 / (1.0 + np.exp(-z_span))
d_sig = sig * (1.0 - sig)
tanh_val = np.tanh(z_span)
d_tanh = 1.0 - tanh_val**2

ax.plot(z_span, d_sig, color='#1f77b4', lw=2.5, label=r"Sigmoid Derivative $\sigma'(z) = \sigma(1-\sigma) \leq 0.25$")
ax.plot(z_span, d_tanh, color='#2ca02c', lw=2.5, label=r"Tanh Derivative $\tanh'(z) = 1 - \tanh^2(z) \leq 1.0$")
ax.axvspan(-6, -3, color='gray', alpha=0.2, label=r"Saturation Zone ($|z| > 3, \sigma' \to 0$)")
ax.axvspan(3, 6, color='gray', alpha=0.2)

ax.set_title("Activation Gradient Saturation & Vanishing Regimes")
ax.set_xlabel("Input Activation Potential $z$")
ax.set_ylabel("First Derivative Magnitude")
ax.set_ylim(-0.05, 1.1)
ax.legend(loc="upper center", framealpha=0.9, fontsize=9.5)
plt.tight_layout()
fig.savefig(assets_dir / "activation_derivative_saturation.png")
plt.close()

print("All 12 Week 2 figures successfully generated!")
