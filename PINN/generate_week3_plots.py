"""Generate all publication-grade figures for PINN Week 3."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

OUT = Path("month 1/week 3/assets")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. Collocation Sampling Strategies
N_pts = 100
# 1. Uniform Grid
grid_1d = np.linspace(0.05, 0.95, 10)
gx, gy = np.meshgrid(grid_1d, grid_1d)
pts_grid = np.column_stack([gx.ravel(), gy.ravel()])

# 2. Pure Random (Monte Carlo)
pts_rand = rng.uniform(0, 1, size=(N_pts, 2))

# 3. Latin Hypercube Sampling (LHS)
def latin_hypercube(n, d, rng):
    result = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.uniform(0, 1, n)) / n
    return result
pts_lhs = latin_hypercube(N_pts, 2, rng)

# 4. Residual-Adaptive (cluster around shock/boundary x=0.5)
pts_adaptive_x = np.concatenate([rng.normal(0.5, 0.08, int(N_pts * 0.7)), rng.uniform(0, 1, int(N_pts * 0.3))])
pts_adaptive_x = np.clip(pts_adaptive_x, 0.0, 1.0)
pts_adaptive_y = rng.uniform(0, 1, len(pts_adaptive_x))
pts_adaptive = np.column_stack([pts_adaptive_x, pts_adaptive_y])

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
titles = ["Uniform Grid (Grid Locking)", "Uniform Random (Clustering & Voids)", "Latin Hypercube (Optimal Space-Filling)", "Adaptive Sampling (Residual-Driven)"]
datasets = [pts_grid, pts_rand, pts_lhs, pts_adaptive]
colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

for ax, data, title, c in zip(axes, datasets, titles, colors):
    ax.scatter(data[:, 0], data[:, 1], c=c, s=40, edgecolors='k', alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Spatial Coordinate x")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

axes[0].set_ylabel("Spatial Coordinate y")
fig.suptitle(r"Collocation Point Sampling Strategies in Domain $\Omega \subset \mathbb{R}^2$", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "collocation_sampling_strategies.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. Autodiff vs Finite Difference Precision
h_steps = np.logspace(-15, 0, 80)
exact_d = np.cos(1.0)

fd_forward = np.abs((np.sin(1.0 + h_steps) - np.sin(1.0)) / h_steps - exact_d)
fd_central = np.abs((np.sin(1.0 + h_steps) - np.sin(1.0 - h_steps)) / (2 * h_steps) - exact_d)
autograd_err = np.full_like(h_steps, 2.2e-16)

fig, ax = plt.subplots(figsize=(9, 6))
ax.loglog(h_steps, fd_forward, 'r--', lw=2, label="Forward Difference: Truncation $\mathcal{O}(h)$ + Roundoff $\mathcal{O}(\epsilon/h)$")
ax.loglog(h_steps, fd_central, 'b-', lw=2.5, label="Central Difference: Truncation $\mathcal{O}(h^2)$ + Roundoff $\mathcal{O}(\epsilon/h)$")
ax.loglog(h_steps, autograd_err, 'g-', lw=3, label=r"Automatic Differentiation (Exact to Machine Precision $\sim 10^{-16}$)")
ax.axvline(1e-8, color='k', ls=':', alpha=0.6, label=r"Optimal FD Step $h \approx \sqrt{\epsilon} \approx 10^{-8}$")
ax.set_title("Differentiation Precision: Automatic Differentiation vs. Finite Differences", fontsize=14)
ax.set_xlabel("Step Size h")
ax.set_ylabel("Absolute Derivative Error")
ax.set_ylim(1e-18, 1e2)
ax.legend(fontsize=10, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "autodiff_vs_finitediff_error.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. PINN Architecture / Dual Graph Schematic
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
boxes = [
    (0.05, 0.4, 0.18, 0.2, "Input Coordinates\n$(\mathbf{x}, t) \sim \Omega \cup \partial\Omega$", "#e8f4f8"),
    (0.32, 0.4, 0.2, 0.2, "Neural Network\n$u_\\theta(\\mathbf{x}, t)$\n(Shared Weights $\\theta$)", "#d1e7dd"),
    (0.60, 0.6, 0.2, 0.22, "Automatic Diff\n$\\partial_t u_\\theta, \\nabla u_\\theta, \\Delta u_\\theta$\n(Exact Derivatives)", "#fff3cd"),
    (0.60, 0.18, 0.2, 0.2, "Boundary & Initial\nEvaluation\n$u_\\theta|_{\\partial\\Omega}, u_\\theta|_{t=0}$", "#f8d7da"),
    (0.88, 0.4, 0.15, 0.22, "Total Loss\n$\\mathcal{L}_\\mathrm{total} = \\lambda_p \\mathcal{L}_p\n+ \\lambda_b \\mathcal{L}_b + \\lambda_d \\mathcal{L}_d$", "#cff4fc")
]
for x0, y0, w, h, text, color in boxes:
    patch = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=color, edgecolor='black', lw=1.5)
    ax.add_patch(patch)
    ax.text(x0 + w/2, y0 + h/2, text, ha='center', va='center', fontsize=11, weight='bold')

ax.annotate("", xy=(0.32, 0.5), xytext=(0.23, 0.5), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax.annotate("", xy=(0.60, 0.71), xytext=(0.52, 0.55), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax.annotate("", xy=(0.60, 0.28), xytext=(0.52, 0.45), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax.annotate("", xy=(0.88, 0.58), xytext=(0.80, 0.68), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax.annotate("", xy=(0.88, 0.45), xytext=(0.80, 0.31), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
ax.annotate("Backpropagation via $\\nabla_\\theta \\mathcal{L}_\\mathrm{total}$", xy=(0.42, 0.38), xytext=(0.88, 0.38),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.35", color='crimson', lw=2.5))

ax.set_title("The Dual Computational Graph of Physics-Informed Neural Networks", fontsize=15, y=0.95)
fig.tight_layout()
fig.savefig(OUT / "pinn_computational_dual_graph.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. Physics vs Pure Data Generalization
t_all = np.linspace(0, 5, 500)
u_true = np.exp(-0.2 * t_all) * np.cos(2 * np.pi * t_all)

t_train = np.linspace(0, 1.5, 18)
u_train = np.exp(-0.2 * t_train) * np.cos(2 * np.pi * t_train) + rng.normal(0, 0.05, len(t_train))

u_pure_data = np.exp(-0.2 * t_all) * np.cos(2 * np.pi * t_all)
u_pure_data[t_all > 1.5] = (u_true[t_all > 1.5] + 0.8 * (t_all[t_all > 1.5] - 1.5)**1.7 * np.sin(3 * t_all[t_all > 1.5]))

u_pinn = u_true + 0.02 * np.sin(4 * np.pi * t_all)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

ax1.plot(t_all, u_true, 'k--', lw=2, label="Physical Truth")
ax1.scatter(t_train, u_train, color='red', s=50, zorder=5, label="Sparse Measurements (Training Region)")
ax1.plot(t_all, u_pure_data, 'r-', lw=2.5, label="Standard Deep Learning (Black Box)")
ax1.axvspan(0, 1.5, color='green', alpha=0.1, label="Interpolation Regime")
ax1.axvspan(1.5, 5.0, color='gray', alpha=0.15, label="Extrapolation Regime")
ax1.set_title("Pure Data-Driven Model: Catastrophic Extrapolation Failure", fontsize=13)
ax1.set_xlabel("Time t (s)")
ax1.set_ylabel("System State u(t)")
ax1.legend(loc="lower left", fontsize=9.5)

ax2.plot(t_all, u_true, 'k--', lw=2, label="Physical Truth")
ax2.scatter(t_train, u_train, color='red', s=50, zorder=5, label="Sparse Measurements")
ax2.plot(t_all, u_pinn, 'b-', lw=2.5, label="PINN (Physics Governing Law Constrained)")
ax2.axvspan(0, 1.5, color='green', alpha=0.1)
ax2.axvspan(1.5, 5.0, color='gray', alpha=0.15)
ax2.set_title("Physics-Informed Model: Physically Consistent Generalization", fontsize=13)
ax2.set_xlabel("Time t (s)")
ax2.legend(loc="lower left", fontsize=9.5)

fig.suptitle("Inductive Bias Comparison: Pure Data Learning vs. Physics-Informed Learning", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "physics_vs_pure_data_learning.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. Physics Residual Field Decay
x_dom = np.linspace(0, 1, 60)
y_dom = np.linspace(0, 1, 60)
X_dom, Y_dom = np.meshgrid(x_dom, y_dom)

res_early = np.abs(np.sin(2 * np.pi * X_dom) * np.cos(2 * np.pi * Y_dom)) + 0.4
res_late = 1e-4 * np.abs(np.sin(2 * np.pi * X_dom) * np.sin(2 * np.pi * Y_dom)) + 1e-5 * rng.uniform(0, 1, (60, 60))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cp1 = ax1.contourf(X_dom, Y_dom, res_early, levels=20, cmap="inferno")
fig.colorbar(cp1, ax=ax1, label=r"Residual $|\mathcal{R}(\mathbf{x})|$")
ax1.set_title("Epoch 100: High Physics Violation ($\sim 10^0$)", fontsize=13)
ax1.set_xlabel("x")
ax1.set_ylabel("y")

cp2 = ax2.contourf(X_dom, Y_dom, np.log10(res_late), levels=20, cmap="viridis")
fig.colorbar(cp2, ax=ax2, label=r"Log10 Residual $\log_{10}|\mathcal{R}(\mathbf{x})|$")
ax2.set_title(r"Epoch 5000: Physics Adherence ($\sim 10^{-4}$)", fontsize=13)
ax2.set_xlabel("x")
ax2.set_ylabel("y")

fig.suptitle(r"Spatial Evolution of the PDE Residual Field $\mathcal{R}(\mathbf{x}) = \mathcal{N}[u_\theta] - f$", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "physics_residual_field_2d.png")
plt.close(fig)

print("Week 3 plots generated successfully:")
for p in OUT.glob("*.png"):
    print(f"  - {p}")
