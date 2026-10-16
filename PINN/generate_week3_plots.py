"""
generate_week3_plots.py
Deterministically generates all 12 publication-grade figures for PINN Week 3.
Conforms strictly to vault styling standards: Seaborn whitegrid, 180 DPI, bbox_inches='tight'.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import scipy.spatial

# Set publication style
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

OUT = Path("month 1/week 3/assets")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

print("Generating Week 3 visual assets in month 1/week 3/assets/...")

# -------------------------------------------------------------------------
# Figure 1: physics_vs_pure_data_learning.png
# Extrapolation catastrophe: interpolation vs extrapolation for black-box vs PINN
# -------------------------------------------------------------------------
t_full = np.linspace(0, 5.0, 500)
gamma = 0.2
omega = 2.0
u_exact = np.exp(-gamma * t_full) * np.cos(omega * t_full)

# Data in [0, 1.5]
t_train = np.linspace(0.1, 1.4, 15)
u_train = np.exp(-gamma * t_train) * np.cos(omega * t_train) + rng.normal(0, 0.03, len(t_train))

# Pure data-driven polynomial / unconstrained MLP extrapolation failure
poly = np.poly1d(np.polyfit(t_train, u_train, 6))
u_blackbox = poly(t_full)
# Suppress overflow for plotting
u_blackbox = np.clip(u_blackbox, -4.0, 4.0)

# PINN prediction: enforces m*u'' + c*u' + k*u = 0, matches exact closely
u_pinn = u_exact + 0.015 * np.exp(-gamma * t_full) * np.sin(omega * t_full * 1.5)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.axvspan(0, 1.5, color="#e6f5d0", alpha=0.5, label="Training Domain $[0, 1.5]$ (Interpolation)")
ax.axvspan(1.5, 5.0, color="#fde0dd", alpha=0.5, label="Unseen Domain $(1.5, 5.0]$ (Extrapolation)")

ax.plot(t_full, u_exact, 'k--', linewidth=2.2, label=r"Physical Ground Truth $u(t) = e^{-\gamma t}\cos(\omega t)$")
ax.plot(t_full, u_pinn, color="#2b8cbe", linewidth=2.5, label=r"Physics-Informed Neural Network (PINN)")
ax.plot(t_full, u_blackbox, color="#e41a1c", linewidth=2.2, linestyle="-.", label="Pure Data-Driven Black-Box (MLP / Polynomial)")
ax.scatter(t_train, u_train, color="#252525", s=50, zorder=5, edgecolors='white', label=r"Sparse Sensor Observations ($N=15$)")

ax.set_ylim(-2.5, 2.5)
ax.set_xlim(0, 5.0)
ax.set_xlabel("Time Coordinate $t$ (s)")
ax.set_ylabel("Displacement Field $u(t)$ (m)")
ax.set_title("The Extrapolation Catastrophe: Inductive Physics vs. Pure Data Fitting", pad=12)
ax.legend(loc="upper right", framealpha=0.92)
fig.savefig(OUT / "physics_vs_pure_data_learning.png")
plt.close(fig)
print("1. physics_vs_pure_data_learning.png generated.")

# -------------------------------------------------------------------------
# Figure 2: pinn_computational_dual_graph.png
# Architectural schematic of Primary, Secondary, and Tertiary computational graphs
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

# Graph boxes
def draw_card(ax, x, y, w, h, title, subtitle, color, edgecolor):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                fc=color, ec=edgecolor, lw=2.0)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#111111")
    ax.text(x + w/2, y + h*0.3, subtitle, ha="center", va="center", fontsize=9.5, color="#333333")

# Inputs
draw_card(ax, 0.4, 4.5, 2.0, 1.6, "Coordinates $\mathbf{x}$", "Spatial $x$, Time $t$\nTensor `[Batch, 2]`", "#edf8fb", "#66c2a4")

# Primary Graph
draw_card(ax, 3.2, 4.5, 2.6, 1.6, "Primary Graph $\mathcal{N}_\\theta$", "Deep MLP Surrogate\n$u_\\theta = \\mathbf{W}_L \\sigma(\\dots)$", "#ccece6", "#41ae76")

# State prediction
draw_card(ax, 6.6, 4.5, 2.0, 1.6, "State Field $u_\\theta$", "Continuous Output\nTensor `[Batch, 1]`", "#99d8c9", "#238b45")

# Secondary Graph: Autodiff Operator
draw_card(ax, 6.2, 1.2, 2.8, 1.8, "Secondary Graph", "Exact Reverse Autodiff\n`torch.autograd.grad`\n$\\partial_t u, \\partial_x u, \\partial_{xx} u$", "#e0f3f8", "#2b8cbe")

# Residual Assembly & Composite Loss
draw_card(ax, 9.6, 2.8, 2.1, 2.4, "Loss Assembly", "$\\mathcal{L}_{\\mathrm{total}} = \\lambda_f \\mathcal{L}_{\\mathrm{pde}}$\n$+ \\lambda_b \\mathcal{L}_{\\mathrm{bc}} + \\lambda_0 \\mathcal{L}_{\\mathrm{ic}}$", "#fee8c8", "#e34a33")

# Boundary / IC data source
draw_card(ax, 3.2, 1.2, 2.6, 1.8, "Boundary / IC Data", "Dirichlet / Neumann\n$u(0, x), u(t, \\pm 1)$", "#fef0d9", "#fdbb84")

# Arrows
def draw_arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="#252525"))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.25, text, ha="center", fontsize=9, fontweight="bold", color="#08519c")

draw_arrow(ax, 2.4, 5.3, 3.2, 5.3, "requires_grad=True")
draw_arrow(ax, 5.8, 5.3, 6.6, 5.3, "Forward Pass")
draw_arrow(ax, 7.6, 4.5, 7.6, 3.0, "Coordinate Diff")
draw_arrow(ax, 9.0, 2.1, 9.6, 3.5, "Residual $\\mathcal{R}$")
draw_arrow(ax, 8.6, 5.3, 9.6, 4.5, "Data Fit")
draw_arrow(ax, 5.8, 2.1, 9.6, 3.0, "BC / IC Error")

# Tertiary Parameter Graph Arrow
ax.annotate("", xy=(4.5, 4.5), xytext=(10.65, 2.8),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="#d95f0e", linestyle="dashed",
                            connectionstyle="arc3,rad=0.35"))
ax.text(7.5, 0.4, "Tertiary Graph: Parameter Backpropagation $\\nabla_\\theta \\mathcal{L}_{\\mathrm{total}}$",
        ha="center", fontsize=10.5, fontweight="bold", color="#d95f0e")

ax.set_title("Dual Computational Graph Architecture of Physics-Informed Neural Networks", fontsize=14, pad=15)
fig.savefig(OUT / "pinn_computational_dual_graph.png")
plt.close(fig)
print("2. pinn_computational_dual_graph.png generated.")

# -------------------------------------------------------------------------
# Figure 3: autodiff_vs_finitediff_error.png
# Log-log error scaling: FD truncation, roundoff, and Autodiff machine precision
# -------------------------------------------------------------------------
h_steps = np.logspace(-15, 0, 80)
# Target function f(x) = sin(x) at x = 1.0, exact f'(1.0) = cos(1.0)
x0 = 1.0
exact_df = np.cos(x0)

# Forward Difference: (f(x+h) - f(x))/h
fd_forward = (np.sin(x0 + h_steps) - np.sin(x0)) / h_steps
err_forward = np.abs(fd_forward - exact_df)

# Central Difference: (f(x+h) - f(x-h))/(2h)
fd_central = (np.sin(x0 + h_steps) - np.sin(x0 - h_steps)) / (2.0 * h_steps)
err_central = np.abs(fd_central - exact_df)

# Autograd: invariant machine precision ~ 1e-16
err_autodiff = np.full_like(h_steps, 1.11e-16)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(h_steps, err_forward, label=r"Forward Difference $\mathcal{O}(h)$", color="#3182bd", linewidth=2.2)
ax.loglog(h_steps, err_central, label=r"Central Difference $\mathcal{O}(h^2)$", color="#31a354", linewidth=2.2)
ax.loglog(h_steps, err_autodiff, label="Coordinate Automatic Differentiation", color="#e6550d", linewidth=2.5, linestyle="--")

# Theoretical error bounds
h_theory = np.logspace(-8, 0, 50)
ax.loglog(h_theory, 0.5 * h_theory, 'k:', alpha=0.5, label=r"Theoretical Truncation Bound")
ax.axvline(1e-8, color="#756bb1", linestyle=":", label="Optimal FD Step h* ~ sqrt(eps_mach)")

ax.set_xlabel("Perturbation Step Size $h$ (Dimensionless)")
ax.set_ylabel("Absolute Derivative Error $|f'(x) - \hat{f}'(x)|$")
ax.set_title("Numerical Derivative Error Scaling: Autodiff vs. Finite Differences", pad=12)
ax.set_ylim(1e-18, 1e2)
ax.set_xlim(1e-15, 1e0)
ax.legend(loc="upper center", framealpha=0.92, ncol=2)
fig.savefig(OUT / "autodiff_vs_finitediff_error.png")
plt.close(fig)
print("3. autodiff_vs_finitediff_error.png generated.")

# -------------------------------------------------------------------------
# Figure 4: collocation_sampling_strategies.png
# 4-panel comparison: Uniform, Random, Latin Hypercube (LHS), Residual-Adaptive (RAD)
# -------------------------------------------------------------------------
N_pts = 100
# Uniform Grid
grid_1d = np.linspace(0.05, 0.95, 10)
gx, gy = np.meshgrid(grid_1d, grid_1d)
pts_grid = np.column_stack([gx.ravel(), gy.ravel()])

# Monte Carlo Random
pts_rand = rng.uniform(0, 1, size=(N_pts, 2))

# Latin Hypercube Sampling
def lhs_2d(n, rng):
    x = (rng.permutation(n) + rng.uniform(0, 1, n)) / n
    y = (rng.permutation(n) + rng.uniform(0, 1, n)) / n
    return np.column_stack([x, y])
pts_lhs = lhs_2d(N_pts, rng)

# Residual-Adaptive Sampling (clustered around shock front x=0.5)
pts_rad_x = np.concatenate([rng.normal(0.5, 0.07, int(N_pts * 0.7)), rng.uniform(0, 1, int(N_pts * 0.3))])
pts_rad_x = np.clip(pts_rad_x, 0.02, 0.98)
pts_rad_y = rng.uniform(0, 1, len(pts_rad_x))
pts_rad = np.column_stack([pts_rad_x, pts_rad_y])

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
panels = [
    (pts_grid, "Uniform Grid (Grid Locking)", "#2b8cbe"),
    (pts_rand, "Uniform Random (Clustering / Voids)", "#e41a1c"),
    (pts_lhs, "Latin Hypercube (Optimal Space-Filling)", "#31a354"),
    (pts_rad, "Residual-Adaptive (Shock Clustering)", "#756bb1")
]

for ax, (data, title, col) in zip(axes, panels):
    ax.scatter(data[:, 0], data[:, 1], c=col, s=45, edgecolors='k', alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Spatial Coordinate $x$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

axes[0].set_ylabel("Spatial / Temporal Coordinate $y$")
fig.suptitle(r"Collocation Sampling Strategies across Physical Domain $\Omega \subset \mathbb{R}^2$", fontsize=15, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "collocation_sampling_strategies.png")
plt.close(fig)
print("4. collocation_sampling_strategies.png generated.")

# -------------------------------------------------------------------------
# Figure 5: physics_residual_field_2d.png
# 2D contour of PDE residual field comparing early vs converged epochs
# -------------------------------------------------------------------------
x_grid = np.linspace(-1, 1, 100)
t_grid = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_grid, t_grid)

# Early epoch residual: large magnitude violations everywhere, O(1)
R_early = np.abs(np.cos(np.pi * X) * np.exp(-T) + 0.8 * np.sin(2 * np.pi * X))

# Converged epoch residual: orders of magnitude smaller, concentrated near steep gradient
R_conv = 1e-4 * np.exp(-((X - 0.1 * T)**2) / 0.02) + 1e-5 * rng.uniform(0, 1, X.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
im1 = ax1.contourf(X, T, R_early, levels=30, cmap="inferno")
fig.colorbar(im1, ax=ax1, label="Residual Magnitude $|\mathcal{R}(\mathbf{x})|$")
ax1.set_title("Epoch 10: Untrained Network Residual Field ($|\mathcal{R}| \sim 10^0$)")
ax1.set_xlabel("Spatial Coordinate $x$ (m)")
ax1.set_ylabel("Temporal Coordinate $t$ (s)")

im2 = ax2.contourf(X, T, R_conv, levels=30, cmap="viridis")
fig.colorbar(im2, ax=ax2, label="Residual Magnitude $|\mathcal{R}(\mathbf{x})|$")
ax2.set_title("Epoch 1000: Converged Physical Adherence ($|\mathcal{R}| \sim 10^{-4}$)")
ax2.set_xlabel("Spatial Coordinate $x$ (m)")
ax2.set_ylabel("Temporal Coordinate $t$ (s)")

fig.tight_layout()
fig.savefig(OUT / "physics_residual_field_2d.png")
plt.close(fig)
print("5. physics_residual_field_2d.png generated.")

# -------------------------------------------------------------------------
# Figure 6: soft_vs_hard_boundary_constraints.png
# Soft boundary penalties vs exact boundary ansatz
# -------------------------------------------------------------------------
x = np.linspace(0, 1, 200)
# Target problem: u(0) = 0, u(1) = 0. True solution: sin(pi*x)
u_true = np.sin(np.pi * x)

# Soft boundary constraint (small boundary leakage eps ~ 0.05)
u_soft = np.sin(np.pi * x) + 0.06 * (1.0 - x) - 0.04 * x
# Hard boundary constraint: u(x) = x(1-x)*N(x), exactly zero at x=0, 1
u_hard = np.sin(np.pi * x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(x, u_true, 'k--', linewidth=2, label="Exact Dirichlet Target $u(0)=u(1)=0$")
ax1.plot(x, u_soft, color="#e41a1c", linewidth=2.2, label=r"Soft Penalty Formulation ($\mathcal{L}_{\mathrm{bc}} > 0$)")
ax1.plot(x, u_hard, color="#2ca02c", linewidth=2.2, label=r"Hard Ansatz $u = x(1-x)\mathcal{N}_\theta(x)$")
ax1.scatter([0, 1], [u_soft[0], u_soft[-1]], color="#e41a1c", s=60, zorder=5, label="Boundary Leakage / Violation")
ax1.set_title("Boundary Value Adherence Comparison")
ax1.set_xlabel("Spatial Coordinate $x$")
ax1.set_ylabel("State Field $u(x)$")
ax1.legend(loc="lower center", framealpha=0.9)

# Boundary Error zoom-in
ax2.plot(x, np.abs(u_soft - u_true), color="#e41a1c", linewidth=2.2, label="Soft Penalty Absolute Error")
ax2.plot(x, np.abs(u_hard - u_true), color="#2ca02c", linewidth=2.2, label="Hard Ansatz Error (Exact Boundary)")
ax2.set_yscale("log")
ax2.set_ylim(1e-7, 1e0)
ax2.set_title("Point-Wise Spatial Error $|u(x) - u_{\mathrm{true}}(x)|$")
ax2.set_xlabel("Spatial Coordinate $x$")
ax2.set_ylabel("Absolute Error (Log Scale)")
ax2.legend(loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT / "soft_vs_hard_boundary_constraints.png")
plt.close(fig)
print("6. soft_vs_hard_boundary_constraints.png generated.")

# -------------------------------------------------------------------------
# Figure 7: gradient_stiffness_and_pathology.png
# Gradient norm trajectories: boundary loss gradient vs PDE residual gradient
# -------------------------------------------------------------------------
epochs = np.arange(1, 1001)
# Boundary gradients are orders of magnitude stiffer than PDE interior gradients
grad_norm_bc = 45.0 * np.exp(-epochs / 180.0) + 2.5 + 0.3 * rng.normal(0, 0.5, len(epochs))
grad_norm_pde = 0.45 * np.exp(-epochs / 500.0) + 0.05 + 0.01 * rng.normal(0, 0.5, len(epochs))
ratio = grad_norm_bc / np.maximum(grad_norm_pde, 1e-4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs, grad_norm_bc, color="#d95f02", linewidth=2.2, label=r"Boundary Gradient Norm $\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\|$")
ax1.plot(epochs, grad_norm_pde, color="#1f78b4", linewidth=2.2, label=r"PDE Residual Gradient Norm $\|\nabla_\theta \mathcal{L}_{\mathrm{pde}}\|$")
ax1.set_yscale("log")
ax1.set_xlabel("Training Epochs")
ax1.set_ylabel(r"Gradient Frobenius Norm $\|\nabla_\theta \mathcal{L}\|$")
ax1.set_title("Gradient Stiffness: Severe Scale Disparity across Loss Components")
ax1.legend(loc="upper right", framealpha=0.9)

ax2.plot(epochs, ratio, color="#7570b3", linewidth=2.2, label=r"Stiffness Ratio $\|\nabla_\theta \mathcal{L}_{\mathrm{bc}}\| / \|\nabla_\theta \mathcal{L}_{\mathrm{pde}}\|$")
ax2.axhline(1.0, color="k", linestyle="--", alpha=0.6, label="Balanced Gradient Ideal (Ratio = 1)")
ax2.set_yscale("log")
ax2.set_xlabel("Training Epochs")
ax2.set_ylabel("Gradient Norm Ratio (Stiffness Factor)")
ax2.set_title("Pathology: Optimization Dominated by Boundary Constraints")
ax2.legend(loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT / "gradient_stiffness_and_pathology.png")
plt.close(fig)
print("7. gradient_stiffness_and_pathology.png generated.")

# -------------------------------------------------------------------------
# Figure 8: learning_rate_annealing_convergence.png
# Loss trajectories: unweighted Adam vs dynamic Learning Rate Annealing (Wang et al., 2021)
# -------------------------------------------------------------------------
epochs = np.arange(1, 1501)
# Unweighted Adam: boundary drops fast, PDE residual plateaus high
loss_unweighted_bc = 1.0 * np.exp(-epochs / 120.0) + 1e-5
loss_unweighted_pde = 0.8 * np.exp(-epochs / 600.0) + 0.12 # stagnation

# Learning Rate Annealing: balanced convergence, PDE drops to 1e-5
loss_lra_bc = 1.0 * np.exp(-epochs / 180.0) + 2e-6
loss_lra_pde = 0.8 * np.exp(-epochs / 150.0) + 3e-6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

ax1.plot(epochs, loss_unweighted_bc, color="#fc8d59", linestyle="--", linewidth=2, label=r"Boundary Loss $\mathcal{L}_{\mathrm{bc}}$")
ax1.plot(epochs, loss_unweighted_pde, color="#d73027", linewidth=2.4, label=r"PDE Residual Loss $\mathcal{L}_{\mathrm{pde}}$ (Stagnation)")
ax1.set_yscale("log")
ax1.set_xlabel("Training Epochs")
ax1.set_ylabel("Loss Magnitude")
ax1.set_title("Unweighted Loss: PDE Residual Failure")
ax1.set_ylim(1e-6, 2e0)
ax1.legend(loc="upper right", framealpha=0.9)

ax2.plot(epochs, loss_lra_bc, color="#91bfdb", linestyle="--", linewidth=2, label=r"Boundary Loss $\mathcal{L}_{\mathrm{bc}}$")
ax2.plot(epochs, loss_lra_pde, color="#4575b4", linewidth=2.4, label=r"PDE Residual Loss $\mathcal{L}_{\mathrm{pde}}$ (Balanced)")
ax2.set_yscale("log")
ax2.set_xlabel("Training Epochs")
ax2.set_ylabel("Loss Magnitude")
ax2.set_title("Learning Rate Annealing (Wang et al., 2021): Balanced Convergence")
ax2.set_ylim(1e-6, 2e0)
ax2.legend(loc="upper right", framealpha=0.9)

fig.suptitle("Mitigating Optimization Pathologies via Dynamic Multi-Objective Loss Balancing", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "learning_rate_annealing_convergence.png")
plt.close(fig)
print("8. learning_rate_annealing_convergence.png generated.")

# -------------------------------------------------------------------------
# Figure 9: discrete_mesh_vs_meshfree_collocation.png
# Structural comparison: Classical irregular FEM mesh vs PINN collocation cloud
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Triangular FEM mesh on irregular airfoil / circular domain
theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
r_inner = 0.5
r_outer = 1.5
inner_pts = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
outer_pts = np.column_stack([r_outer * np.cos(theta), r_outer * np.sin(theta)])
mid_pts = np.column_stack([1.0 * np.cos(theta + 0.1), 1.0 * np.sin(theta + 0.1)])
all_fem_pts = np.vstack([inner_pts, mid_pts, outer_pts])
tri = scipy.spatial.Delaunay(all_fem_pts)

ax1.triplot(all_fem_pts[:, 0], all_fem_pts[:, 1], tri.simplices, color="#2b8cbe", lw=1.2)
ax1.plot(all_fem_pts[:, 0], all_fem_pts[:, 1], 'o', color="#08519c", markersize=4)
ax1.set_title("Classical Finite Element Method (FEM)\nConformal Discretization Mesh", fontsize=12)
ax1.set_xlabel("Spatial Coordinate $x$ (m)")
ax1.set_ylabel("Spatial Coordinate $y$ (m)")
ax1.set_aspect("equal")

# Gridless PINN Collocation Cloud
colloc_r = rng.uniform(r_inner, r_outer, 250)
colloc_th = rng.uniform(0, 2*np.pi, 250)
pinn_pts_x = colloc_r * np.cos(colloc_th)
pinn_pts_y = colloc_r * np.sin(colloc_th)

circle_in = plt.Circle((0, 0), r_inner, color='k', fill=False, linestyle="--", linewidth=1.5)
circle_out = plt.Circle((0, 0), r_outer, color='k', fill=False, linestyle="--", linewidth=1.5)
ax2.add_patch(circle_in)
ax2.add_patch(circle_out)
ax2.scatter(pinn_pts_x, pinn_pts_y, color="#e41a1c", s=30, alpha=0.75, edgecolors='k', lw=0.5, label="Collocation Points")
ax2.set_title("Physics-Informed Neural Network (PINN)\nGridless Mesh-Free Collocation Cloud", fontsize=12)
ax2.set_xlabel("Spatial Coordinate $x$ (m)")
ax2.set_ylabel("Spatial Coordinate $y$ (m)")
ax2.set_aspect("equal")
ax2.legend(loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT / "discrete_mesh_vs_meshfree_collocation.png")
plt.close(fig)
print("9. discrete_mesh_vs_meshfree_collocation.png generated.")

# -------------------------------------------------------------------------
# Figure 10: activation_smoothness_for_higher_order_pde.png
# 1st, 2nd, 3rd derivatives of tanh, SIREN (sin), and GELU
# -------------------------------------------------------------------------
z = np.linspace(-3, 3, 300)

# Tanh derivatives
tanh_0 = np.tanh(z)
tanh_1 = 1.0 - tanh_0**2
tanh_2 = -2.0 * tanh_0 * tanh_1
tanh_3 = -2.0 * (tanh_1**2 - 2 * tanh_0**2 * tanh_1)

# SIREN sin(omega*z) with omega=1
siren_0 = np.sin(z)
siren_1 = np.cos(z)
siren_2 = -np.sin(z)
siren_3 = -np.cos(z)

# GELU numerical derivatives
from scipy.special import erf
gelu_0 = 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
gelu_1 = np.gradient(gelu_0, z)
gelu_2 = np.gradient(gelu_1, z)
gelu_3 = np.gradient(gelu_2, z)

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
deriv_titles = ["Base Activation $\sigma(z)$", "1st Derivative $\sigma'(z)$", "2nd Derivative $\sigma''(z)$", "3rd Derivative $\sigma'''(z)$"]
for idx, ax in enumerate(axes):
    if idx == 0:
        ax.plot(z, tanh_0, label=r"Hyperbolic Tangent $\tanh$", color="#2b8cbe", lw=2.2)
        ax.plot(z, siren_0, label="SIREN $\sin(z)$", color="#31a354", lw=2.2)
        ax.plot(z, gelu_0, label="GELU", color="#e41a1c", lw=2.2)
    elif idx == 1:
        ax.plot(z, tanh_1, color="#2b8cbe", lw=2.2)
        ax.plot(z, siren_1, color="#31a354", lw=2.2)
        ax.plot(z, gelu_1, color="#e41a1c", lw=2.2)
    elif idx == 2:
        ax.plot(z, tanh_2, color="#2b8cbe", lw=2.2)
        ax.plot(z, siren_2, color="#31a354", lw=2.2)
        ax.plot(z, gelu_2, color="#e41a1c", lw=2.2)
    elif idx == 3:
        ax.plot(z, tanh_3, color="#2b8cbe", lw=2.2)
        ax.plot(z, siren_3, color="#31a354", lw=2.2)
        ax.plot(z, gelu_3, color="#e41a1c", lw=2.2)
    ax.set_title(deriv_titles[idx], fontsize=12)
    ax.set_xlabel("Latent Coordinate $z$")
    ax.set_xlim(-3, 3)

axes[0].set_ylabel("Activation Amplitude")
axes[0].legend(loc="lower right", framealpha=0.9)
fig.suptitle("Activation Function Differentiability and Higher-Order Coordinate Derivatives in SciML", fontsize=14, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "activation_smoothness_for_higher_order_pde.png")
plt.close(fig)
print("10. activation_smoothness_for_higher_order_pde.png generated.")

# -------------------------------------------------------------------------
# Figure 11: rad_adaptive_refinement_evolution.png
# Spatial point redistribution across 3 training stages
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
N = 120

# Stage 1: Initial LHS (Uniform coverage)
p1 = lhs_2d(N, rng)

# Stage 2: Intermediate refinement (Points start moving to shock near x=0.5)
p2_x = np.concatenate([rng.normal(0.5, 0.15, int(N*0.5)), rng.uniform(0, 1, int(N*0.5))])
p2_y = rng.uniform(0, 1, len(p2_x))
p2 = np.column_stack([np.clip(p2_x, 0.01, 0.99), p2_y])

# Stage 3: Fully converged RAD (Heavy clustering along shock x=0.5)
p3_x = np.concatenate([rng.normal(0.5, 0.04, int(N*0.75)), rng.uniform(0, 1, int(N*0.25))])
p3_y = rng.uniform(0, 1, len(p3_x))
p3 = np.column_stack([np.clip(p3_x, 0.01, 0.99), p3_y])

stages = [
    (p1, "Stage 1 (Epoch 0): Uniform LHS Baseline", "#2b8cbe"),
    (p2, "Stage 2 (Epoch 500): Residual Gradient Migration", "#f59b32"),
    (p3, "Stage 3 (Epoch 1500): Converged Shock Clustering (RAD)", "#e41a1c")
]

for ax, (pts, title, col) in zip(axes, stages):
    ax.scatter(pts[:, 0], pts[:, 1], color=col, s=40, edgecolors='k', alpha=0.85)
    ax.axvline(0.5, color="k", linestyle=":", alpha=0.4, label="Physical Shock Front $x=0.5$")
    ax.set_title(title, fontsize=11.5)
    ax.set_xlabel("Spatial Coordinate $x$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Temporal Coordinate $t$")
axes[0].legend(loc="lower left", framealpha=0.85)
fig.suptitle("Residual-Based Adaptive Distribution (RAD): Dynamic Collocation Point Migration", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "rad_adaptive_refinement_evolution.png")
plt.close(fig)
print("11. rad_adaptive_refinement_evolution.png generated.")

# -------------------------------------------------------------------------
# Figure 12: week3_sciml_taxonomy_map.png
# Comprehensive classification map across Physics Knowledge and Data Availability
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Quadrant boundaries
ax.axvline(5.0, color="#cccccc", linestyle="--", lw=1.5)
ax.axhline(5.0, color="#cccccc", linestyle="--", lw=1.5)

# Background quadrant shading
ax.fill_between([0, 5], 5, 10, color="#e5f5f9", alpha=0.4) # High Physics, Low Data
ax.fill_between([5, 10], 5, 10, color="#e0f3db", alpha=0.4) # High Physics, High Data
ax.fill_between([0, 5], 0, 5, color="#fee8c8", alpha=0.3) # Low Physics, Low Data
ax.fill_between([5, 10], 0, 5, color="#fee0d2", alpha=0.4) # Low Physics, High Data

# Quadrant labels
ax.text(2.5, 9.5, "Theory-Dominated Regime", ha="center", fontsize=11, fontweight="bold", color="#006d2c")
ax.text(7.5, 9.5, "Synergistic SciML Frontier", ha="center", fontsize=11, fontweight="bold", color="#006d2c")
ax.text(2.5, 0.5, "Under-Determined / Ill-Posed", ha="center", fontsize=11, fontweight="bold", color="#7f2704")
ax.text(7.5, 0.5, "Pure Data-Driven Black-Box", ha="center", fontsize=11, fontweight="bold", color="#a50f15")

def plot_method(ax, x, y, label, desc, color):
    ax.scatter(x, y, color=color, s=160, edgecolors='k', zorder=5)
    ax.text(x, y + 0.35, label, ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#111111")
    ax.text(x, y - 0.45, desc, ha="center", va="top", fontsize=9, color="#444444")

plot_method(ax, 1.8, 8.2, "Classical Solvers (FEM / FVM)", "Governing PDEs exact;\nzero empirical data required", "#2b8cbe")
plot_method(ax, 3.2, 6.8, "Forward PINNs", "Mesh-free PDE surrogate;\nzero interior data needed", "#31a354")
plot_method(ax, 7.8, 7.8, "Inverse PINNs / SciML", "Sparse sensor fusion +\ngoverning conservation laws", "#756bb1")
plot_method(ax, 8.2, 6.0, "Neural Operators (FNO / DeepONet)", "Learns infinite-dim operator maps;\nhigh data & simulation demand", "#d95f0e")
plot_method(ax, 8.5, 2.2, "Deep Learning (MLP / CNN / ResNet)", "Standard black-box curve fitting;\nviolates physical conservation", "#e41a1c")

ax.set_xlabel("Empirical Data Availability $\longrightarrow$", fontsize=12, fontweight="bold")
ax.set_ylabel("First-Principles Physical Knowledge $\longrightarrow$", fontsize=12, fontweight="bold")
ax.set_title("Scientific Machine Learning (SciML) Methodological Taxonomy Map", fontsize=13.5, pad=15)
ax.set_xticks([1, 5, 9])
ax.set_xticklabels(["Zero / Sparse Data", "Moderate Probes", "Massive Data Regimes"])
ax.set_yticks([1, 5, 9])
ax.set_yticklabels(["Empirical / Black-Box", "Incomplete Physics", "Exact Conservation Laws"])

fig.tight_layout()
fig.savefig(OUT / "week3_sciml_taxonomy_map.png")
plt.close(fig)
print("12. week3_sciml_taxonomy_map.png generated.")
print("All 12 Week 3 visual assets generated successfully!")
