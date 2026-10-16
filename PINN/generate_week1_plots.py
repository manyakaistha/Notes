import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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

assets_dir = Path("month 1/week 1/assets")
assets_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path("data")

print("=== Generating Week 1 Publication Figures (12 Figures) ===")

# -------------------------------------------------------------
# Figure 1: functions_and_derivatives.png
# -------------------------------------------------------------
print("Generating Figure 1: functions_and_derivatives.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.linspace(-2.2, 2.2, 400)
f = x**3 - 3*x
df = 3*x**2 - 3
x0 = 0.5
y0 = x0**3 - 3*x0
m_tangent = 3*x0**2 - 3

ax1.plot(x, f, color='#1f77b4', lw=2.5, label=r"$f(x) = x^3 - 3x$")
ax1.plot(x, y0 + m_tangent*(x - x0), color='#d62728', lw=2.0, linestyle='--', label=f"True Tangent at $x_0=0.5$ (slope={m_tangent:.2f})")

deltas = [1.2, 0.6, 0.2]
colors = ['#ff7f0e', '#2ca02c', '#9467bd']
for dx, c in zip(deltas, colors):
    x1 = x0 + dx
    y1 = x1**3 - 3*x1
    m_sec = (y1 - y0) / dx
    x_sec = np.linspace(-0.5, 2.0, 100)
    ax1.plot(x_sec, y0 + m_sec*(x_sec - x0), color=c, lw=1.5, linestyle=':', label=f"Secant $\Delta x={dx}$ (slope={m_sec:.2f})")
    ax1.scatter([x1], [y1], color=c, s=40, zorder=5)

ax1.scatter([x0], [y0], color='black', s=60, zorder=6, label=r"Pivot Point $(x_0, f(x_0))$")
ax1.set_title("Secant-to-Tangent Convergence ($f(x) = x^3 - 3x$)")
ax1.set_xlabel("Coordinate $x$")
ax1.set_ylabel("$f(x)$")
ax1.set_ylim(-4, 4)
ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

ax2.plot(x, df, color='#2ca02c', lw=2.5, label=r"Analytical Derivative $f'(x) = 3x^2 - 3$")
ax2.axhline(0, color='gray', linestyle='--', lw=1)
ax2.scatter([x0], [m_tangent], color='#d62728', s=60, zorder=5, label=f"$f'(0.5) = {m_tangent:.2f}$")
ax2.set_title("Continuous Derivative Field $f'(x)$")
ax2.set_xlabel("Coordinate $x$")
ax2.set_ylabel("$f'(x)$")
ax2.legend(loc="upper center", framealpha=0.9)

plt.tight_layout()
fig.savefig(assets_dir / "functions_and_derivatives.png")
plt.close()

# -------------------------------------------------------------
# Figure 2: gradient_vector_field.png
# -------------------------------------------------------------
print("Generating Figure 2: gradient_vector_field.png...")
fig, ax = plt.subplots(figsize=(8, 6.5))
xg = np.linspace(-2, 2, 250)
yg = np.linspace(-2, 2, 250)
X, Y = np.meshgrid(xg, yg)
Phi = X**2 + 2*Y**2  # Anisotropic elliptic potential

cs = ax.contour(X, Y, Phi, levels=12, cmap='viridis', linewidths=1.8)
ax.clabel(cs, inline=True, fontsize=9, fmt="%.1f")

# Quiver arrows
xq = np.linspace(-1.8, 1.8, 16)
yq = np.linspace(-1.8, 1.8, 16)
XQ, YQ = np.meshgrid(xq, yq)
U_grad = 2 * XQ
V_grad = 4 * YQ
norm = np.sqrt(U_grad**2 + V_grad**2) + 1e-8
U_norm = U_grad / norm
V_norm = V_grad / norm

ax.quiver(XQ, YQ, U_norm, V_norm, color='#d62728', scale=25, width=0.004, headwidth=4, headlength=5, label=r"Normalized Gradient $\nabla \Phi / \|\nabla \Phi\|$")
ax.set_title(r"Scalar Field $\Phi(x, y) = x^2 + 2y^2$ & Gradient Vector Field $\nabla \Phi$")
ax.set_xlabel("Spatial Coordinate $x$")
ax.set_ylabel("Spatial Coordinate $y$")
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
fig.savefig(assets_dir / "gradient_vector_field.png")
plt.close()

# -------------------------------------------------------------
# Figure 3: ode_decay_and_phase.png
# -------------------------------------------------------------
print("Generating Figure 3: ode_decay_and_phase.png...")
df_osc = pd.read_csv(data_dir / "week1_harmonic_oscillator_ivp.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(df_osc["time_s"], df_osc["u_underdamped_m"], color='#1f77b4', lw=2.2, label=r"Underdamped ($\zeta=0.1, c=0.4$)")
ax1.plot(df_osc["time_s"], df_osc["u_critically_damped_m"], color='#2ca02c', lw=2.2, label=r"Critically Damped ($\zeta=1.0, c=4.0$)")
ax1.plot(df_osc["time_s"], df_osc["u_overdamped_m"], color='#ff7f0e', lw=2.2, label=r"Overdamped ($\zeta=2.0, c=8.0$)")
ax1.axhline(0, color='gray', linestyle='--', lw=1)
ax1.set_title("Harmonic Oscillator Time Series $u(t)$")
ax1.set_xlabel("Time $t$ [s]")
ax1.set_ylabel("Displacement $u(t)$ [m]")
ax1.legend(loc="upper right", framealpha=0.9)

ax2.plot(df_osc["u_underdamped_m"], df_osc["v_underdamped_mps"], color='#1f77b4', lw=1.8, label="Underdamped Spiral")
ax2.plot(df_osc["u_critically_damped_m"], df_osc["v_critically_damped_mps"], color='#2ca02c', lw=2.2, label="Critically Damped Path")
ax2.plot(df_osc["u_overdamped_m"], df_osc["v_overdamped_mps"], color='#ff7f0e', lw=2.2, label="Overdamped Path")
ax2.scatter([1.0], [0.0], color='red', s=60, zorder=5, label=r"Initial State $(u_0=1, v_0=0)$")
ax2.scatter([0.0], [0.0], color='black', s=60, zorder=5, label="Equilibrium Sink $(0, 0)$")
ax2.set_title("Phase Space Portrait $(u, \dot{u})$")
ax2.set_xlabel("Displacement $u$ [m]")
ax2.set_ylabel(r"Velocity $\dot{u}$ [m/s]")
ax2.legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
fig.savefig(assets_dir / "ode_decay_and_phase.png")
plt.close()

# -------------------------------------------------------------
# Figure 4: mechanical_energy_dissipation.png
# -------------------------------------------------------------
print("Generating Figure 4: mechanical_energy_dissipation.png...")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(df_osc["time_s"], df_osc["energy_underdamped_J"], color='#1f77b4', lw=2.5, label=r"Underdamped ($\zeta=0.1$)")
ax.plot(df_osc["time_s"], df_osc["energy_critically_damped_J"], color='#2ca02c', lw=2.5, label=r"Critically Damped ($\zeta=1.0$)")
ax.plot(df_osc["time_s"], df_osc["energy_overdamped_J"], color='#ff7f0e', lw=2.5, label=r"Overdamped ($\zeta=2.0$)")
ax.set_yscale('log')
ax.set_title(r"Mechanical Energy Dissipation $E(t) = \frac{1}{2}m\dot{u}^2 + \frac{1}{2}ku^2$")
ax.set_xlabel("Time $t$ [s]")
ax.set_ylabel("Total Mechanical Energy $E(t)$ [Joules] (Log Scale)")
ax.set_ylim(1e-6, 3.0)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
fig.savefig(assets_dir / "mechanical_energy_dissipation.png")
plt.close()

# -------------------------------------------------------------
# Figure 5: pde_classification_domains.png
# -------------------------------------------------------------
print("Generating Figure 5: pde_classification_domains.png...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.0))

# Elliptic: Laplace equation
xe = np.linspace(-1, 1, 100)
ye = np.linspace(-1, 1, 100)
XE, YE = np.meshgrid(xe, ye)
ZE = np.sin(np.pi*XE) * np.sinh(np.pi*(YE+1)) / np.sinh(2*np.pi)
c1 = ax1.contourf(XE, YE, ZE, levels=20, cmap='coolwarm')
ax1.set_title("Elliptic: Poisson/Laplace\n($B^2 - 4AC < 0$)")
ax1.set_xlabel("Spatial Coordinate $x$")
ax1.set_ylabel("Spatial Coordinate $y$")
fig.colorbar(c1, ax=ax1, orientation='horizontal', pad=0.18, shrink=0.8)

# Parabolic: Heat equation
xp = np.linspace(0, 1, 100)
tp = np.linspace(0, 0.5, 100)
TP, XP = np.meshgrid(tp, xp)
ZP = np.exp(-0.1 * np.pi**2 * TP) * np.sin(np.pi * XP)
c2 = ax2.contourf(XP, TP, ZP, levels=20, cmap='inferno')
ax2.set_title("Parabolic: Heat Equation\n($B^2 - 4AC = 0$)")
ax2.set_xlabel("Spatial Coordinate $x$")
ax2.set_ylabel("Time $t$ [s]")
fig.colorbar(c2, ax=ax2, orientation='horizontal', pad=0.18, shrink=0.8)

# Hyperbolic: Wave equation characteristics
xh = np.linspace(-2, 2, 200)
th = np.linspace(0, 2, 200)
TH, XH = np.meshgrid(th, xh)
c_wave = 1.0
ax3.plot([0, 1.5], [0, 1.5/c_wave], color='#d62728', lw=2.5, label="Right Characteristic $x - ct = x_0$")
ax3.plot([0, -1.5], [0, 1.5/c_wave], color='#1f77b4', lw=2.5, label="Left Characteristic $x + ct = x_0$")
ax3.fill_betweenx([0, 1.5/c_wave], [-1.5, 0], [1.5, 0], color='gray', alpha=0.25, label="Domain of Dependence")
ax3.scatter([0], [0], color='black', s=80, zorder=5, label="Disturbance Source $(0, 0)$")
ax3.set_title("Hyperbolic: Wave Equation\n($B^2 - 4AC > 0$)")
ax3.set_xlabel("Spatial Coordinate $x$")
ax3.set_ylabel("Time $t$ [s]")
ax3.set_xlim(-2, 2)
ax3.set_ylim(0, 2)
ax3.legend(loc="upper center", fontsize=8.5, framealpha=0.9)

plt.tight_layout()
fig.savefig(assets_dir / "pde_classification_domains.png")
plt.close()

# -------------------------------------------------------------
# Figure 6: pde_heat_space_time.png
# -------------------------------------------------------------
print("Generating Figure 6: pde_heat_space_time.png...")
U_heat = np.load(data_dir / "week1_heat_diffusion_analytical.npy")
fig, ax = plt.subplots(figsize=(8.5, 6))
Nx, Nt = U_heat.shape
x_coords = np.linspace(0, 1.0, Nx)
t_coords = np.linspace(0, 0.5, Nt)
c = ax.pcolormesh(t_coords, x_coords, U_heat, shading='auto', cmap='plasma')
cb = fig.colorbar(c, ax=ax)
cb.set_label("Field Value $u(t, x)$")
ax.set_title(r"2D Space-Time Thermal Diffusion $u(t, x) = e^{-\alpha \pi^2 t} \sin(\pi x)$")
ax.set_xlabel("Time $t$ [s]")
ax.set_ylabel("Spatial Position $x$ [m]")
plt.tight_layout()
fig.savefig(assets_dir / "pde_heat_space_time.png")
plt.close()

# -------------------------------------------------------------
# Figure 7: boundary_condition_types.png
# -------------------------------------------------------------
print("Generating Figure 7: boundary_condition_types.png...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.8))
x_rod = np.linspace(0, 1, 100)

# Dirichlet: u(0) = 0, u(1) = 1
u_dir = x_rod
ax1.plot(x_rod, u_dir, color='#1f77b4', lw=2.5, label="Steady Profile $u(x)$")
ax1.scatter([0, 1], [0, 1], color='#d62728', s=70, zorder=5, label="Dirichlet Clamped Pins")
ax1.set_title("Dirichlet (Essential)\n$u(0)=0, u(1)=1$")
ax1.set_xlabel("Position $x$")
ax1.set_ylabel("State $u(x)$")
ax1.legend(loc="upper left")

# Neumann: u'(0) = 0 (insulated), u(1) = 1
u_neu = 1.0 - 0.5*(1.0 - x_rod)**2  # u'(0) = (1-0) = 1? Let's do u(x) = 1 - (1-x)^2 -> u'(x) = 2(1-x), u'(1)=0.
# u'(0) = 0 -> u(x) = x^2, u'(0) = 0, u(1) = 1
u_neu = x_rod**2
ax2.plot(x_rod, u_neu, color='#2ca02c', lw=2.5, label="Steady Profile $u(x) = x^2$")
ax2.scatter([0], [0], color='#ff7f0e', s=70, zorder=5, label=r"Neumann Zero Flux $u'(0)=0$")
ax2.set_title("Neumann (Natural / Flux)\n$u'(0) = 0$ (Insulated)")
ax2.set_xlabel("Position $x$")
ax2.set_ylabel("State $u(x)$")
ax2.legend(loc="upper left")

# Robin: u'(0) - h u(0) = 0
# u(x) = (x + 1)/(2), u'(0) = 1, u(0) = 0.5 -> u'(0) - 2*u(0) = 0
u_rob = (x_rod + 1.0) / 2.0
ax3.plot(x_rod, u_rob, color='#9467bd', lw=2.5, label="Profile $u(x)$")
ax3.scatter([0], [0.5], color='#d62728', s=70, zorder=5, label=r"Robin Convective $u'-hu=g$")
ax3.set_title("Robin (Mixed / Radiation)\n$u'(0) - h u(0) = g$")
ax3.set_xlabel("Position $x$")
ax3.set_ylabel("State $u(x)$")
ax3.legend(loc="upper left")

plt.tight_layout()
fig.savefig(assets_dir / "boundary_condition_types.png")
plt.close()

# -------------------------------------------------------------
# Figure 8: discretization_and_residuals.png
# -------------------------------------------------------------
print("Generating Figure 8: discretization_and_residuals.png...")
df_fd = pd.read_csv(data_dir / "week1_fd_convergence_table.csv")
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(df_fd["step_size_h"], df_fd["forward_diff_err_float64"], color='#ff7f0e', lw=2.2, label=r"Forward Diff Float64 $\mathcal{O}(h)$")
ax.plot(df_fd["step_size_h"], df_fd["central_diff_err_float64"], color='#1f77b4', lw=2.2, label=r"Central Diff Float64 $\mathcal{O}(h^2)$")
ax.plot(df_fd["step_size_h"], df_fd["second_diff_err_float64"], color='#2ca02c', lw=2.2, label=r"Laplacian Central Float64 $\mathcal{O}(h^2)$")
ax.plot(df_fd["step_size_h"], df_fd["central_diff_err_float32"], color='#d62728', lw=1.8, linestyle='--', label=r"Central Diff Float32 (cancellation at $h \approx 10^{-4}$)")

ax.axvline(1e-8, color='black', linestyle=':', lw=1.5, label=r"Float64 Machine Limit $h^* \approx \sqrt{\epsilon} \approx 10^{-8}$")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Discretization Error vs. Floating-Point Roundoff Cancellation")
ax.set_xlabel("Discretization Step Size $h$ (Log Scale)")
ax.set_ylabel("Absolute Derivative Error (Log Scale)")
ax.set_ylim(1e-13, 1e2)
ax.legend(loc="upper center", framealpha=0.9, fontsize=9)
plt.tight_layout()
fig.savefig(assets_dir / "discretization_and_residuals.png")
plt.close()

# -------------------------------------------------------------
# Figure 9: curse_of_dimensionality_grids.png
# -------------------------------------------------------------
print("Generating Figure 9: curse_of_dimensionality_grids.png...")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
N_per_dim = np.linspace(10, 100, 10)
ax.plot(N_per_dim, N_per_dim, marker='o', lw=2, label="1D Grid: $N$ points")
ax.plot(N_per_dim, N_per_dim**2, marker='s', lw=2, label=r"2D Grid: $N^2$ points")
ax.plot(N_per_dim, N_per_dim**3, marker='^', lw=2, label=r"3D Grid: $N^3$ points")
ax.plot(N_per_dim, N_per_dim**4, marker='d', lw=2, label=r"4D Space-Time Grid: $N^4$ points")
ax.axhline(1e6, color='#d62728', linestyle='--', label="1 Million Node Threshold")
ax.set_yscale('log')
ax.set_title("The Curse of Dimensionality in Discrete Grid Stencils")
ax.set_xlabel("Points Per Dimension $N$")
ax.set_ylabel("Total Node Count (Log Scale)")
ax.legend(loc="upper left", framealpha=0.9)
plt.tight_layout()
fig.savefig(assets_dir / "curse_of_dimensionality_grids.png")
plt.close()

# -------------------------------------------------------------
# Figure 10: continuous_vs_discrete_residual.png
# -------------------------------------------------------------
print("Generating Figure 10: continuous_vs_discrete_residual.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Discrete stencil
x_discrete = np.linspace(0, 1, 11)
y_discrete = np.sin(np.pi * x_discrete)
ax1.plot(x_discrete, y_discrete, 'o--', color='#1f77b4', lw=2, markersize=8, label="Grid Nodes $x_i$")
ax1.annotate("Local Stencil:\n$[u_{i-1} - 2u_i + u_{i+1}] / h^2$", xy=(x_discrete[5], y_discrete[5]), xytext=(0.2, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6))
ax1.set_title("Discrete Grid Stencil Residual\n(Defined only at discrete nodes)")
ax1.set_xlabel("Coordinate $x$")
ax1.set_ylabel("Discretized State $u_i$")
ax1.legend(loc="upper right")

# Continuous differential residual
x_cont = np.linspace(0, 1, 200)
u_exact = np.sin(np.pi * x_cont)
u_pert = u_exact + 0.05 * np.sin(3 * np.pi * x_cont)
res_cont = abs(-np.pi**2 * u_pert - 9 * np.pi**2 * 0.05 * np.sin(3 * np.pi * x_cont) + np.pi**2 * u_exact)
ax2.plot(x_cont, res_cont, color='#d62728', lw=2.2, label=r"Continuous Residual $|\mathcal{R}(x)| = |\nabla^2 u - f|$")
ax2.fill_between(x_cont, 0, res_cont, color='#d62728', alpha=0.25)
ax2.set_title("Continuous Differential Residual Field\n(Evaluated anywhere in domain $\Omega$)")
ax2.set_xlabel("Continuous Coordinate $x$")
ax2.set_ylabel(r"$|\mathcal{R}(x)|$")
ax2.legend(loc="upper right")

plt.tight_layout()
fig.savefig(assets_dir / "continuous_vs_discrete_residual.png")
plt.close()

# -------------------------------------------------------------
# Figure 11: ill_posed_boundary_divergence.png
# -------------------------------------------------------------
print("Generating Figure 11: ill_posed_boundary_divergence.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
x_b = np.linspace(0, 1, 100)

# Under-constrained: u''(x) = 0 with only u(0) = 1 (family of solutions u(x) = 1 + C*x)
for C, col in zip([-2, -1, 0, 1, 2], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']):
    ax1.plot(x_b, 1.0 + C*x_b, color=col, lw=2, label=f"Solution $C={C}$")
ax1.scatter([0], [1], color='black', s=80, zorder=5, label="Single Dirichlet Pin $u(0)=1$")
ax1.set_title("Ill-Posed: Non-Unique Infinitude\n($u''(x)=0$ with only $u(0)=1$)")
ax1.set_xlabel("Coordinate $x$")
ax1.set_ylabel("$u(x)$")
ax1.legend(loc="upper left", fontsize=8.5)

# Well-posed: u(0) = 1 and u(1) = 3 (unique line u(x) = 1 + 2x)
ax2.plot(x_b, 1.0 + 2.0*x_b, color='#2ca02c', lw=3, label="Unique Solution $u(x) = 1 + 2x$")
ax2.scatter([0, 1], [1, 3], color='#d62728', s=80, zorder=5, label="Complete Dirichlet Pins: $u(0)=1, u(1)=3$")
ax2.set_title("Well-Posed (Hadamard Satisfied)\nUnique, Stable Solution")
ax2.set_xlabel("Coordinate $x$")
ax2.set_ylabel("$u(x)$")
ax2.legend(loc="upper left")

plt.tight_layout()
fig.savefig(assets_dir / "ill_posed_boundary_divergence.png")
plt.close()

# -------------------------------------------------------------
# Figure 12: taylor_series_polynomial_approximations.png
# -------------------------------------------------------------
print("Generating Figure 12: taylor_series_polynomial_approximations.png...")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
xt = np.linspace(-3.5, 3.5, 300)
y_cos = np.cos(xt)

# Taylor polynomials around x0 = 0:
P0 = np.ones_like(xt)
P2 = 1.0 - (xt**2)/2.0
P4 = 1.0 - (xt**2)/2.0 + (xt**4)/24.0
P6 = 1.0 - (xt**2)/2.0 + (xt**4)/24.0 - (xt**6)/720.0

ax.plot(xt, y_cos, color='black', lw=3.0, label=r"True Function $\cos(x)$")
ax.plot(xt, P0, color='#1f77b4', lw=1.8, linestyle='--', label=r"Order 0: $P_0(x) = 1$")
ax.plot(xt, P2, color='#ff7f0e', lw=2.0, linestyle='--', label=r"Order 2: $P_2(x) = 1 - x^2/2$")
ax.plot(xt, P4, color='#2ca02c', lw=2.0, linestyle='-.', label=r"Order 4: $P_4(x) = 1 - x^2/2 + x^4/24$")
ax.plot(xt, P6, color='#d62728', lw=2.0, linestyle=':', label=r"Order 6: $P_6(x) = 1 - x^2/2 + x^4/24 - x^6/720$")

ax.set_title(r"Taylor Polynomial Approximations of $\cos(x)$ Around $x_0 = 0$")
ax.set_xlabel("Coordinate $x$")
ax.set_ylabel("$P_n(x)$")
ax.set_ylim(-2.0, 2.5)
ax.legend(loc="lower center", framealpha=0.9, fontsize=9.5)
plt.tight_layout()
fig.savefig(assets_dir / "taylor_series_polynomial_approximations.png")
plt.close()

print("All 12 Week 1 figures successfully generated!")
