"""Generate all publication-grade figures for PINN Week 1."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

OUT = Path("month 1/week 1/assets")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. Functions & Derivatives
x = np.linspace(-2.5, 2.5, 500)
f = x**3 - 3*x
df = 3*x**2 - 3

x0 = 1.0
y0 = x0**3 - 3*x0
dx_vals = [1.2, 0.6, 0.15]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(x, f, 'b-', lw=2.5, label=r"$f(x) = x^3 - 3x$")
ax1.plot(x0, y0, 'ro', markersize=8, label=r"Point of tangency $(x_0, f(x_0))$")

for dx in dx_vals:
    x1 = x0 + dx
    y1 = x1**3 - 3*x1
    m = (y1 - y0) / dx
    x_sec = np.linspace(x0 - 0.5, x1 + 0.5, 50)
    y_sec = y0 + m * (x_sec - x0)
    ax1.plot(x_sec, y_sec, '--', alpha=0.75, label=rf"Secant $\Delta x={dx}$ (m={m:.2f})")

m_tan = 3*x0**2 - 3
x_tan = np.linspace(x0 - 1.0, x0 + 1.0, 50)
y_tan = y0 + m_tan * (x_tan - x0)
ax1.plot(x_tan, y_tan, 'k-', lw=2, label=rf"Tangent $f'(1.0)={m_tan:.1f}$")
ax1.set_title(r"Limit of Difference Quotient $\lim_{\Delta x \to 0} \frac{\Delta f}{\Delta x}$", fontsize=14)
ax1.set_xlabel("x")
ax1.set_ylabel("f(x)")
ax1.set_ylim(-6, 6)
ax1.legend(fontsize=9, loc="upper left")

ax2.plot(x, df, 'r-', lw=2.5, label=r"Analytical $f'(x) = 3x^2 - 3$")
h_test = 0.4
x_sample = np.linspace(-2.2, 2.2, 12)
f_sample = x_sample**3 - 3*x_sample
fd_central = ((x_sample + h_test)**3 - 3*(x_sample + h_test) - ((x_sample - h_test)**3 - 3*(x_sample - h_test))) / (2 * h_test)
ax2.scatter(x_sample, fd_central, color='darkgreen', s=60, zorder=5, label=rf"Central Difference ($h={h_test}$)")
ax2.set_title("Exact Derivative vs. Numerical Approximation", fontsize=14)
ax2.set_xlabel("x")
ax2.set_ylabel("f'(x)")
ax2.legend(fontsize=11, loc="upper center")

fig.suptitle("Differential Calculus Foundations: Tangents, Secants, and Rate of Change", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "functions_and_derivatives.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. Gradient Vector Field
x1 = np.linspace(-2.0, 2.0, 100)
x2 = np.linspace(-2.0, 2.0, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * X1**2 + X2**2 - 0.3 * np.cos(3*np.pi*X1)

# Gradient grid
x1_sub = np.linspace(-1.8, 1.8, 15)
x2_sub = np.linspace(-1.8, 1.8, 15)
X1_sub, X2_sub = np.meshgrid(x1_sub, x2_sub)
grad_X1 = X1_sub + 0.3 * 3 * np.pi * np.sin(3*np.pi*X1_sub)
grad_X2 = 2 * X2_sub

norm = np.hypot(grad_X1, grad_X2)
norm[norm == 0] = 1.0
u_norm = grad_X1 / norm
v_norm = grad_X2 / norm

fig, ax = plt.subplots(figsize=(8, 7))
cp = ax.contourf(X1, X2, Z, levels=25, cmap="viridis", alpha=0.85)
cb = fig.colorbar(cp, ax=ax)
cb.set_label(r"Scalar Potential $\Phi(x_1, x_2)$", fontsize=13)
ax.quiver(X1_sub, X2_sub, u_norm, v_norm, color='white', scale=25, width=0.005, headwidth=4, alpha=0.9, label=r"$\nabla \Phi$ (Steepest Ascent)")
ax.set_title(r"Multivariate Calculus: Scalar Field & Gradient Field $\nabla \Phi(\mathbf{x})$", fontsize=15)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.legend(loc="lower right", facecolor="black", labelcolor="white")
fig.tight_layout()
fig.savefig(OUT / "gradient_vector_field.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. ODE Damped Oscillator & Phase Portrait
t = np.linspace(0, 10, 1000)

# Underdamped, critically damped, overdamped
omega0 = 2.0
under_gamma = 0.4
under_wd = np.sqrt(omega0**2 - under_gamma**2)
u_under = np.exp(-under_gamma * t) * np.cos(under_wd * t)
v_under = -under_gamma * u_under - under_wd * np.exp(-under_gamma * t) * np.sin(under_wd * t)

crit_gamma = 2.0
u_crit = (1.0 + crit_gamma * t) * np.exp(-crit_gamma * t)
v_crit = crit_gamma * np.exp(-crit_gamma * t) - crit_gamma * u_crit

over_gamma = 3.0
r1 = -over_gamma + np.sqrt(over_gamma**2 - omega0**2)
r2 = -over_gamma - np.sqrt(over_gamma**2 - omega0**2)
c1 = (0.0 - r2) / (r1 - r2)
c2 = 1.0 - c1
u_over = c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(t, u_under, 'b-', lw=2, label="Underdamped ($\gamma < \omega_0$)")
ax1.plot(t, u_crit, 'r--', lw=2, label="Critically Damped ($\gamma = \omega_0$)")
ax1.plot(t, u_over, 'g-.', lw=2, label="Overdamped ($\gamma > \omega_0$)")
ax1.set_title(r"Harmonic Oscillator $m\ddot{u} + c\dot{u} + ku = 0$ in Time", fontsize=14)
ax1.set_xlabel("Time t (s)")
ax1.set_ylabel("Displacement u(t)")
ax1.legend(fontsize=11)

ax2.plot(u_under, v_under, 'b-', lw=1.8, label="Underdamped Spiral")
ax2.plot(u_under[0], v_under[0], 'ro', markersize=8, label="Initial State $(u_0, v_0) = (1, 0)$")
ax2.plot(0, 0, 'kx', markersize=10, markeredgewidth=2.5, label="Equilibrium $(0,0)$")
ax2.set_title(r"Phase Space Trajectory $(u, \dot{u})$", fontsize=14)
ax2.set_xlabel("Displacement u")
ax2.set_ylabel(r"Velocity $\dot{u}$")
ax2.legend(fontsize=11)

fig.suptitle("Ordinary Differential Equations: Damped Harmonic Motion", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "ode_decay_and_phase.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. PDE Heat Equation Space-Time Field
nx, nt = 100, 100
x_pde = np.linspace(0, 1, nx)
t_pde = np.linspace(0, 0.5, nt)
X_pde, T_pde = np.meshgrid(x_pde, t_pde)
alpha = 0.5
U_pde = np.sin(np.pi * X_pde) * np.exp(-alpha * (np.pi**2) * T_pde)

fig, ax = plt.subplots(figsize=(9, 6))
c = ax.contourf(X_pde, T_pde, U_pde, levels=30, cmap="magma")
cb = fig.colorbar(c, ax=ax)
cb.set_label("Temperature u(t, x)", fontsize=13)
ax.set_title(r"1D Heat Equation $\partial_t u = \alpha \partial_{xx} u$ [Dirichlet BCs: $u(t, 0) = u(t, 1) = 0$]", fontsize=14)
ax.set_xlabel("Spatial Coordinate x")
ax.set_ylabel("Temporal Coordinate t (s)")
fig.tight_layout()
fig.savefig(OUT / "pde_heat_space_time.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. Discretization, Error & Residual
h_steps = np.logspace(-3, -0.5, 20)
x_eval = 1.0
f_exact_prime = 3 * x_eval**2 - 3  # for x^3 - 3x at x=1 is 0.0

err_forward = np.abs(((x_eval + h_steps)**3 - 3*(x_eval + h_steps) - (x_eval**3 - 3*x_eval))/h_steps - f_exact_prime)
err_central = np.abs(((x_eval + h_steps)**3 - 3*(x_eval + h_steps) - ((x_eval - h_steps)**3 - 3*(x_eval - h_steps)))/(2*h_steps) - f_exact_prime)

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(h_steps, err_forward, 'ro-', lw=2, label="Forward Difference: $\mathcal{O}(h)$")
ax.loglog(h_steps, err_central, 'bs--', lw=2, label="Central Difference: $\mathcal{O}(h^2)$")
ax.loglog(h_steps, 3 * h_steps, 'k:', alpha=0.5, label="Theoretical $\mathcal{O}(h)$ Reference")
ax.loglog(h_steps, 2 * h_steps**2, 'm:', alpha=0.5, label="Theoretical $\mathcal{O}(h^2)$ Reference")
ax.set_title("Numerical Discretization Error Scaling (Truncation Error)", fontsize=14)
ax.set_xlabel("Discretization Step Size h")
ax.set_ylabel("Absolute Error $|f'(x) - D_h f(x)|$")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "discretization_and_residuals.png")
plt.close(fig)

print("Week 1 plots generated successfully:")
for p in OUT.glob("*.png"):
    print(f"  - {p}")
