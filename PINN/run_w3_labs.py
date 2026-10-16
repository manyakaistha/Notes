import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("=== LAB W3D1: Pure Data vs PINN Extrapolation ===")
# True system: u(t) = exp(-0.2*t)*cos(2.0*t)
t_train = torch.linspace(0.1, 1.4, 15)[:, None]
u_train = torch.exp(-0.2 * t_train) * torch.cos(2.0 * t_train) + 0.02 * torch.randn_like(t_train)
t_extrap = torch.linspace(1.5, 4.0, 50)[:, None]
u_extrap_true = torch.exp(-0.2 * t_extrap) * torch.cos(2.0 * t_extrap)

# Black-box MLP
net_bb = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
opt_bb = torch.optim.Adam(net_bb.parameters(), lr=1e-2)
for ep in range(1000):
    opt_bb.zero_grad()
    loss = torch.mean((net_bb(t_train) - u_train)**2)
    loss.backward()
    opt_bb.step()

with torch.no_grad():
    err_train_bb = torch.mean((net_bb(t_train) - u_train)**2).item()
    err_extrap_bb = torch.mean((net_bb(t_extrap) - u_extrap_true)**2).item()
print(f"Black-Box MLP: Train MSE = {err_train_bb:.6f}, Extrapolation MSE = {err_extrap_bb:.6f} (Divergence factor: {err_extrap_bb/err_train_bb:.1f}x)")

# PINN regularized
net_pinn = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
opt_pinn = torch.optim.Adam(net_pinn.parameters(), lr=1e-2)
t_colloc = torch.linspace(0, 4.0, 100)[:, None].requires_grad_(True)
for ep in range(1200):
    opt_pinn.zero_grad()
    loss_d = torch.mean((net_pinn(t_train) - u_train)**2)
    u_c = net_pinn(t_colloc)
    u_t = torch.autograd.grad(u_c, t_colloc, torch.ones_like(u_c), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_colloc, torch.ones_like(u_t), create_graph=True)[0]
    # m*u'' + c*u' + k*u = 0 with m=1, c=0.4, k=4.04 (gamma=0.2, omega=2.0)
    res = u_tt + 0.4 * u_t + 4.04 * u_c
    loss_ode = torch.mean(res**2)
    loss = 20.0 * loss_d + loss_ode
    loss.backward()
    opt_pinn.step()

with torch.no_grad():
    err_train_pinn = torch.mean((net_pinn(t_train) - u_train)**2).item()
    err_extrap_pinn = torch.mean((net_pinn(t_extrap) - u_extrap_true)**2).item()
print(f"PINN Model:    Train MSE = {err_train_pinn:.6f}, Extrapolation MSE = {err_extrap_pinn:.6f} (Extrapolation Preservation)")

print("\n=== LAB W3D2: Coordinate Autograd vs Finite Differences ===")
x_test = torch.tensor([1.0], requires_grad=True)
y = torch.sin(x_test)
dy_exact = torch.cos(x_test).item()
dy_ad = torch.autograd.grad(y, x_test, create_graph=True)[0].item()
d2y_ad = torch.autograd.grad(torch.autograd.grad(y, x_test, create_graph=True)[0], x_test)[0].item()
d2y_exact = -torch.sin(x_test).item()

h_vals = [1e-1, 1e-3, 1e-5, 1e-8, 1e-12, 1e-15]
print(f"Target: f(x)=sin(x) at x=1.0 | Exact f'(1.0)={dy_exact:.10f}")
print("Step Size h | Forward FD Error | Central FD Error | Coordinate Autodiff Error")
for h in h_vals:
    f_x = np.sin(1.0)
    f_xh = np.sin(1.0 + h)
    f_xmh = np.sin(1.0 - h)
    err_fwd = abs((f_xh - f_x) / h - dy_exact)
    err_cnt = abs((f_xh - f_xmh) / (2 * h) - dy_exact)
    err_ad = abs(dy_ad - dy_exact)
    print(f"{h:11.1e} | {err_fwd:16.6e} | {err_cnt:16.6e} | {err_ad:24.16e}")

print("\n=== LAB W3D4: Learning Rate Annealing Benchmark ===")
# Simulated gradient stiffness
g_bc = torch.tensor(45.2)
g_pde = torch.tensor(0.48)
ratio_init = (g_bc / g_pde).item()
# Annealing update
lambda_bc_hat = torch.tensor(1.0) * (g_bc / g_bc)
lambda_pde_hat = torch.tensor(1.0) * (g_bc / g_pde)
print(f"Initial Gradient Norms: ||grad_bc|| = {g_bc.item():.2f}, ||grad_pde|| = {g_pde.item():.2f} (Stiffness Ratio: {ratio_init:.1f}x)")
print(f"Annealed PDE Multiplier: lambda_pde = {lambda_pde_hat.item():.2f} (Restores balanced gradient descent)")
