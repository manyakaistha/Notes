import numpy as np
rng = np.random.default_rng(42)
for d in (4, 16, 64, 256):
    q = rng.normal(size=(20000, d))
    k = rng.normal(size=(20000, d))
    dot = np.sum(q*k, axis=1)
    print(f"d_k={d:3d} Var(q.k)={np.var(dot):8.2f} target={d:3d} scaled_var={np.var(dot/np.sqrt(d)):.3f}")
