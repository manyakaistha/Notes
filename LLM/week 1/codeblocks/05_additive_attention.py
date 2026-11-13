import numpy as np
rng = np.random.default_rng(42)
enc = rng.normal(size=(4, 3))
dec = rng.normal(size=3)
W, U, v = rng.normal(size=(3, 5)), rng.normal(size=(3, 5)), rng.normal(size=5)
score = np.tanh(enc @ W + dec @ U) @ v
z = score - score.max()
alpha = np.exp(z) / np.exp(z).sum()
context = alpha @ enc
print("alignment scores=" + np.array2string(score, precision=3))
print("alpha=" + np.array2string(alpha, precision=3) + f"; sum={alpha.sum():.6f}")
print("context=" + np.array2string(context, precision=3))
