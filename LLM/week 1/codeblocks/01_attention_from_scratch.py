import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(42)
torch.manual_seed(42)
q = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
k = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
v = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)
scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
z = scores - scores.max(axis=-1, keepdims=True)
a = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
y = a @ v
yt = F.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v))
print(f"attention shape={y.shape}; row-sum range={a.sum(-1).min():.6f}..{a.sum(-1).max():.6f}")
print(f"max |NumPy - torch SDPA|={np.max(np.abs(y-yt.numpy())):.8f}")
