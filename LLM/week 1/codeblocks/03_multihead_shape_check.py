import torch
import torch.nn as nn

torch.manual_seed(42)
B, T, D, H = 2, 5, 16, 4
x = torch.randn(B, T, D)
layer = nn.MultiheadAttention(D, H, dropout=0.0, batch_first=True)
q, k, v = [x @ layer.in_proj_weight[i*D:(i+1)*D].T + layer.in_proj_bias[i*D:(i+1)*D] for i in range(3)]
dh = D // H
split = lambda a: a.reshape(B, T, H, dh).transpose(1, 2)
qh, kh, vh = map(split, (q, k, v))
weights = torch.softmax(qh @ kh.transpose(-2, -1) / dh**0.5, dim=-1)
heads = (weights @ vh).transpose(1, 2).reshape(B, T, D)
y_manual = heads @ layer.out_proj.weight.T + layer.out_proj.bias
y_torch, _ = layer(x, x, x, need_weights=False)
print(f"input [B,T,d_model]=[{B},{T},{D}]")
print(f"heads [B,h,T,d_k]=[{B},{H},{T},{dh}]; concat={list(heads.shape)}")
print(f"max |manual - torch.nn.MultiheadAttention|={(y_manual-y_torch).abs().max().item():.8f}")
