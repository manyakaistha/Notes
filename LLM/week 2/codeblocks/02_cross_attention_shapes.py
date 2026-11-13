import torch
import torch.nn as nn

torch.manual_seed(42)
B, Tdec, Tenc, D, H = 2, 3, 5, 16, 4
q, kv = torch.randn(B,Tdec,D), torch.randn(B,Tenc,D)
layer = nn.MultiheadAttention(D,H,batch_first=True,dropout=0.0)
y, a = layer(q,kv,kv,need_weights=True,average_attn_weights=False)
print(f"Q=[B,T_dec,d]=[{B},{Tdec},{D}], K=V=[B,T_enc,d]=[{B},{Tenc},{D}]")
print(f"weights [B,h,T_dec,T_enc]={list(a.shape)}; output={list(y.shape)}")
print("cross-attention has no causal mask over source positions")
