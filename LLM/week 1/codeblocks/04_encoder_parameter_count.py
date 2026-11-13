import torch.nn as nn

d, h, ff = 512, 8, 2048
# Formula matches PyTorch's combined QKV projection, biases, output projection,
# two FFN biases, and two affine LayerNorms (eps=1e-5, post-norm).
mha = 4*d*d + 4*d
ffn = 2*d*ff + ff + d
norms = 4*d
formula = mha + ffn + norms
layer = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=ff, dropout=0.0, batch_first=True, norm_first=False, bias=True)
torch_count = sum(p.numel() for p in layer.parameters())
print(f"MHA formula={mha:,}; FFN formula={ffn:,}; two LayerNorms={norms:,}")
print(f"encoder-layer formula={formula:,}; torch count={torch_count:,}; equal={formula == torch_count}")
print("Course shorthand: about 3.15M with FFN biases; original paper's base layer count is approximate.")
