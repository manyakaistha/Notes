import torch.nn as nn

d, ff, h = 512, 2048, 8
# Two attention modules (self + cross), FFN, and three affine LayerNorms.
attention = 2*(4*d*d+4*d)
feedforward = 2*d*ff+ff+d
norms = 6*d
formula=attention+feedforward+norms
layer=nn.TransformerDecoderLayer(d_model=d,nhead=h,dim_feedforward=ff,dropout=0.0,batch_first=True,norm_first=False,bias=True)
torch_count=sum(p.numel() for p in layer.parameters())
print(f"self+cross attention={attention:,}; FFN={feedforward:,}; 3 LayerNorms={norms:,}")
print(f"decoder layer formula={formula:,}; torch count={torch_count:,}; equal={formula==torch_count}")
print("Course shorthand ~4M; exact biased PyTorch module is 4.204M.")
