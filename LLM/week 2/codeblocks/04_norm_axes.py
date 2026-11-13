import torch
import torch.nn as nn

torch.manual_seed(42)
x=torch.tensor([[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]]) # [B,T,d]
ln=nn.LayerNorm(3,elementwise_affine=False)(x)
# BatchNorm1d on [B,C,T] normalizes each feature across B and T.
bn=nn.BatchNorm1d(3,affine=False,track_running_stats=False)(x.transpose(1,2)).transpose(1,2)
short=x[:1,:1,:]
print(f"[B,T,d]={list(x.shape)}; LN per-token means={ln.mean(-1)}")
print(f"BN changes with batch/padding population: feature-0 normalized values={bn[:,:,0]}")
print(f"single token LN output={nn.LayerNorm(3,elementwise_affine=False)(short).flatten()}")
# With one value per channel, training BatchNorm cannot estimate variance.
try:
    nn.BatchNorm1d(3)(torch.ones(1,3,1))
except ValueError as exc:
    print(f"BN on [1,3,1]: {str(exc).split('.')[0]}.")
