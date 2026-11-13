import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)
class Causal(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(20,8); self.out=nn.Linear(8,20)
    def forward(self,x):
        h=self.emb(x); n=x.size(1); mask=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
        a=h @ h.transpose(1,2) / 8**.5; a=a.masked_fill(mask,-torch.inf)
        return self.out(torch.softmax(a,dim=-1) @ h)
m=Causal().eval(); a=torch.tensor([[1,2,3,4]]); b=torch.tensor([[1,2,9,8]])
la,lb=m(a),m(b); print(f"future-change max |logit diff| at t=1: {(la[:,0]-lb[:,0]).abs().max().item():.8f}")
logits=torch.tensor([[[2.,0.,-1.],[0.,1.,2.],[1.,0.,2.]]]); target=torch.tensor([[0,1,2]])
loss=F.cross_entropy(logits[:,:-1].reshape(-1,3),target[:,1:].reshape(-1))
print(f"shifted next-token CE over positions 1..2: {loss.item():.6f}")
