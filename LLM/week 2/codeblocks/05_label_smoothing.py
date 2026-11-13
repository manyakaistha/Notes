import torch
import torch.nn.functional as F

torch.manual_seed(42)
logits=torch.tensor([[0.2,-0.3,1.4,0.1]])
target=torch.tensor([2]); eps=.1; classes=4
prob=torch.softmax(logits,dim=-1)
y=torch.full_like(logits,eps/classes); y[0,target.item()]+=1-eps
manual=-(y*torch.log_softmax(logits,dim=-1)).sum(-1).mean()
torch_loss=F.cross_entropy(logits,target,label_smoothing=eps)
print(f"smoothed target={y.flatten().tolist()}")
print(f"manual loss={manual.item():.8f}; torch loss={torch_loss.item():.8f}; equal={torch.allclose(manual,torch_loss)}")
