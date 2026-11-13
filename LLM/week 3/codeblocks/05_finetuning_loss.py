import torch
import torch.nn.functional as F
weights=torch.tensor([1.0,-2.0,0.5]); task_loss=torch.tensor(0.7); lam=0.1
l2=(weights**2).sum(); l1=weights.abs().sum(); total=task_loss+0.5*l2+lam*l1
print(f"task={task_loss.item():.3f}; L2 sumsq={l2.item():.3f}; L1={l1.item():.3f}")
print(f"composed loss task + 0.5*L2 + {lam}*L1 = {total.item():.3f}")
