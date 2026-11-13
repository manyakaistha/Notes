import torch

torch.manual_seed(42)
scores = torch.tensor([[1.2,.5,2.1,-.3],[.2,1.,.8,1.1],[.4,-.2,1.4,.7],[.6,.3,-.1,1.3]])
mask = torch.triu(torch.ones(4,4,dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
assert torch.all(weights.masked_select(mask) == 0)
print("causal mask (1=blocked):\n", mask.int())
print("masked attention weights:\n", weights)
print(f"future weight sum={weights.masked_select(mask).sum().item():.1f}")
