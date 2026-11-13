import torch.nn as nn
V,D,P,L,F=40478,768,512,12,3072
class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.qkv=nn.Linear(D,3*D); self.proj=nn.Linear(D,D); self.fc1=nn.Linear(D,F); self.fc2=nn.Linear(F,D); self.n1=nn.LayerNorm(D); self.n2=nn.LayerNorm(D)
class GPT(nn.Module):
    def __init__(self):
        super().__init__(); self.tok=nn.Embedding(V,D); self.pos=nn.Embedding(P,D); self.blocks=nn.ModuleList([Block() for _ in range(L)])
        self.lm=nn.Linear(D,V,bias=False); self.lm.weight=self.tok.weight
m=GPT(); total=sum(p.numel() for p in m.parameters())
emb=V*D+P*D; att=L*(4*D*D+4*D); ffn=L*(2*D*F+F+D); ln=L*4*D
print(f"formula tied total={emb+att+ffn+ln:,}; torch tied total={total:,}")
print(f"embeddings={emb:,}; attention={att:,}; FFN={ffn:,}; LayerNorm={ln:,}")
print("Convention: tied token/LM weights, QKV/output and FFN biases, two affine LayerNorms per block; no LM-head bias.")
