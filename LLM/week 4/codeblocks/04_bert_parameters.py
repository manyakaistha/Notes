import torch.nn as nn
class Bert(nn.Module):
    def __init__(self,V,H,L,F,maxlen=512):
        super().__init__(); self.word=nn.Embedding(V,H); self.pos=nn.Embedding(maxlen,H); self.seg=nn.Embedding(2,H); self.embed_norm=nn.LayerNorm(H)
        self.blocks=nn.ModuleList([nn.ModuleDict({'q':nn.Linear(H,H),'k':nn.Linear(H,H),'v':nn.Linear(H,H),'o':nn.Linear(H,H),'ff1':nn.Linear(H,F),'ff2':nn.Linear(F,H),'n1':nn.LayerNorm(H),'n2':nn.LayerNorm(H)}) for _ in range(L)])
        self.pool=nn.Linear(H,H)
def formula(V,H,L,F):
    emb=V*H+512*H+2*H+2*H; block=L*(4*H*H+4*H+2*H*F+F+H+4*H); pool=H*H+H
    return emb+block+pool
for name,H,L,A,F in (("base",768,12,12,3072),("large",1024,24,16,4096)):
    configured=Bert(30522,H,L,F); paper_vocab=Bert(30000,H,L,F)
    count=sum(p.numel() for p in configured.parameters())
    print(f"BERT-{name}: paper-vocab-30000={formula(30000,H,L,F):,}; configured-vocab-30522 formula={formula(30522,H,L,F):,}; torch={count:,}; paper~{110 if name=='base' else 340}M")
