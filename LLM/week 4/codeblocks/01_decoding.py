import numpy as np
T={():{'A':.51,'B':.49},('A',):{'a':.4,'b':.6},('B',):{'a':.99,'b':.01}}
def greedy():
    s=(); out=[]
    for _ in range(2):
        t=max(T[s],key=T[s].get);out.append(t);s+= (t,)
    return out

def beam(k=2,alpha=0):
    beams=[((),0.)]
    for _ in range(2):
        cand=[(s+(t,),lp+np.log(p)) for s,lp in beams for t,p in T[s].items()]
        beams=sorted(cand,key=lambda z:z[1]/len(z[0])**alpha,reverse=True)[:k]
    return [(s,round(lp,4),round(lp/len(s)**alpha,4)) for s,lp in beams]
print(f"greedy={greedy()}, probability=.51*.6={.51*.6:.4f}")
print(f"beam k=2 (seq, logP, normalised alpha=0): {beam(2)}")
print(f"beam k=2, alpha=1: {beam(2,1)}")
