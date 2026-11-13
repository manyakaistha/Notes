import numpy as np
rng=np.random.default_rng(42); p=np.array([.42,.25,.15,.10,.08]); names=np.array(list("ABCDE"))
def truncated_topk(p,k):
    ix=np.argsort(p)[::-1][:k]; q=np.zeros_like(p);q[ix]=p[ix];return q/q.sum()
def top_p(p,threshold):
    ix=np.argsort(p)[::-1]; cum=np.cumsum(p[ix]); keep=ix[:np.searchsorted(cum,threshold,side='left')+1]
    q=np.zeros_like(p);q[keep]=p[keep];return q/q.sum()
def temperature(logits,t):
    z=logits/t; q=np.exp(z-z.max());return q/q.sum()
print(f"top-k k=3: {truncated_topk(p,3).round(4)}")
print(f"top-p p=.80 (include crossing token): {top_p(p,.8).round(4)}")
for t in (.5,1.,2.):
    q=temperature(np.log(p),t); print(f"T={t:g}: entropy={-(q*np.log(q)).sum():.4f}, sample={rng.choice(names,p=q)}")
