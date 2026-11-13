import numpy as np
rng=np.random.default_rng(42)
def apply_mlm(ids,vocab_size,rng):
    selected=rng.random(len(ids))<.15; draw=rng.random(len(ids)); corrupted=ids.copy(); action=np.full(len(ids),"unselected",dtype=object)
    mask=selected & (draw<.8); rand=selected & (draw>=.8)&(draw<.9); same=selected & (draw>=.9)
    corrupted[mask]=vocab_size; corrupted[rand]=rng.integers(0,vocab_size,size=rand.sum()); action[mask]="mask"; action[rand]="random"; action[same]="unchanged"
    return corrupted,ids.copy(),selected,action
n=1_000_000; ids=np.arange(n)%30522; _,_,selected,action=apply_mlm(ids,30522,rng)
print(f"selected={selected.mean():.4%} (target 15%)")
for k in ("mask","random","unchanged"):
    print(f"{k} among selected={(action==k).sum()/selected.sum():.4%}")
