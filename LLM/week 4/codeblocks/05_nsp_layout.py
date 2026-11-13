import numpy as np
rng=np.random.default_rng(42); sentences=[f"sentence-{i}" for i in range(10001)]; labels=np.array([1]*5000+[0]*5000); rng.shuffle(labels); pairs=[]
for i,label in enumerate(labels):
    if label: pairs.append((sentences[i],sentences[i+1]))
    else:
        j=int(rng.integers(0,len(sentences)))
        while j in (i,i+1): j=int(rng.integers(0,len(sentences)))
        pairs.append((sentences[i],sentences[j]))
print(f"constructed consecutive={np.mean(labels==1):.1%}; random-next={np.mean(labels==0):.1%}")
tokens=['[CLS]','A1','A2','[SEP]','B1','B2','[SEP]']; segments=[0,0,0,0,1,1,1]
print('tokens: '+ ' '.join(tokens));print('segment ids: '+ ' '.join(map(str,segments)))
