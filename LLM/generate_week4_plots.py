import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
OUT=os.path.join(os.path.dirname(__file__),"week 4","assets")
os.makedirs(OUT,exist_ok=True)
def save(fig,name): fig.savefig(os.path.join(OUT,name),metadata={"Software":"Matplotlib"}); plt.close(fig)
def main():
    fig,ax=plt.subplots(figsize=(10,6)); coords={"":(0,3),"A":(-1,2),"B":(1,2),"AA":(-1.5,1),"AB":(-.5,1),"BA":(.5,1),"BB":(1.5,1)}
    edges=[("","A",.6),("","B",.4),("A","AA",.1),("A","AB",.9),("B","BA",.9),("B","BB",.1)]
    for par,ch,p in edges:
        x1,y1=coords[par]; x2,y2=coords[ch]; ax.plot([x1,x2],[y1,y2],color="gray"); ax.text((x1+x2)/2,(y1+y2)/2+.08,f"{p:g}",fontsize=9)
    for n,(x,y) in coords.items(): ax.scatter(x,y,s=500,color="C0" if n in ("A","AB") else "C1"); ax.text(x,y,n or "start",ha="center",va="center",fontsize=9)
    ax.text(-1.9,.3,"Greedy: A→AB = .54",fontsize=10); ax.text(.35,.3,"Beam k=2: B→BA = .36",fontsize=10)
    ax.set(xlabel="Beam-tree horizontal coordinate (arbitrary)",ylabel="Decode step (token index)",title="Toy conditional tree: greedy (0.54) beats beam's final best (0.36)")
    # change branch probabilities to demonstrate greedy suboptimal actually tree needs B branch .95 => .38 still below .54; fix greedy A=.6, A-> .7=.42; B=.4 B->.99=.396. choose A .5, B .49, A cont .6=.30 B cont .99=.485; greedy A overall .30
    # graph labels above reflects A=.6; patch outcome via the actual data is set below in title and annotation
    ax.clear(); coords={"root":(0,3),"A":(-1,2),"B":(1,2),"AA":(-1.5,1),"AB":(-.5,1),"BA":(.5,1),"BB":(1.5,1)}
    edges=[("root","A",.51),("root","B",.49),("A","AA",.4),("A","AB",.6),("B","BA",.99),("B","BB",.01)]
    for pa,ch,p in edges:
        x1,y1=coords[pa];x2,y2=coords[ch];ax.plot([x1,x2],[y1,y2],color="gray");ax.text((x1+x2)/2,(y1+y2)/2+.08,f"{p:g}",fontsize=9)
    for n,(x,y) in coords.items():ax.scatter(x,y,s=500,color="C0" if n in ("A","AB") else "C1");ax.text(x,y,n if n!="root" else "start",ha="center",va="center",fontsize=9)
    ax.text(-1.8,.3,"Greedy: A→AB = 0.306",fontsize=10);ax.text(.25,.3,"Beam k=2: B→BA = 0.4851",fontsize=10)
    ax.set(xlabel="Beam-tree horizontal coordinate (arbitrary)",ylabel="Decode step (token index)",title="Beam search retains a lower-probability prefix with a stronger continuation");save(fig,"01_greedy_beam_tree.png")

    p=np.array([.42,.25,.15,.10,.08]); order=np.argsort(p)[::-1]; topk=np.zeros(5); topk[order[:3]]=p[order[:3]]; topp=np.zeros(5); cum=0
    for i in order:
        if cum<.8: topp[i]=p[i];cum+=p[i]
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(5);ax.bar(x-.2,p,.2,label="Original");ax.bar(x,topk,.2,label="Top-k (k=3)");ax.bar(x+.2,topp,.2,label="Top-p (p=.80, include crossing token)");ax.set_xticks(x,["A","B","C","D","E"]);ax.set(xlabel="Token candidate",ylabel="Probability mass (dimensionless)",title="Top-k and nucleus truncation before renormalisation");ax.legend(fontsize=8);save(fig,"02_topk_topp.png")

    logits=np.array([2.,1.,0.,-1.]);fig,ax=plt.subplots(figsize=(8,5));
    for t in (.5,1.,2.):
        z=logits/t; q=np.exp(z-z.max());q/=q.sum(); h=-(q*np.log(q)).sum();ax.plot(np.arange(4),q,"o-",label=f"T={t:g}, H={h:.3f} nats")
    ax.set_xticks(range(4),["A","B","C","D"]);ax.set(xlabel="Token candidate",ylabel="Probability (dimensionless)",title="Temperature rescales logits and changes entropy");ax.legend(fontsize=9);save(fig,"03_temperature_entropy.png")

    names=["short","long"]; lp=np.array([-1.0,-1.5]); mean=lp/np.array([1,3]);fig,ax=plt.subplots(figsize=(7,5));x=np.arange(2);ax.bar(x-.18,lp,.36,label="Raw log probability");ax.bar(x+.18,mean,.36,label="Mean log probability");ax.set_xticks(x,names);ax.set(ylabel="Sequence score (log-probability, nats)",xlabel="Candidate sequence",title="Length normalisation can reverse beam ranking");ax.legend();save(fig,"04_length_normalisation.png")

    rng=np.random.default_rng(42); n=1_000_000; selected=rng.random(n)<.15; r=rng.random(n); masked=r<.8; random=(r>=.8)&(r<.9); unchanged=r>=.9
    vals=[selected.mean()*100,(masked&selected).sum()/selected.sum()*100,(random&selected).sum()/selected.sum()*100,(unchanged&selected).sum()/selected.sum()*100]
    fig,ax=plt.subplots(figsize=(8,5));ax.bar(["Selected","[MASK]","Random token","Unchanged"],vals,color=["C0","C1","C2","C3"]);ax.axhline(15,color="gray",ls=":");ax.set(ylabel="Empirical rate (%)",xlabel="MLM selection/replacement category",title=f"Seeded BERT MLM masking sample (N={n:,})");save(fig,"05_bert_masking_statistics.png")

    specs=[("Base",768,12,30522), ("Large",1024,24,30522)]; totals=[]; labels=[]
    for label,h,l,v in specs:
        # Exact encoder with pooler; embeddings include 512 position and 2 type rows; per block QKV/O and FFN + biases + two LayerNorms.
        emb=v*h+512*h+2*h+2*h; block=l*(4*h*h+4*h+2*h*(3072 if label=="Base" else 4096)+(3072 if label=="Base" else 4096)+h+4*h); pool=h*h+h
        totals.append((emb/1e6, (block)/1e6,pool/1e6));labels.append(label)
    fig,ax=plt.subplots(figsize=(8,5));base=np.zeros(2)
    for j,label in enumerate(["Embeddings","Encoder blocks","Pooler"]):
        vals=[q[j] for q in totals];ax.bar(["BERT-base","BERT-large"],vals,bottom=base,label=label);base+=vals
    ax.set(ylabel="Parameters (millions)",xlabel="BERT configuration",title="BERT encoder and pooler parameter breakdown");ax.legend();save(fig,"06_bert_parameter_breakdown.png")
    print(f"[Week 4] MLM empirical: selected={selected.mean()*100:.3f}% (target 15%); N={n:,}")
    print(f"[Week 4] BERT counts: base={sum(totals[0])*1e6:.0f}, large={sum(totals[1])*1e6:.0f}")
    print("[Completed] 6 figures")
if __name__=="__main__":main()
