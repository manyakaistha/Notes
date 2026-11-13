import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

np.random.seed(42)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
OUT = os.path.join(os.path.dirname(__file__), "week 3", "assets")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), metadata={"Software": "Matplotlib"})
    plt.close(fig)

def main():
    x = np.linspace(0, 5, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.exp(x), label=r"$\mathrm{PPL}=e^{\mathrm{CE}}$")
    ces = np.array([0.5, 1., 2., 3.])
    ax.scatter(ces, np.exp(ces), color="C1", zorder=3, label="Example models")
    for c in ces: ax.annotate(f"CE={c:g}, PPL={np.exp(c):.2f}", (c, np.exp(c)), xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Cross-entropy CE (nats/token)", ylabel="Perplexity (dimensionless)", title="Perplexity is the exponential of mean cross-entropy"); ax.legend(); fig.tight_layout(); save(fig,"01_perplexity_ce.png")

    seq = ["<s>","I","like","tea"]
    full = [0.4,0.6,0.5]; bigram=[0.4,0.3,0.2]
    fig, ax=plt.subplots(figsize=(9,5)); pos=np.arange(3)
    ax.bar(pos-.18,-np.log(full),.36,label="Full history"); ax.bar(pos+.18,-np.log(bigram),.36,label="Bigram")
    ax.set_xticks(pos,["I | <s>","like | <s>, I","tea | <s>, I, like"]); ax.set(ylabel="Token NLL (nats)",xlabel="Conditional prediction",title="Chain rule decomposes sentence negative log-likelihood"); ax.legend(); fig.tight_layout(); save(fig,"02_chain_rule.png")

    tokens=np.arange(5); mat=np.triu(np.ones((5,5)),1)
    fig,ax=plt.subplots(figsize=(7,6)); sns.heatmap(mat,mask=mat==0,cmap="Reds",cbar=False,square=True,linewidths=1,ax=ax)
    ax.set_xticklabels(["t1","t2","t3","t4","t5"]); ax.set_yticklabels(["t1","t2","t3","t4","t5"],rotation=0); ax.set(xlabel="Key position (token index)",ylabel="Query position (token index)",title="Causal mask blocks future keys; shifted targets train all positions")
    save(fig,"03_causal_mask.png")

    # GPT-1 standard config, independent biases: token+position embeddings; per-layer QKV/O, FFN and LN.
    d,v,p,L,f=768,40478,512,12,3072
    emb=v*d+p*d; att=L*(4*d*d+4*d); ffn=L*(2*d*f+f+d); ln=L*4*d
    values=np.array([emb,att,ffn,ln])/1e6
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(["Embeddings","Attention","FFN","LayerNorm"],values)
    for i,z in enumerate(values): ax.text(i,z+.5,f"{z:.2f}M",ha="center",fontsize=10)
    ax.set(ylabel="Parameters (millions)",xlabel="GPT-1 component",title=f"GPT-1 explicit count: {values.sum():.2f}M (untied LM head excluded)"); fig.tight_layout(); save(fig,"04_gpt1_parameters.png")

    fig,ax=plt.subplots(figsize=(12,4)); ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis('off')
    rows=[("Classification",["start","sentence","extract"]),("Entailment",["start","premise","delim","hypothesis","extract"]),("Similarity",["start","text A","delim","text B","extract"]),("Multiple choice",["start","context+question","delim","option i","extract"])]
    for y,(label,parts) in enumerate(rows):
        yy=4-y; ax.text(.05,yy+.23,label,ha="left",va="center",fontsize=11)
        xx=2.25; width=9.4/len(parts)
        for part in parts:
            ax.add_patch(Rectangle((xx,yy),width-.06,.48,facecolor=("#f4a261" if part in ("start","delim","extract") else "#8ecae6"),edgecolor="black")); ax.text(xx+(width-.06)/2,yy+.24,part,ha="center",va="center",fontsize=9); xx+=width
    ax.set_title("GPT-1 task input transformations (orange = special token role)"); ax.set_xlabel("Serialized token sequence (position index →)"); save(fig,"05_task_input_layouts.png")

    print(f"[Week 3] GPT1 formula parts: embeddings={emb:,}, attention={att:,}, FFN={ffn:,}, LayerNorm={ln:,}, total={emb+att+ffn+ln:,}")
    print("[Completed] 5 figures")
if __name__ == "__main__": main()
