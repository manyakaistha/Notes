"""Generate deterministic Week 5 ASR foundations figures from week5_lab."""
from pathlib import Path
import hashlib
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import week5_lab as lab

OUT = Path(__file__).parent / "week 5" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({"figure.dpi":150,"savefig.bbox":"tight","font.size":10})

def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/name,dpi=180,facecolor="white"); plt.close(fig)
    print(f"{name}: sha256={hashlib.sha256((OUT/name).read_bytes()).hexdigest()}")

def main():
    # 1: density heights and integrated cell masses use different measures.
    code=np.array([-1.,0.,1.]); edges=np.array([-np.inf,-.5,.5,np.inf]); x=np.linspace(-3.5,3.5,700)
    fig,(ax,mass_ax)=plt.subplots(1,2,figsize=(12,4.7))
    ax.plot(x,[lab.gaussian_pdf(v,0,1) for v in x],lw=2.5)
    colors=sns.color_palette("deep",3)
    masses=[]
    for i in range(3):
        masses.append(lab.gaussian_cell_mass(edges[i],edges[i+1],0,1))
        ax.axvspan(max(edges[i],x.min()),min(edges[i+1],x.max()),alpha=.12,color=colors[i])
        ax.scatter([code[i]],[lab.gaussian_pdf(code[i],0,1)],s=65,color=colors[i],zorder=3)
    ax.set(xlabel="Scalar feature (normalized units)",ylabel="Density (feature$^{-1}$)",title="Gaussian density and VQ cells")
    mass_ax.bar(range(3),masses,color=colors)
    for i,mass in enumerate(masses): mass_ax.text(i,mass+.012,f"{mass:.4f}",ha="center",fontsize=11)
    mass_ax.set(xticks=range(3),xticklabels=["0: c=-1","1: c=0","2: c=1"],ylim=(0,.46),
                xlabel="Codebook index and centroid",ylabel="Integrated probability mass (unitless)",title="Probability of each quantization cell")
    save(fig,"01_continuous_density_vq_cells.png")
    # 2: measured nearest-centroid distortion.
    samples=np.array([-1.4,-.9,-.2,.1,.7,1.3]); ids,q,d=lab.quantize_1d(samples,code)
    fig,ax=plt.subplots(figsize=(8.5,3.8)); ax.scatter(samples,np.zeros_like(samples),s=75,label="Feature samples")
    for value,centroid in zip(samples,q): ax.plot([value,centroid],[0,-.35],color="#999",lw=1)
    ax.scatter(code,np.full(3,-.35),marker="D",s=85,label="Codebook centroids")
    ax.set(yticks=[-.35,0],yticklabels=["centroid","sample"],xlabel="Scalar feature value (normalized units)",title=f"Nearest-centroid quantization; empirical MSE = {d:.4f} feature$^2$")
    ax.legend(fontsize=9); save(fig,"02_vq_assignments_distortion.png")
    # 3: compare transition support/topology.
    matrices=[np.ones((3,3))/3,np.array([[.6,.4,0],[0,.6,.4],[0,0,1.]])]
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,m,title in zip(axs,matrices,["Fully connected (uniform example)","Left-to-right with absorbing final state"]):
        annotations=np.full((3,3),"1/3") if np.allclose(m,np.ones((3,3))/3) else np.array([[f"{v:.1f}" for v in row] for row in m])
        sns.heatmap(m,annot=annotations,fmt="",cmap="Blues",vmin=0,vmax=1,cbar=False,square=True,xticklabels=["1","2","3"],yticklabels=["1","2","3"],ax=ax,linewidths=.5)
        ax.set(title=title,xlabel="Next state",ylabel="Current state")
    fig.suptitle("Topology constrains allowed state transitions",y=1.05); save(fig,"03_hmm_topology_comparison.png")
    # 4: the decoder combines acoustic, pronunciation, and language constraints jointly.
    fig,ax=plt.subplots(figsize=(11,5)); ax.axis("off")
    nodes=[(.12,.56,"Feature frames","continuous observations"),
           (.40,.56,"Acoustic model","state-conditioned likelihoods"),
           (.68,.85,"Pronunciation lexicon","word → phone alternatives"),
           (.68,.18,"Language model","word-sequence prior"),
           (.86,.56,"Joint decoder","word / phone / state search")]
    for x,y,head,sub in nodes:
        ax.text(x,y,head,ha="center",va="center",weight="bold",fontsize=10,
                bbox=dict(boxstyle="round,pad=.45",fc="#eaf1f8",ec="#4c78a8"))
        ax.text(x,y-.10,sub,ha="center",va="center",fontsize=8.5)
    for origin,target in [((.23,.56),(.29,.56)),((.51,.56),(.75,.56)),
                          ((.75,.80),(.85,.62)),((.75,.26),(.85,.49))]:
        ax.annotate("",target,origin,arrowprops=dict(arrowstyle="->",lw=1.6,color="#555"))
    ax.text(.5,.02,"Schematic information flow: word hypotheses are scored together with their admissible acoustic paths.",
            ha="center",fontsize=9)
    ax.set(xlim=(0,1.03),ylim=(-.04,1)); save(fig,"04_recognition_components_schematic.png")
    # 5: pronunciation graph for explicit toy lexicon.
    lex={"to":[("T","UW")],"two":[("T","UW")],"read":[("R","EH","D"),("R","IY","D")],"red":[("R","EH","D")]}
    fig,ax=plt.subplots(figsize=(9,4.4)); ax.axis("off")
    rows=[("to",2.9,["T","UW"]),("two",2.2,["T","UW"]),("read",1.3,["R","EH","D"]),("read",.75,["R","IY","D"]),("red",.2,["R","EH","D"])]
    for word,y,phones in rows:
        ax.text(.07,y,word,ha="left",va="center",weight="bold",fontsize=10)
        xx=np.linspace(.35,.87,len(phones))
        for i,(px,ph) in enumerate(zip(xx,phones)):
            ax.scatter([px],[y],s=680,color="#4c78a8",zorder=3); ax.text(px,y,ph,ha="center",va="center",color="white",weight="bold",fontsize=9)
            if i: ax.plot([xx[i-1]+.035,px-.035],[y,y],color="#555",lw=1.5)
    ax.text(.5,3.35,"Multiple words can share a pronunciation; a word can have variants",ha="center",fontsize=11)
    ax.set(xlim=(0,1),ylim=(-.2,3.7)); save(fig,"05_toy_lexicon_pronunciation_paths.png")
    # 6: count-derived conditional distribution.
    words=lab.VOCAB; histories=["<s>","i","you","eat","apples","bananas"]
    mat=np.array([[lab.bigram_mle(h,w) if lab.HISTORY_COUNTS[h] else 0 for w in words] for h in histories])
    fig,ax=plt.subplots(figsize=(10,5)); sns.heatmap(mat,annot=True,fmt=".2f",cmap="mako",vmin=0,vmax=1,xticklabels=words,yticklabels=histories,cbar_kws={"label":"Bigram MLE probability"},ax=ax)
    ax.set(title="Sentence-bounded bigram MLE from the three-sentence toy corpus",xlabel="Predicted token (BOS excluded)",ylabel="History token")
    save(fig,"06_bigram_mle_heatmap.png")
    # 7: empirical unigram counts and trigram sparsity.
    fig,axs=plt.subplots(1,2,figsize=(12,5.5))
    axs[0].bar(list(lab.UNIGRAM),list(lab.UNIGRAM.values()),color=sns.color_palette("deep"))
    axs[0].set(xlabel="Lexical token",ylabel="Corpus occurrence count (tokens)",title="Observed lexical unigram counts")
    axs[0].tick_params(axis="x",labelrotation=30,labelsize=11)
    histories=sorted(lab.TRIGRAM_HISTORY)
    tr=np.array([[lab.TRIGRAM[(h1,h2,w)] for w in lab.VOCAB] for h1,h2 in histories])
    sns.heatmap(tr,annot=True,fmt=".0f",cmap="rocket_r",xticklabels=lab.VOCAB,
                yticklabels=[f"{h1}, {h2}" for h1,h2 in histories],
                cbar_kws={"label":"Observed trigram count"},ax=axs[1])
    axs[1].set(xlabel="Predicted token",ylabel="Complete two-token history",title="Trigram continuation counts")
    axs[1].tick_params(axis='y',labelsize=10)
    save(fig,"07_ngram_counts_and_sparsity.png")
    # 8: genuine forward evidence and LM reweighting on the two fixed HMM candidates.
    acoustic=lab.acoustic_candidates(); lm={w:lab.bigram_mle("eat",w) for w in acoustic}; combined={w:acoustic[w]*lm[w] for w in acoustic}
    fig,ax=plt.subplots(figsize=(8.5,4.8)); xx=np.arange(2); width=.34
    ax.bar(xx-width/2,[acoustic[w] for w in ("apples","bananas")],width,label="Acoustic forward evidence")
    ax.bar(xx+width/2,[combined[w] for w in ("apples","bananas")],width,label="Evidence × LM prior")
    ax.set(xticks=xx,xticklabels=["apples","bananas"],xlabel="Candidate word",ylabel="Unnormalized score (feature$^{-3}$)",title="A corpus-derived LM prior can change the acoustic ranking")
    ax.legend(fontsize=9); save(fig,"08_acoustic_lm_candidate_scores.png")
    print("Figures use fixed analytic parameters and count-derived corpus values; no fitted metrics or random samples.")

if __name__=="__main__": main()
