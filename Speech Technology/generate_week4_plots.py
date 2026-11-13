"""Generate deterministic Week 4 HMM/HMM-GMM figures from the lab model."""
from pathlib import Path
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import week4_lab as lab

OUT = Path(__file__).parent / 'week 4' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='talk', palette='deep')
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight', 'font.size': 10})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, facecolor='white')
    plt.close(fig)
    print(f"{name}: sha256={hashlib.sha256((OUT/name).read_bytes()).hexdigest()}")

def main():
    states, obs = lab.STATES, lab.OBS
    path, delta, back, best = lab.viterbi(lab.PI, lab.A, lab.B, obs)
    alpha,beta,gamma,xi,filt,ll = lab.forward_backward(lab.PI,lab.A,lab.B,obs)
    records=lab.enumerate_paths(lab.PI,lab.A,lab.B,obs)
    # 1. Dynamic Bayesian network / HMM conditional-independence structure.
    fig, ax=plt.subplots(figsize=(9,3.2)); ax.axis('off')
    xs=[.15,.5,.85]; yq=.68; yo=.24
    for t,x in enumerate(xs, start=1):
        ax.scatter([x],[yq],s=1800,c='#4C78A8',zorder=3); ax.text(x,yq,f'$Q_{t}$',ha='center',va='center',color='white',weight='bold')
        ax.scatter([x],[yo],s=1800,c='#F58518',zorder=3); ax.text(x,yo,f'$O_{t}$',ha='center',va='center',color='white',weight='bold')
        ax.annotate('',(x,yo+.09),(x,yq-.09),arrowprops={'arrowstyle':'->','lw':1.8,'color':'#555'})
        if t > 1:
            ax.annotate('',(x-.075,yq),(xs[t-2]+.075,yq),arrowprops={'arrowstyle':'->','lw':1.8,'color':'#555'})
    ax.text(.5,.02,'State transition: $P(Q_t|Q_{t-1})$     Emission: $P(O_t|Q_t)$',ha='center',fontsize=11)
    ax.set(xlim=(0,1),ylim=(-.04,.94)); save(fig,'01_hmm_dependency_graph.png')
    # 2/3. Transition/emission parameters.
    for data, title, fname, cols in [(lab.A,'Transition probabilities $P(Q_t|Q_{t-1})$','02_transition_matrix.png',states),(lab.B,'Emission probabilities $P(O_t|Q_t)$','03_emission_matrix.png',lab.SYMBOLS)]:
        fig,ax=plt.subplots(figsize=(8,4.8)); sns.heatmap(data,annot=True,fmt='.3f',cmap='Blues',vmin=0,vmax=1,xticklabels=cols,yticklabels=states,linewidths=.5,cbar_kws={'label':'Probability'},ax=ax)
        fig.tight_layout()
        ax.set(title=title,xlabel='Next state' if data.shape[1]==3 else 'Observed symbol',ylabel='Current hidden state')
        save(fig,fname)
    # 4. Viterbi trellis scores and selected predecessor edges.
    fig,ax=plt.subplots(figsize=(8,5)); ts=np.arange(1,4)
    for s in range(3):
        ax.plot(ts,delta[:,s],marker='o',lw=2,label=states[s])
    for t in range(1,3):
        for j in range(3):
            i=back[t,j]
            ax.annotate('',(t+1,delta[t,j]),(t,delta[t-1,i]),arrowprops={'arrowstyle':'->','color':'#555','alpha':.55,'lw':1})
    ax.plot(ts,delta[np.arange(3),path],color='black',lw=3.3,marker='*',ms=12,label='Viterbi path')
    ax.set(xticks=ts,xticklabels=['Dry','Dry','Soggy'],xlabel=r'Observation $o_t$',ylabel=r'Best joint log probability $\log\delta_t(j)$',title='Viterbi dynamic program and recovered path'); ax.legend(ncol=2,fontsize=9)
    save(fig,'04_viterbi_trellis.png')
    # 5. Forward recursion state-mass by time.
    fig,ax=plt.subplots(figsize=(7.5,4.5)); sns.heatmap(np.exp(alpha),annot=True,fmt='.5f',cmap='YlGnBu',xticklabels=states,yticklabels=['t=1 Dry','t=2 Dry','t=3 Soggy'],cbar_kws={'label':'Joint prefix mass'},ax=ax)
    ax.set(title=r'Forward recursion: $\alpha_t(j)=P(o_{1:t},q_t=j)$',xlabel='Current hidden state',ylabel='Observed prefix')
    save(fig,'05_forward_prefix_mass.png')
    # 6. Filtering versus smoothing.
    fig,axs=plt.subplots(1,2,figsize=(11,4.6))
    for ax,data,title in [(axs[0],filt,r'Filtering $P(q_t|o_{1:t})$'),(axs[1],gamma,r'Smoothing $P(q_t|o_{1:T})$')]:
        sns.heatmap(data,annot=True,fmt='.3f',cmap='mako',vmin=0,vmax=1,xticklabels=states,yticklabels=['t=1','t=2','t=3'],cbar=False,ax=ax); ax.set(title=title,xlabel='Hidden state',ylabel='Time')
    fig.suptitle('Causal state estimate versus full-sequence posterior',y=1.03); save(fig,'06_filtering_vs_smoothing.png')
    # 7. Exact mass allocation across all 27 hidden paths.
    ranked=sorted(records,key=lambda r:r[1],reverse=True)
    fig,ax=plt.subplots(figsize=(9,4.8)); probs=np.array([r[1] for r in ranked]); ax.bar(np.arange(1,len(probs)+1),probs,color=['#E45756']+['#4C78A8']*26)
    ax.axhline(probs.sum(),color='#54A24B',ls='--',label=f'All-path evidence = {probs.sum():.4f}')
    ax.set(xlabel='Hidden-state path rank (descending joint mass)',ylabel='$P(Q,O)$',title='The best path is one term in the observation evidence'); ax.legend(fontsize=9)
    save(fig,'07_path_mass_and_evidence.png')
    # 8. Actual GMM emission densities and real feature sequence used for decoding.
    w,m,v,x,g,p,d,bk,bg,gg,xx,ff,llg=lab.hmm_gmm_demo()
    fig,(ax,alignment)=plt.subplots(2,1,figsize=(10,8)); grid=np.linspace(-2.2,2.0,500)
    for s in range(3):
        ax.plot(grid,lab.gaussian_mixture_pdf(grid,w[s],m[s],v[s]),lw=2,label=f'{lab.SPEECH_STATES[s]} density')
        ax.scatter(x,g[s],s=30,alpha=.55)
    for i,val in enumerate(x): ax.axvline(val,color='#777',alpha=.14,lw=.8)
    ax.set(xlabel='Normalized log-energy feature (dimensionless)',ylabel='Emission density (feature$^{-1}$)',title='State-conditional GMM likelihoods at decoded feature frames'); ax.legend(fontsize=9)
    alignment.step(np.arange(1,len(p)+1),p,where='mid',color='#4C78A8',lw=2)
    alignment.scatter(np.arange(1,len(p)+1),p,s=55,color='#4C78A8')
    alignment.set(xticks=np.arange(1,len(p)+1),yticks=range(3),yticklabels=lab.SPEECH_STATES,
                  xlabel='Acoustic frame index',ylabel='Decoded state (categorical)',
                  title='Computed left-to-right Viterbi alignment (synthetic features)')
    alignment.set_ylim(-.4,2.4)
    save(fig,'08_hmm_gmm_emissions.png')
    print(f"Discrete model: evidence={np.exp(ll):.9f}; best joint={np.exp(best):.9f}; posterior={np.exp(best-ll):.9f}")
    print(f"HMM-GMM model: frames={len(x)}; log evidence={llg:.6f}; path={'-'.join(lab.SPEECH_STATES[i] for i in p)}")

if __name__=='__main__': main()
