"""Deterministic Week 2 figures restricted to the supplied lecture scope."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "week 2" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=180, metadata={"Software": "Matplotlib"})
    plt.close(fig)


def masking():
    f = np.linspace(0, 8000, 1200)
    masker = 0.25 + 0.8 * np.exp(-0.5 * ((f - 1800) / 230) ** 2)
    threshold = 0.08 + 0.36 * np.exp(-0.5 * ((f - 1800) / 700) ** 2)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(f, masker, label="Illustrative masker spectrum")
    ax.plot(f, threshold, "--", label="Illustrative raised detection threshold")
    ax.fill_between(f, 0, threshold, alpha=.17, label="Below local threshold")
    ax.set(xlabel="Frequency (Hz)", ylabel="Relative amplitude (arbitrary units)",
           title="Frequency masking: schematic, not a psychoacoustic calibration")
    ax.legend(fontsize=10)
    save(fig, "01_frequency_masking_schematic.png")


def cepstrum():
    # A voiced harmonic comb shaped by a smooth two-formant vocal-tract envelope.
    fs, n = 16000, 1024
    f = np.fft.rfftfreq(n, 1 / fs)
    envelope = 0.12 + 1.2*np.exp(-.5*((f-700)/150)**2) + .85*np.exp(-.5*((f-1500)/220)**2)
    harmonic = (.5 + .5*np.cos(2*np.pi*f/125))**8
    mag = envelope * harmonic
    c = np.fft.irfft(np.log(np.maximum(mag, 1e-8)), n=n)
    q = np.arange(n) / fs * 1000
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(f, mag); axes[0].set(xlim=(0, 4000), xlabel="Frequency (Hz)", ylabel="Magnitude (a.u.)", title="Synthetic excitation comb × smooth vocal-tract envelope")
    axes[1].plot(q[:n//2], c[:n//2]); axes[1].axvspan(0, 1.4, alpha=.18, color="tab:red", label="Low-quefrency envelope region (illustrative)")
    axes[1].set(xlabel="Quefrency (ms)", ylabel="Real cepstrum (a.u.)", title="Real cepstrum of log magnitude; low-quefrency liftering")
    axes[1].legend(fontsize=9)
    fig.tight_layout(h_pad=2.0)
    save(fig, "02_real_cepstrum_liftering.png")


def mel_mfcc():
    hz = np.linspace(0, 8000, 1025)
    mel = lambda x: 1127*np.log1p(x/700)
    inv = lambda x: 700*np.expm1(x/1127)
    points = inv(np.linspace(mel(0), mel(8000), 28))
    bank = np.zeros((26, hz.size))
    for j in range(26):
        l, c, r = points[j:j+3]
        bank[j] = np.maximum(0, np.minimum((hz-l)/(c-l), (r-hz)/(r-c)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in bank: ax.plot(hz, row, lw=1.1)
    ax.set(xlim=(0,8000), xlabel="Frequency (Hz)", ylabel="Triangular filter gain (unitless)", title="26-channel Mel-spaced triangular analysis bank")
    save(fig, "03_mel_filterbank.png")

    # Deterministic correlated log-band-energy sample; orthonormal DCT-II projection.
    rng = np.random.default_rng(42)
    corr = np.fromfunction(lambda i,j: .82**np.abs(i-j), (26,26))
    x = np.einsum("bi,ji->bj", rng.standard_normal((500,26)), np.linalg.cholesky(corr))
    k, m = np.arange(13)[:,None], np.arange(26)[None,:]
    D = np.sqrt(2/26)*np.cos(np.pi*(m+.5)*k/26); D[0] /= np.sqrt(2)
    z = np.einsum("bi,ki->bk", x, D)
    fig, ax = plt.subplots(1,2,figsize=(10,4.8))
    ticks=np.arange(0,26,3)
    sns.heatmap(np.corrcoef(x,rowvar=False), vmin=-1,vmax=1,cmap="vlag",square=True,ax=ax[0],cbar=True,cbar_kws={"label":"Correlation coefficient"},xticklabels=3,yticklabels=3)
    sns.heatmap(np.corrcoef(z,rowvar=False), vmin=-1,vmax=1,cmap="vlag",square=True,ax=ax[1],cbar=True,cbar_kws={"label":"Correlation coefficient"},xticklabels=2,yticklabels=2)
    ax[0].set(title="Correlated log-band energies",xlabel="Band index",ylabel="Band index")
    ax[1].set(title="DCT coefficients (truncated to 13)",xlabel="Coefficient index",ylabel="Coefficient index")
    fig.suptitle("DCT often reduces correlation; finite-sample residuals remain",y=1.02)
    fig.tight_layout()
    save(fig,"04_dct_covariance_comparison.png")


def gaussian():
    x=np.linspace(-4,7,800)
    def pdf(x,mu,var): return np.exp(-.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
    a,b=pdf(x,0,1),pdf(x,2,1.5**2)
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot(x,a,label=r"Class A: $N(0,1)$")
    ax.plot(x,b,label=r"Class B: $N(2,1.5^2)$")
    ax.fill_between(x,0,np.minimum(a,b),alpha=.25,label="Density overlap (not a probability by itself)")
    ax.set(xlabel="Scalar observation $x$ (arbitrary feature units)",ylabel="Probability density (feature-unit$^{-1}$)",title="Single-Gaussian class likelihoods and overlap")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig,"05_single_gaussian_classification.png")


def time_frequency():
    fs=16000
    t=np.arange(int(.32*fs))/fs
    # Voiced source with a smooth, time-varying harmonic envelope.
    x=np.sin(2*np.pi*125*t)+.4*np.sin(2*np.pi*250*t)+.18*np.sin(2*np.pi*375*t)
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),constrained_layout=True)
    meshes=[]
    for ax,win_ms in zip(axes,(10,40)):
        n=int(fs*win_ms/1000); hop=int(fs*.005)
        starts=np.arange(0,len(x)-n+1,hop)
        spec=np.stack([np.abs(np.fft.rfft(x[s:s+n]*np.hamming(n),n=1024))**2 for s in starts],axis=1)
        meshes.append(ax.pcolormesh(starts*1000/fs,np.fft.rfftfreq(1024,1/fs),10*np.log10(spec+1e-10),shading="auto",cmap="magma",vmin=-70,vmax=40))
        ax.set(ylim=(0,2000),xlabel="Time (ms)",ylabel="Frequency (Hz)",title=f"{win_ms} ms analysis window")
    fig.colorbar(meshes[-1],ax=axes,label="Power (dB re 1 a.u.$^2$)",shrink=.84)
    save(fig,"06_window_length_spectrogram.png")


def main():
    masking(); cepstrum(); mel_mfcc(); gaussian(); time_frequency()
    print("Generated 6 scoped Week 2 figures in week 2/assets/")


if __name__ == "__main__": main()
